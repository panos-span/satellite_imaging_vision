"""
Sentinel-2 Land Cover Prediction with zipfile input support

This script processes Sentinel-2 data for land cover prediction directly from:
- Zipped SAFE files (.zip)
- SAFE directories (.SAFE)
- GeoTIFF files (.tif)

It implements memory-efficient processing for large images and handles the full
Sentinel-2 workflow from extraction to prediction.

Usage:
    python achaia_prediction.py --model_path "F:\\processed_data\experiment_results\experiments\fine_tuning\unet_sentinel2_best.pth" --input_path pred.zip --reference_path GBDA24_ex2_34SEH_ref_data.tif --output_dir "F:\\prediction"
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap
import json
from pathlib import Path
import gc
import time
import zipfile
import tempfile
import shutil
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes=9, in_channels=13, encoder_type="best_performance"):
    """Load the trained UNet model"""
    print(f"Loading model from: {model_path}")
    
    try:
        # Try to import the custom UNet model
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.unet import UNet
    except ImportError:
        print("Warning: Could not import UNet model. Using a basic model structure.")
        # Define a simplified UNet class to at least load the checkpoint
        class UNet(torch.nn.Module):
            def __init__(self, in_channels=13, num_classes=9, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.num_classes = num_classes
                self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(64)
                self.conv2 = torch.nn.Conv2d(64, num_classes, kernel_size=1)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.conv2(x)
                return x
                
            def get_receptive_field_size(self):
                # Default values based on ResNet backbone
                return {"theoretical_rf": 1067, "effective_rf": 711}
    
    # Create model with the same architecture
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_type=encoder_type,
        use_batchnorm=True,
        skip_connections=4,
        freeze_backbone=False  # Not relevant for inference
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Get the receptive field information
    rf_info = model.get_receptive_field_size()
    print(f"Model receptive field: theoretical={rf_info['theoretical_rf']}px, effective={rf_info['effective_rf']}px")
    
    # Set model to evaluation mode
    model.eval()
    return model, rf_info

def extract_sentinel2_zipfile(zip_path, output_dir=None):
    """
    Extract Sentinel-2 zipfile to a temporary directory
    
    Parameters:
    -----------
    zip_path : str
        Path to the Sentinel-2 zipfile
    output_dir : str, optional
        Directory to extract files to. If None, uses a temporary directory.
        
    Returns:
    --------
    safe_dir : str
        Path to the extracted SAFE directory
    temp_dir : str or None
        Path to the temporary directory (if created)
    """
    print(f"Processing Sentinel-2 zipfile: {zip_path}")
    
    # Check if input is a zipfile
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Input file is not a valid zipfile: {zip_path}")
    
    # Create temporary directory if no output dir specified
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="sentinel2_")
        output_dir = temp_dir
        print(f"Created temporary directory: {temp_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Extracting zipfile to {output_dir}...")
        zip_ref.extractall(output_dir)
    
    # Find SAFE directory
    safe_dirs = list(Path(output_dir).glob("*.SAFE"))
    if not safe_dirs:
        # Some zipfiles might contain the SAFE directory in a subdirectory
        safe_dirs = list(Path(output_dir).glob("**/*.SAFE"))
        
    if not safe_dirs:
        if temp_dir:
            shutil.rmtree(temp_dir)
        raise FileNotFoundError(f"No SAFE directory found in zipfile: {zip_path}")
    
    safe_dir = str(safe_dirs[0])
    print(f"Found SAFE directory: {safe_dir}")
    return safe_dir, temp_dir

def find_sentinel2_bands(sentinel2_path):
    """
    Find Sentinel-2 band files using multiple search patterns.
    
    This function handles various Sentinel-2 directory structures including:
    - SAFE directory structure (both L1C and L2A)
    - Extracted band files in GeoTIFF or JP2 format
    - Non-standard directory layouts
    
    Parameters:
    -----------
    sentinel2_path : str
        Path to Sentinel-2 data (SAFE directory or directory with extracted bands)
    
    Returns:
    --------
    band_files : dict
        Dictionary mapping band names to file paths
    metadata : dict
        Metadata about the found bands (resolution groups, etc.)
    """
    print(f"Searching for Sentinel-2 bands in: {sentinel2_path}")
    
    # Define band names to look for
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    # Define resolution groups
    high_res_bands = ['B02', 'B03', 'B04', 'B08']  # 10m
    medium_res_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # 20m
    low_res_bands = ['B01', 'B09', 'B10']  # 60m
    
    # Initialize result dictionary
    band_files = {}
    
    # Case 1: Standard SAFE directory structure
    if os.path.isdir(sentinel2_path) and sentinel2_path.endswith('.SAFE'):
        # Pattern 1: L1C structure
        granule_dirs = list(Path(sentinel2_path).glob('GRANULE/*/'))
        if granule_dirs:
            # Look for IMG_DATA directory and band files
            img_data_paths = []
            for granule in granule_dirs:
                # L1C structure
                img_data = granule / 'IMG_DATA'
                if img_data.exists():
                    img_data_paths.append(img_data)
                
                # L2A structure (with resolution directories)
                r10m = granule / 'IMG_DATA' / 'R10m'
                r20m = granule / 'IMG_DATA' / 'R20m'
                r60m = granule / 'IMG_DATA' / 'R60m'
                
                if r10m.exists():
                    img_data_paths.append(r10m)
                if r20m.exists():
                    img_data_paths.append(r20m)
                if r60m.exists():
                    img_data_paths.append(r60m)
            
            # Search for band files in all IMG_DATA directories
            for img_data in img_data_paths:
                print(f"Searching in {img_data}")
                for band_name in band_names:
                    # Various patterns for band files
                    patterns = [
                        f"*_{band_name}_*.jp2",
                        f"*_{band_name}.jp2",
                        f"*_{band_name}_*.tif",
                        f"*_{band_name}.tif",
                        f"*{band_name}*.jp2",
                        f"*{band_name}*.tif"
                    ]
                    
                    for pattern in patterns:
                        matches = list(img_data.glob(pattern))
                        if matches and band_name not in band_files:
                            band_files[band_name] = str(matches[0])
                            break
    
    # Case 2: Directory with extracted band files
    elif os.path.isdir(sentinel2_path):
        # Look for band files directly in the directory
        for band_name in band_names:
            patterns = [
                os.path.join(sentinel2_path, f"*_{band_name}_*.jp2"),
                os.path.join(sentinel2_path, f"*_{band_name}.jp2"),
                os.path.join(sentinel2_path, f"*_{band_name}_*.tif"),
                os.path.join(sentinel2_path, f"*_{band_name}.tif"),
                os.path.join(sentinel2_path, f"*{band_name}*.jp2"),
                os.path.join(sentinel2_path, f"*{band_name}*.tif")
            ]
            
            for pattern in patterns:
                matches = list(Path(sentinel2_path).glob(pattern))
                if matches and band_name not in band_files:
                    band_files[band_name] = str(matches[0])
                    break
    
    # Case 3: Single GeoTIFF with all bands
    elif os.path.isfile(sentinel2_path) and sentinel2_path.endswith(('.tif', '.tiff')):
        print(f"Input is a single GeoTIFF file, assuming it contains all bands")
        # Return a special indicator that this is a multiband file
        return {"multiband_file": sentinel2_path}, {
            "is_multiband_file": True,
            "high_res_bands": high_res_bands,
            "medium_res_bands": medium_res_bands,
            "low_res_bands": low_res_bands
        }
    
    # Print summary of found bands
    if band_files:
        print(f"Found {len(band_files)} band files:")
        for band, path in band_files.items():
            print(f"  {band}: {os.path.basename(path)}")
    else:
        print("Warning: No band files found!")
        # Do a more exhaustive search to help with debugging
        print("\nSearching for any .jp2 or .tif files:")
        all_jp2 = list(Path(sentinel2_path).glob("**/*.jp2"))
        all_tif = list(Path(sentinel2_path).glob("**/*.tif"))
        
        print(f"  Found {len(all_jp2)} .jp2 files")
        print(f"  Found {len(all_tif)} .tif files")
        
        if all_jp2 or all_tif:
            print("\nSample of found files:")
            for f in (all_jp2 + all_tif)[:5]:
                print(f"  {f}")
    
    return band_files, {
        "high_res_bands": high_res_bands,
        "medium_res_bands": medium_res_bands,
        "low_res_bands": low_res_bands
    }

def simple_pansharpening(low_res_band, target_shape):
    """
    Simple resize-based pansharpening.
    
    Parameters:
    -----------
    low_res_band : numpy.ndarray
        Low resolution band to pansharpen
    target_shape : tuple
        Target shape (height, width)
    
    Returns:
    --------
    high_res_band : numpy.ndarray
        Pansharpened band at target resolution
    """
    # Use PIL for resampling
    img = Image.fromarray(low_res_band.astype(np.float32))
    img_resized = img.resize((target_shape[1], target_shape[0]), resample=Image.BICUBIC)
    return np.array(img_resized)

def process_sentinel2_data(input_path, output_path=None, cleanup_temp=True):
    """
    Process Sentinel-2 data from various formats (zip, SAFE, GeoTIFF)
    
    Parameters:
    -----------
    input_path : str
        Path to input data (zipfile, SAFE directory, or GeoTIFF)
    output_path : str, optional
        Path to save processed multiband GeoTIFF
    cleanup_temp : bool
        Whether to clean up temporary files after processing
        
    Returns:
    --------
    data : numpy.ndarray
        Processed Sentinel-2 data as a multiband array
    meta : dict
        Metadata for the processed data
    temp_dir : str or None
        Path to temporary directory (for cleanup)
    """
    # If output path exists, use it directly
    if output_path and os.path.exists(output_path):
        print(f"Using existing preprocessed data: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(), src.meta.copy(), None
    
    # Variable to store temporary directory if created
    temp_dir = None
    sentinel2_path = input_path
    
    # Check input type
    if input_path.endswith('.zip'):
        # Extract zipfile
        sentinel2_path, temp_dir = extract_sentinel2_zipfile(input_path)
    elif input_path.endswith('.SAFE') and os.path.isdir(input_path):
        # Input is already a SAFE directory
        sentinel2_path = input_path
    elif os.path.isdir(input_path):
        # Input is a directory, assume it contains band files
        sentinel2_path = input_path
    elif input_path.endswith(('.tif', '.tiff')) and os.path.isfile(input_path):
        # Input is a GeoTIFF, load directly
        with rasterio.open(input_path) as src:
            data = src.read()
            meta = src.meta.copy()
            
            # Save a copy if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data)
                print(f"Saved copy to: {output_path}")
            
            return data, meta, temp_dir
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    # Find band files
    band_files, metadata = find_sentinel2_bands(sentinel2_path)
    
    # Handle case where input is already a multiband file
    if "multiband_file" in band_files:
        print(f"Input is already a multiband file: {band_files['multiband_file']}")
        with rasterio.open(band_files["multiband_file"]) as src:
            data = src.read()
            meta = src.meta.copy()
            
            # Save a copy if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data)
                print(f"Saved multiband copy to: {output_path}")
            
            return data, meta, temp_dir
    
    # Check if we found enough band files
    if len(band_files) == 0:
        # Clean up temporary directory if it was created
        if temp_dir and cleanup_temp:
            shutil.rmtree(temp_dir)
        raise ValueError(f"No band files found in {sentinel2_path}")
    
    print(f"Found {len(band_files)} band files. Processing...")
    
    # Step 1: Read high-resolution bands and get target metadata
    high_res_bands = metadata["high_res_bands"]
    high_res_data = {}
    high_res_meta = None
    high_res_shape = None
    
    for band_name in high_res_bands:
        if band_name in band_files:
            with rasterio.open(band_files[band_name]) as src:
                high_res_data[band_name] = src.read(1)
                if high_res_meta is None:
                    high_res_meta = src.meta.copy()
                    high_res_shape = (src.height, src.width)
    
    # If no high-res bands found, use whatever we have
    if not high_res_data:
        print("Warning: No high-resolution bands found. Using first available band for reference.")
        first_band = next(iter(band_files.values()))
        with rasterio.open(first_band) as src:
            high_res_shape = (src.height, src.width)
            high_res_meta = src.meta.copy()
    
    print(f"Target shape for all bands: {high_res_shape}")
    
    # Step 2: Create a dictionary to store all bands (original or pansharpened)
    processed_bands = {}
    
    # Process all bands
    for band_name in tqdm(sorted(band_files.keys()), desc="Processing bands"):
        with rasterio.open(band_files[band_name]) as src:
            band_data = src.read(1)
            band_shape = (src.height, src.width)
            
            # Check if pansharpening is needed
            if band_shape != high_res_shape:
                print(f"Pansharpening {band_name} from {band_shape} to {high_res_shape}")
                processed_bands[band_name] = simple_pansharpening(band_data, high_res_shape)
            else:
                processed_bands[band_name] = band_data
    
    # Step 3: Create a multiband array with all processed bands
    # Order bands properly (standard Sentinel-2 order)
    standard_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    ordered_bands = []
    
    for band_name in standard_order:
        if band_name in processed_bands:
            ordered_bands.append(processed_bands[band_name])
    
    # Create multiband array
    if not ordered_bands:
        # Clean up temporary directory if it was created
        if temp_dir and cleanup_temp:
            shutil.rmtree(temp_dir)
        raise ValueError("No valid bands were processed")
    
    multiband_data = np.stack(ordered_bands)
    print(f"Created multiband array with shape: {multiband_data.shape}")
    
    # Step 4: Create metadata for the multiband image
    multiband_meta = high_res_meta.copy() if high_res_meta else {}
    multiband_meta.update({
        'driver': 'GTiff',
        'height': high_res_shape[0],
        'width': high_res_shape[1],
        'count': len(ordered_bands),
        'dtype': str(multiband_data.dtype)
    })
    
    # Step 5: Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **multiband_meta) as dst:
            for i, band in enumerate(multiband_data):
                dst.write(band, i+1)
                # Add band name to metadata
                dst.set_band_description(i+1, standard_order[i] if i < len(standard_order) else f"Band{i+1}")
        
        print(f"Saved multiband image to: {output_path}")
    
    # Step 6: Clean up temporary directory if requested
    if temp_dir and cleanup_temp:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        temp_dir = None
    
    return multiband_data, multiband_meta, temp_dir

def normalize_data(data, normalizer=None):
    """
    Normalize the data for model input:
    1. Apply normalizer if provided
    2. Fall back to simple standardization if no normalizer
    """
    print("Normalizing input data...")
    
    # Initialize normalized data array
    norm_data = np.zeros_like(data, dtype=np.float32)
    
    if normalizer is not None:
        print(f"Using provided normalizer: {getattr(normalizer, 'method', 'unknown type')}")
        
        # Check for normalize method in normalizer
        if hasattr(normalizer, 'normalize'):
            try:
                norm_data = normalizer.normalize(data)
                return norm_data
            except Exception as e:
                print(f"Error using normalizer: {e}")
                print("Falling back to per-band normalization...")
        
        # Try per-band normalization if available
        if hasattr(normalizer, 'normalize_band'):
            for i in range(data.shape[0]):
                norm_data[i] = normalizer.normalize_band(data[i], i)
            return norm_data
    
    # Simple per-band standardization as fallback
    print("Using simple per-band standardization")
    for i in range(data.shape[0]):
        band = data[i].copy()
        # Check if values are likely to be Sentinel-2 reflectance values (typically 0-10000)
        if np.max(band) > 100:
            band = band / 10000.0
        
        # Get statistics from non-zero pixels (assuming 0 is often nodata)
        mask = band > 0
        if mask.sum() > 0:
            mean = np.mean(band[mask])
            std = np.std(band[mask])
            # Ensure non-zero std
            std = max(std, 1e-6)
        else:
            mean = np.mean(band)
            std = max(np.std(band), 1e-6)
        
        # Apply standardization
        norm_data[i] = (band - mean) / std
    
    return norm_data

def predict_chunk(model, data_chunk, patch_size, overlap, batch_size=4):
    """
    Make predictions on a chunk of data using overlapping patches.
    
    Args:
        model: The UNet model
        data_chunk: Normalized input data for the chunk [channels, height, width]
        patch_size: Size of patches to process
        overlap: Size of overlap between patches
        batch_size: Batch size for prediction
        
    Returns:
        Predicted class labels for the chunk
    """
    height, width = data_chunk.shape[1], data_chunk.shape[2]
    n_classes = model.num_classes
    
    # Calculate effective patch size (area that will be kept after removing overlap regions)
    effective_patch = patch_size - 2 * overlap
    
    # Calculate number of patches in height and width dimensions
    n_h = int(np.ceil((height - 2 * overlap) / effective_patch))
    n_w = int(np.ceil((width - 2 * overlap) / effective_patch))
    
    # Pad the chunk to fit exactly n_h x n_w patches
    pad_h = n_h * effective_patch + 2 * overlap - height
    pad_w = n_w * effective_patch + 2 * overlap - width
    
    if pad_h > 0 or pad_w > 0:
        padded_data = np.pad(data_chunk, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded_data = data_chunk
        
    # Create prediction array for accumulating class probabilities
    padded_height, padded_width = padded_data.shape[1], padded_data.shape[2]
    class_probs = np.zeros((n_classes, padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)
    
    # Create importance mask that gives higher weight to the central region of each patch
    importance = np.ones((patch_size, patch_size), dtype=np.float32)
    y, x = np.ogrid[:patch_size, :patch_size]
    center_y, center_x = patch_size / 2, patch_size / 2
    # Create a radial mask that decreases weight towards the edges
    mask = (1 - 0.5 * np.sqrt(((y - center_y) / (patch_size / 2)) ** 2 + 
                           ((x - center_x) / (patch_size / 2)) ** 2))
    # Clip the mask to ensure values are reasonable
    importance = np.clip(mask, 0.4, 1.0)
    
    # Process patches in batches
    batches = []
    batch_indices = []
    
    total_patches = n_h * n_w
    
    with tqdm(total=total_patches, desc=f"Predicting {total_patches} patches") as pbar:
        for i in range(n_h):
            for j in range(n_w):
                # Calculate patch coordinates
                y_start = i * effective_patch
                x_start = j * effective_patch
                
                # Extract patch
                patch = padded_data[:, y_start:y_start+patch_size, x_start:x_start+patch_size]
                
                # Add to batch
                batches.append(patch)
                batch_indices.append((y_start, x_start))
                
                # Process batch if it's full or last patch
                if len(batches) == batch_size or (i == n_h-1 and j == n_w-1):
                    # Convert to torch tensor
                    batch_tensor = torch.tensor(np.array(batches), dtype=torch.float32).to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        
                        # Convert to probabilities if multi-class
                        if n_classes > 1:
                            outputs = F.softmax(outputs, dim=1)
                        else:
                            outputs = torch.sigmoid(outputs)
                    
                    # Add predictions to the result array
                    for idx, (y_start, x_start) in enumerate(batch_indices):
                        output = outputs[idx].cpu().numpy()
                        
                        # Apply the prediction with importance weighting
                        for c in range(n_classes):
                            class_probs[c, y_start:y_start+patch_size, x_start:x_start+patch_size] += output[c] * importance
                        
                        # Update the count matrix
                        counts[y_start:y_start+patch_size, x_start:x_start+patch_size] += importance
                    
                    # Clear batches
                    batches = []
                    batch_indices = []
                    
                    # Update progress bar
                    pbar.update(len(batch_indices))
    
    # Normalize predictions by counts
    for c in range(n_classes):
        class_probs[c] /= np.maximum(counts, 1e-8)
    
    # Crop back to original chunk size
    class_probs = class_probs[:, :height, :width]
    
    # Convert the class probabilities to class labels
    if n_classes > 1:
        # Memory-efficient approach: process one row at a time
        chunk_labels = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            chunk_labels[row] = np.argmax(class_probs[:, row, :], axis=0)
    else:
        chunk_labels = (class_probs[0] > 0.5).astype(np.uint8)
    
    # Free memory
    del class_probs, counts, padded_data
    gc.collect()
    
    return chunk_labels

def predict_large_image(model, input_data, input_meta, rf_info, patch_size=256, batch_size=4, chunk_size=2000, output_path=None):
    """
    Memory-efficient prediction for large images by processing in chunks.
    
    Args:
        model: The UNet model
        input_data: Normalized input data [channels, height, width]
        input_meta: Metadata for the input image
        rf_info: Receptive field information
        patch_size: Size of patches for prediction
        batch_size: Batch size for prediction
        chunk_size: Size of chunks to process
        output_path: Path to save the prediction
        
    Returns:
        Full prediction as a numpy array
    """
    print(f"Predicting large image with memory-efficient chunking...")
    
    # Get image dimensions
    channels, height, width = input_data.shape
    
    # Calculate overlap based on receptive field
    patch_overlap = rf_info["effective_rf"] // 2
    
    # Ensure patch_overlap is not too large relative to patch_size
    if patch_overlap >= patch_size // 2:
        patch_overlap = patch_size // 4
    
    print(f"Using patch size={patch_size}, patch overlap={patch_overlap}")
    
    # Calculate chunk overlap (half the receptive field)
    chunk_overlap = rf_info["effective_rf"] // 2
    
    # Ensure chunk_size is larger than the receptive field
    if chunk_size <= rf_info["effective_rf"]:
        chunk_size = rf_info["effective_rf"] * 2
        print(f"Adjusted chunk size to {chunk_size} to ensure it's larger than receptive field")
    
    # Calculate effective chunk size after removing overlap
    effective_chunk = chunk_size - 2 * chunk_overlap
    
    # Calculate number of chunks in height and width
    n_chunks_h = max(1, int(np.ceil((height - 2 * chunk_overlap) / effective_chunk)))
    n_chunks_w = max(1, int(np.ceil((width - 2 * chunk_overlap) / effective_chunk)))
    
    print(f"Dividing image ({height}x{width}) into {n_chunks_h}x{n_chunks_w} chunks of size {chunk_size}x{chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}px for seamless merging")
    
    # Create output array for the full prediction
    full_prediction = np.zeros((height, width), dtype=np.uint8)
    
    # If output path is provided, prepare to save the result
    if output_path:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create output file with same metadata
        output_meta = input_meta.copy()
        output_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 255  # Set nodata value
        })
        
        # Create output file
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            # We'll write to this file chunk by chunk
            pass
    
    # Process each chunk
    start_time = time.time()
    
    with tqdm(total=n_chunks_h * n_chunks_w, desc="Processing chunks") as pbar:
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                # Calculate chunk coordinates with overlap
                chunk_y_start = max(0, i * effective_chunk - chunk_overlap)
                chunk_y_end = min(height, (i + 1) * effective_chunk + chunk_overlap)
                chunk_x_start = max(0, j * effective_chunk - chunk_overlap)
                chunk_x_end = min(width, (j + 1) * effective_chunk + chunk_overlap)
                
                # Extract chunk
                chunk_height = chunk_y_end - chunk_y_start
                chunk_width = chunk_x_end - chunk_x_start
                
                print(f"\nProcessing chunk ({i},{j}) with size {chunk_height}x{chunk_width}")
                print(f"Coordinates: y=[{chunk_y_start}:{chunk_y_end}], x=[{chunk_x_start}:{chunk_x_end}]")
                
                # Extract data for this chunk
                chunk_data = input_data[:, chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
                
                # Predict this chunk
                chunk_prediction = predict_chunk(
                    model, 
                    chunk_data, 
                    patch_size=patch_size, 
                    overlap=patch_overlap,
                    batch_size=batch_size
                )
                
                # Calculate the non-overlapping region of this chunk
                # This is the region we'll use in the final prediction
                valid_y_start = chunk_overlap if i > 0 else 0
                valid_y_end = chunk_height - chunk_overlap if i < n_chunks_h - 1 else chunk_height
                valid_x_start = chunk_overlap if j > 0 else 0
                valid_x_end = chunk_width - chunk_overlap if j < n_chunks_w - 1 else chunk_width
                
                # Coordinates in the full image for this valid region
                full_y_start = chunk_y_start + valid_y_start
                full_y_end = chunk_y_start + valid_y_end
                full_x_start = chunk_x_start + valid_x_start
                full_x_end = chunk_x_start + valid_x_end
                
                # Copy the valid region to the full prediction
                full_prediction[full_y_start:full_y_end, full_x_start:full_x_end] = chunk_prediction[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                
                # If we're saving to a file, update the output file with this chunk
                if output_path:
                    with rasterio.open(output_path, 'r+') as dst:
                        # Write just the valid part of this chunk
                        window = Window(
                            full_x_start, 
                            full_y_start, 
                            full_x_end - full_x_start, 
                            full_y_end - full_y_start
                        )
                        dst.write(
                            chunk_prediction[valid_y_start:valid_y_end, valid_x_start:valid_x_end], 
                            1, 
                            window=window
                        )
                
                # Free memory
                del chunk_data, chunk_prediction
                gc.collect()
                
                # Update progress bar
                pbar.update(1)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nPrediction completed in {elapsed_time:.1f} seconds")
    print(f"Output shape: {full_prediction.shape}")
    
    return full_prediction

def load_normalizer(normalizer_path):
    """Load the normalizer object from file"""
    if not normalizer_path or not os.path.exists(normalizer_path):
        return None
    
    print(f"Loading normalizer from: {normalizer_path}")
    try:
        # Try to import the custom normalizer
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset.normalizers import load_sentinel2_normalizer
        
        normalizer = load_sentinel2_normalizer(normalizer_path)
        print(f"Successfully loaded normalizer: {getattr(normalizer, 'method', 'unknown')}")
        return normalizer
    except ImportError:
        print("Could not import normalizer module. Will use default normalization.")
        return None
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        return None

def evaluate_prediction(prediction, reference_path, num_classes):
    """
    Evaluate the prediction against reference data.
    Calculates multiple metrics: IoU, F1, Accuracy, Confusion Matrix.
    """
    print(f"Evaluating prediction against reference data: {reference_path}")
    
    # Read reference data
    with rasterio.open(reference_path) as src:
        reference_data = src.read(1)
        reference_meta = src.meta.copy()
    
    # Ensure prediction and reference have the same dimensions
    if prediction.shape != reference_data.shape:
        print(f"Warning: Shape mismatch between prediction {prediction.shape} and reference {reference_data.shape}")
        # Clip to the smaller size
        min_height = min(prediction.shape[0], reference_data.shape[0])
        min_width = min(prediction.shape[1], reference_data.shape[1])
        prediction = prediction[:min_height, :min_width]
        reference_data = reference_data[:min_height, :min_width]
    
    # Create flattened versions of the arrays for metric calculation
    pred_flat = prediction.flatten()
    ref_flat = reference_data.flatten()
    
    # Create masks for valid pixels (not outside the area of interest)
    # Assuming 255 or negative values are invalid
    valid_mask = (ref_flat >= 0) & (ref_flat < 255)
    pred_flat = pred_flat[valid_mask]
    ref_flat = ref_flat[valid_mask]
    
    # Calculate metrics
    metrics = {}
    
    try:
        # Overall accuracy
        accuracy = np.mean(pred_flat == ref_flat)
        metrics["overall_accuracy"] = float(accuracy)
        
        if num_classes > 1:
            # Multi-class metrics
            # Jaccard index (IoU) per class and mean
            iou_per_class = jaccard_score(ref_flat, pred_flat, average=None, 
                                         labels=range(num_classes))
            mean_iou = np.mean(iou_per_class)
            
            # F1 score per class and mean
            f1_per_class = f1_score(ref_flat, pred_flat, average=None, 
                                   labels=range(num_classes))
            mean_f1 = np.mean(f1_per_class)
            
            # Confusion matrix
            cm = confusion_matrix(ref_flat, pred_flat, labels=range(num_classes))
            
            # Per-class accuracy
            class_accuracy = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)
            
            # Add metrics to dict
            metrics.update({
                "mean_iou": float(mean_iou),
                "mean_f1": float(mean_f1),
                "class_iou": iou_per_class.tolist(),
                "class_f1": f1_per_class.tolist(),
                "class_accuracy": class_accuracy.tolist(),
                "confusion_matrix": cm.tolist()
            })
        else:
            # Binary metrics
            iou = jaccard_score(ref_flat, pred_flat)
            f1 = f1_score(ref_flat, pred_flat)
            cm = confusion_matrix(ref_flat, pred_flat)
            
            # True positive, false positive, etc.
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Add metrics to dict
            metrics.update({
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "confusion_matrix": cm.tolist()
            })
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Fallback to basic accuracy
        metrics["accuracy"] = float(accuracy)
    
    # Print metrics
    print("\n===== QUANTITATIVE EVALUATION METRICS =====")
    print(f"Overall Accuracy: {metrics.get('overall_accuracy', 'N/A'):.4f}")
    
    if 'mean_iou' in metrics:
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    if 'mean_f1' in metrics:
        print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
    
    if 'class_iou' in metrics:
        print("\nPer-class IoU:")
        for i, iou in enumerate(metrics['class_iou']):
            print(f"  Class {i}: {iou:.4f}")
    
    return metrics, reference_data, reference_meta

def visualize_results(image, prediction, reference=None, class_names=None, save_path=None):
    """
    Create visualization for qualitative assessment:
    1. RGB representation of the Sentinel-2 image
    2. Predicted land cover classes
    3. Reference land cover classes (if available)
    """
    print("Creating visualization for qualitative assessment...")
    
    # Create RGB visualization from Sentinel-2 bands
    # Use bands 4,3,2 (Red, Green, Blue) - based on 0-indexing of bands
    rgb_indices = [3, 2, 1]  # B04, B03, B02
    
    # Extract RGB bands
    rgb = np.zeros((3, image.shape[1], image.shape[2]), dtype=np.float32)
    for i, band_idx in enumerate(rgb_indices):
        if band_idx < image.shape[0]:
            band = image[band_idx].copy()
            # Check if values need scaling
            if np.max(band) > 100:
                band = band / 10000.0
            # Use percentile based scaling for better visualization
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            rgb[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # Transpose for matplotlib display
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # For large images, downsample for visualization
    max_size = 2000
    if rgb.shape[0] > max_size or rgb.shape[1] > max_size:
        # Downsample by selecting every nth pixel
        scale = max(rgb.shape[0] // max_size, rgb.shape[1] // max_size)
        rgb = rgb[::scale, ::scale]
        prediction = prediction[::scale, ::scale]
        if reference is not None:
            reference = reference[::scale, ::scale]
    
    # Set up colormap for classes
    num_classes = len(class_names) if class_names else np.max(prediction) + 1
    cmap = plt.cm.get_cmap('tab10', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
    class_cmap = ListedColormap(colors)
    
    # Use provided class names or generate defaults
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Create figure
    if reference is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # RGB image
        axs[0].imshow(rgb)
        axs[0].set_title('Sentinel-2 RGB', fontsize=14)
        axs[0].axis('off')
        
        # Prediction
        pred_img = axs[1].imshow(prediction, cmap=class_cmap, vmin=0, vmax=num_classes-1)
        axs[1].set_title('Predicted Land Cover', fontsize=14)
        axs[1].axis('off')
        
        # Reference
        ref_img = axs[2].imshow(reference, cmap=class_cmap, vmin=0, vmax=num_classes-1)
        axs[2].set_title('Reference Land Cover', fontsize=14)
        axs[2].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pred_img, cax=cbar_ax)
        cbar.set_ticks(np.arange(0, num_classes, 1) + 0.5)
        cbar.set_ticklabels(class_names)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # RGB image
        axs[0].imshow(rgb)
        axs[0].set_title('Sentinel-2 RGB', fontsize=14)
        axs[0].axis('off')
        
        # Prediction
        pred_img = axs[1].imshow(prediction, cmap=class_cmap, vmin=0, vmax=num_classes-1)
        axs[1].set_title('Predicted Land Cover', fontsize=14)
        axs[1].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pred_img, cax=cbar_ax)
        cbar.set_ticks(np.arange(0, num_classes, 1) + 0.5)
        cbar.set_ticklabels(class_names)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 Land Cover Prediction from zip/SAFE/GeoTIFF")
    
    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained UNet model")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to Sentinel-2 data (zip, SAFE directory, or GeoTIFF)")
    
    # Optional paths
    parser.add_argument("--reference_path", type=str, default=None,
                        help="Path to reference data (GBDA24_ex2_34SEH_ref_data.tif)")
    parser.add_argument("--output_dir", type=str, default="sentinel2_prediction",
                        help="Directory for saving results")
    parser.add_argument("--normalizer_path", type=str, default=None,
                        help="Path to normalizer file")
    
    # Model parameters
    parser.add_argument("--num_classes", type=int, default=9,
                        help="Number of land cover classes")
    parser.add_argument("--in_channels", type=int, default=13,
                        help="Number of input channels (13 for Sentinel-2)")
    
    # Prediction parameters
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Size of patches for prediction")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for prediction")
    parser.add_argument("--chunk_size", type=int, default=2000,
                        help="Size of chunks for large image processing")
    
    # Class names for visualization
    parser.add_argument("--class_names_file", type=str, default=None,
                        help="JSON file with class names")
    
    # Keep temp files
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary files after processing")
    
    args = parser.parse_args()
    
    # Print info about execution
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load the model
    model, rf_info = load_model(
        args.model_path, 
        num_classes=args.num_classes,
        in_channels=args.in_channels
    )
    
    # 2. Process Sentinel-2 data (from zip/SAFE/GeoTIFF)
    preprocessed_path = os.path.join(args.output_dir, "preprocessed_sentinel2.tif")
    sentinel2_data, sentinel2_meta, temp_dir = process_sentinel2_data(
        args.input_path,
        output_path=preprocessed_path,
        cleanup_temp=not args.keep_temp
    )
    
    # 3. Load normalizer if available
    normalizer = load_normalizer(args.normalizer_path)
    
    # 4. Normalize the data
    normalized_data = normalize_data(sentinel2_data, normalizer)
    
    # 5. Predict using chunking approach
    prediction_path = os.path.join(args.output_dir, "prediction.tif")
    prediction = predict_large_image(
        model,
        normalized_data,
        sentinel2_meta,
        rf_info,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        output_path=prediction_path
    )
    
    # 6. Load class names if provided
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    else:
        # Default class names
        class_names = [f"Class {i}" for i in range(args.num_classes)]
    
    # 7. Evaluate if reference data is available
    metrics = {}
    reference_data = None
    
    if args.reference_path and os.path.exists(args.reference_path):
        metrics, reference_data, reference_meta = evaluate_prediction(
            prediction,
            args.reference_path,
            args.num_classes
        )
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")
        
        # Create visualization for qualitative assessment
        vis_path = os.path.join(args.output_dir, "visualization.png")
        visualize_results(
            sentinel2_data,
            prediction,
            reference=reference_data,
            class_names=class_names,
            save_path=vis_path
        )
    else:
        print("No reference data provided. Skipping quantitative evaluation.")
        
        # Create visualization without reference
        vis_path = os.path.join(args.output_dir, "visualization.png")
        visualize_results(
            sentinel2_data, 
            prediction,
            class_names=class_names,
            save_path=vis_path
        )
    
    # 8. Clean up temporary directory if created and not keeping
    if temp_dir and not args.keep_temp:
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
    
    print(f"\nPrediction completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Prediction map: {prediction_path}")

if __name__ == "__main__":
    main()