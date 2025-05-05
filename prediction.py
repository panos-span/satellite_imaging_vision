"""
Prediction script for Sentinel-2 land cover classification.

This script processes a Sentinel-2 .SAFE directory and its ground truth file,
applies the same preprocessing as the training data (pansharpening, normalization,
class mapping), and makes predictions using a trained model.

Usage:
    python prediction.py --model_path model.pth --safe_dir S2A_MSIL1C_20210615.SAFE --ground_truth ground_truth.tif --output_dir predictions
"""

import argparse
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from scipy.ndimage import uniform_filter, generic_filter
from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

# Try to import from the current directory or from modules
try:
    from data_preparation.geospatial_utils import (
        align_raster_to_reference,
        get_ground_truth_info,
    )
    from data_preparation.pansharpening import get_pansharpening_method
    from dataset.normalizers import normalize_batch
    from models.unet import UNet
except ImportError:
    # Try another path structure
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_preparation.geospatial_utils import (
            align_raster_to_reference,
            get_ground_truth_info,
        )
        from data_preparation.pansharpening import get_pansharpening_method
        from dataset.normalizers import normalize_batch
        from models.unet import UNet
    except ImportError:
        print("Warning: Could not import custom modules. Some functionality might be limited.")
        # Define simplified versions of necessary functions


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spatial_context_refinement(prediction_map, window_size=7):
    """
    Refine predictions using spatial context.
    
    Parameters:
    -----------
    prediction_map : numpy.ndarray
        Input prediction map
    window_size : int
        Size of window for context (should be odd)
        
    Returns:
    --------
    numpy.ndarray
        Refined prediction map
    """
    print("Applying spatial context refinement...")
    
    # Create a smoothed version using a uniform filter for each class
    unique_classes = np.unique(prediction_map)
    class_probabilities = np.zeros((len(unique_classes), *prediction_map.shape), dtype=np.float32)
    
    # Create probability maps for each class
    for i, cls in enumerate(unique_classes):
        # Create binary mask for this class
        binary_mask = (prediction_map == cls).astype(np.float32)
        
        # Apply smoothing to get spatial context
        class_probabilities[i] = uniform_filter(binary_mask, size=window_size)
    
    # For each pixel, select the class with highest probability
    refined_prediction = np.zeros_like(prediction_map)
    best_class_idx = np.argmax(class_probabilities, axis=0)
    
    for i, cls in enumerate(unique_classes):
        refined_prediction[best_class_idx == i] = cls
    
    # Apply majority filtering to remove isolated pixels
    def most_common(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]
    
    final_prediction = generic_filter(
        refined_prediction,
        most_common,
        size=3,  # Smaller window for majority filter
        mode='nearest'
    )
    
    return final_prediction


def post_process_prediction(prediction_map, refined_mapping, window_size=7, ground_truth_info=None):
    """
    Apply comprehensive post-processing to improve prediction quality.
    
    Parameters:
    -----------
    prediction_map : numpy.ndarray
        Raw prediction map from the model
    refined_mapping : dict
        Mapping from model classes to ground truth classes
    window_size : int
        Window size for spatial context refinement
    ground_truth_info : dict, optional
        Additional information about ground truth
        
    Returns:
    --------
    numpy.ndarray
        Post-processed prediction map
    """
    print("Starting post-processing pipeline...")
    
    # Step 1: Apply refined mapping
    if refined_mapping:
        print("Applying refined class mapping...")
        mapped_prediction = np.zeros_like(prediction_map)
        for model_cls, gt_cls in refined_mapping.items():
            mapped_prediction[prediction_map == model_cls] = gt_cls
    else:
        mapped_prediction = prediction_map.copy()
    
    # Step 2: Apply spatial context refinement
    print("Applying spatial context refinement...")
    context_refined = spatial_context_refinement(
        mapped_prediction, 
        window_size=window_size
    )
    
    # Step 3: Remove small objects (noise)
    print("Removing small isolated regions...")
    
    # Process each class separately
    unique_classes = np.unique(context_refined)
    cleaned_prediction = np.zeros_like(context_refined)
    
    for cls in unique_classes:
        # Create binary mask for this class
        binary_mask = (context_refined == cls)
        
        # Remove small objects
        min_size = 20  # Minimum size in pixels
        cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
        
        # Add back to the prediction
        cleaned_prediction[cleaned_mask] = cls
    
    # Fill in any gaps created by small object removal
    # Create a background mask of pixels that need to be filled
    background_mask = ~np.isin(cleaned_prediction, unique_classes)
    
    # For background pixels, assign the class with highest probability in neighborhood
    if np.any(background_mask):
        print("Filling gaps in prediction...")
        
        # Create a distance-weighted map for each class
        for cls in unique_classes:
            cls_mask = (cleaned_prediction == cls)
            
            # Skip if class doesn't exist in prediction
            if not np.any(cls_mask):
                continue
                
            # For each unassigned pixel, check proximity to this class
            for y, x in zip(*np.where(background_mask)):
                # Define neighborhood window
                y_min = max(0, y - window_size//2)
                y_max = min(cleaned_prediction.shape[0], y + window_size//2 + 1)
                x_min = max(0, x - window_size//2)
                x_max = min(cleaned_prediction.shape[1], x + window_size//2 + 1)
                
                # Count occurrences of each class in neighborhood
                neighborhood = cleaned_prediction[y_min:y_max, x_min:x_max]
                unique_neighbors, counts = np.unique(neighborhood, return_counts=True)
                
                # Skip if no valid classes in neighborhood
                if len(unique_neighbors) == 0 or (len(unique_neighbors) == 1 and unique_neighbors[0] == 0):
                    continue
                    
                # Find most common class (excluding background)
                valid_idx = unique_neighbors != 0
                if np.any(valid_idx):
                    valid_unique = unique_neighbors[valid_idx]
                    valid_counts = counts[valid_idx]
                    most_common_cls = valid_unique[np.argmax(valid_counts)]
                    cleaned_prediction[y, x] = most_common_cls
    
    print("Post-processing complete.")
    return cleaned_prediction


def predict_patches_improved(model, image_path, ground_truth, num_classes, patch_size, overlap, batch_size, normalizer=None):
    """
    Memory-efficient prediction with improved edge handling and weighted blending.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model
    image_path : str
        Path to multiband image
    ground_truth : numpy.ndarray
        Ground truth array for co-occurrence calculation
    num_classes : int
        Number of output classes
    patch_size : int
        Size of patches for prediction
    overlap : int
        Overlap between patches
    batch_size : int
        Batch size for prediction
    normalizer : object, optional
        Normalizer for image preprocessing
        
    Returns:
    --------
    tuple
        (prediction_map, profile, co_occurrence, gt_unique)
    """
    print("Initializing memory-efficient prediction with improved edge handling...")
    
    # Open the multiband image
    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()
        
        # Calculate patch stride
        stride = patch_size - overlap
        
        # Calculate number of patches
        n_h = (height - overlap) // stride
        n_w = (width - overlap) // stride
        
        # Adjust for edge cases
        if (n_h * stride + overlap) < height:
            n_h += 1
        if (n_w * stride + overlap) < width:
            n_w += 1
        
        # Gather patch coordinates
        patches_coords = []
        for i in range(n_h):
            for j in range(n_w):
                x_start = j * stride
                y_start = i * stride
                
                # Adjust for right and bottom edges
                if x_start + patch_size > width:
                    x_start = width - patch_size
                if y_start + patch_size > height:
                    y_start = height - patch_size
                
                patches_coords.append((y_start, x_start))
        
        # Create empty prediction map and weight map for blending
        prediction_map = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Create empty co-occurrence matrix
        gt_unique = np.unique(ground_truth)
        co_occurrence = np.zeros((num_classes, len(gt_unique)), dtype=np.int64)
        
        print(f"Processing {len(patches_coords)} patches...")
        
        # Create a weight matrix for blending that gives less weight to patch edges
        patch_weight = np.ones((patch_size, patch_size), dtype=np.float32)
        y_coords = np.linspace(-1, 1, patch_size)
        x_coords = np.linspace(-1, 1, patch_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        # Create a radial falloff from center (1.0) to edges
        r = np.sqrt(X**2 + Y**2) 
        patch_weight = np.clip(1.0 - r, 0.2, 1.0)
        
        # Process patches in batches
        batch_idx = 0
        for i in range(0, len(patches_coords), batch_size):
            batch_coords = patches_coords[i:i+batch_size]
            batch_images = []
            
            # Load batch patches
            for y_start, x_start in batch_coords:
                # Read window
                window = Window(x_start, y_start, patch_size, patch_size)
                patch = src.read(window=window)
                batch_images.append(patch)
            
            # Stack and convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch_images)).float()
            
            # Apply normalization
            if normalizer is not None:
                batch_tensor = normalize_batch(batch_tensor, normalizer, device)
            else:
                # Simple default normalization
                batch_tensor = batch_tensor / 10000.0
            
            # Move to device
            batch_tensor = batch_tensor.to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(batch_tensor)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # Get class predictions for co-occurrence matrix
                predictions = np.argmax(probs, axis=1)
                
                # Update the prediction map with weighted probabilities
                for idx, (y_start, x_start) in enumerate(batch_coords):
                    # Update prediction map (using soft probabilities)
                    for cls in range(num_classes):
                        cls_prob = probs[idx, cls]
                        weighted_prob = cls_prob * patch_weight
                        prediction_map[cls, y_start:y_start+patch_size, x_start:x_start+patch_size] += weighted_prob
                    
                    # Update weight map
                    weight_map[y_start:y_start+patch_size, x_start:x_start+patch_size] += patch_weight
                    
                    # Update co-occurrence matrix
                    gt_patch = ground_truth[y_start:y_start+patch_size, x_start:x_start+patch_size]
                    pred_patch = predictions[idx]
                    
                    # For each GT class in the patch
                    for gt_idx, gt_cls in enumerate(gt_unique):
                        gt_mask = gt_patch == gt_cls
                        if not np.any(gt_mask):
                            continue
                            
                        # For each prediction class
                        for pred_cls in range(num_classes):
                            pred_mask = pred_patch == pred_cls
                            overlap = np.sum(gt_mask & pred_mask)
                            co_occurrence[pred_cls, gt_idx] += overlap
            
            # Print progress
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size}/{len(patches_coords)} patches")
        
        # Normalize the prediction map by the weight map
        valid_mask = weight_map > 0
        final_pred_map = np.zeros((height, width), dtype=np.int32)
        
        for y in range(height):
            for x in range(width):
                if valid_mask[y, x]:
                    # Get all class probabilities at this pixel
                    pixel_probs = prediction_map[:, y, x] / weight_map[y, x]
                    # Assign the class with the highest probability
                    final_pred_map[y, x] = np.argmax(pixel_probs)
    
    return final_pred_map, profile, co_occurrence, gt_unique

def create_refined_mapping(co_occurrence, gt_unique, num_classes):
    """
    Create refined mapping from co-occurrence matrix.
    
    Parameters:
    -----------
    co_occurrence : numpy.ndarray
        Co-occurrence matrix of shape (num_classes, len(gt_unique))
    gt_unique : numpy.ndarray
        Array of unique ground truth class values
    num_classes : int
        Number of classes in the model output
        
    Returns:
    --------
    dict
        Refined class mapping
    """
    print("\nCreating refined mapping from co-occurrence matrix...")
    
    # Print co-occurrence matrix
    print("\nCo-occurrence matrix (rows: prediction class, columns: ground truth class):")
    gt_cls_str = [str(int(cls)) for cls in gt_unique]
    print("   " + " ".join([f"{cls:>6}" for cls in gt_cls_str]))
    for pred_cls in range(num_classes):
        print(f"{pred_cls:2d} " + " ".join([f"{co_occurrence[pred_cls, i]:6d}" for i in range(len(gt_unique))]))
    
    # For each prediction class, find the most common ground truth class
    mapping = {}
    for pred_cls in range(num_classes):
        if np.sum(co_occurrence[pred_cls, :]) == 0:
            continue  # Skip if no overlap
            
        best_gt_idx = np.argmax(co_occurrence[pred_cls, :])
        best_gt_cls = int(gt_unique[best_gt_idx])
        mapping[pred_cls] = best_gt_cls
        
        # Print match information
        overlap = co_occurrence[pred_cls, best_gt_idx]
        total_pred = np.sum(co_occurrence[pred_cls, :])
        print(f"Prediction class {pred_cls} maps to ground truth class {best_gt_cls} "
              f"({overlap/total_pred*100:.2f}% match)")
    
    # Check if any ground truth classes were not mapped
    mapped_gt = set(mapping.values())
    unmapped_gt = [cls for cls in gt_unique if int(cls) not in mapped_gt]
    
    if unmapped_gt:
        print(f"\nWARNING: {len(unmapped_gt)} ground truth classes were not mapped: {unmapped_gt}")
        
        # Try to map any remaining prediction classes to unmapped ground truth classes
        unmapped_pred = [i for i in range(num_classes) if i not in mapping]
        
        if unmapped_pred and unmapped_gt:
            for pred_cls, gt_cls in zip(unmapped_pred[:len(unmapped_gt)], unmapped_gt):
                mapping[pred_cls] = int(gt_cls)
                print(f"Unmapped prediction class {pred_cls} assigned to ground truth class {gt_cls}")
    
    return mapping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentinel-2 land cover classification prediction"
    )
    
    parser.add_argument(
        "--post_process",
        action="store_true",
        default=True,
        help="Apply spatial post-processing to refine predictions",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Window size for spatial context refinement (odd number)",
    )


    # Input/output paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file)",
    )
    parser.add_argument(
        "--safe_dir",
        type=str,
        required=True,
        help="Path to Sentinel-2 .SAFE directory or zip file",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth .tif file for reference",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to training dataset directory (for class mapping and normalizer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save prediction results",
    )

    # Processing options
    parser.add_argument(
        "--pansharpening",
        type=str,
        default="brovey",
        choices=["simple", "brovey", "hpf"],
        help="Pansharpening method to use",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Size of patches for prediction (must match training)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for prediction"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=96,
        help="Overlap between patches for seamless prediction",
    )
    parser.add_argument(
        "--use_normalizer",
        action="store_true",
        default=True,
        help="Use the saved Sentinel-2 normalizer",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Create visualizations of the predictions",
    )
    parser.add_argument(
        "--class_mapping_file",
        type=str,
        default=None,
        help="Path to class mapping file. If not provided, will look in dataset_dir.",
    )

    return parser.parse_args()


def extract_safe_dir(safe_dir):
    """Extract Sentinel-2 zip file if needed and return path to SAFE directory."""
    # Check if input is a zip file
    if safe_dir.endswith(".zip") and os.path.isfile(safe_dir):
        # Extract the zip file
        print(f"Extracting {os.path.basename(safe_dir)}...")
        extract_dir = os.path.dirname(safe_dir)
        with zipfile.ZipFile(safe_dir, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Get the SAFE directory path
        safe_name = os.path.basename(safe_dir).replace(".zip", "")
        safe_dir = os.path.join(extract_dir, safe_name)

    # Make sure the SAFE directory exists
    if not os.path.isdir(safe_dir) or not safe_dir.endswith(".SAFE"):
        raise ValueError(f"Invalid SAFE directory: {safe_dir}")

    return safe_dir


def process_sentinel_data(safe_dir, ground_truth_path, output_dir, pansharpening_method):
    """
    Process Sentinel-2 data for prediction.

    Parameters:
    -----------
    safe_dir : str
        Path to Sentinel-2 SAFE directory
    ground_truth_path : str
        Path to ground truth GeoTIFF file
    output_dir : str
        Output directory for processed data
    pansharpening_method : str
        Pansharpening method to use

    Returns:
    --------
    dict
        Dictionary with paths to processed data
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    pansharpened_dir = os.path.join(output_dir, "pansharpened")
    aligned_dir = os.path.join(output_dir, "aligned")
    os.makedirs(pansharpened_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)

    # Read ground truth for reference
    ground_truth_info = get_ground_truth_info(ground_truth_path)
    print(f"Ground truth CRS: {ground_truth_info['crs']}")
    print(f"Ground truth shape: {ground_truth_info['shape']}")
    print(f"Ground truth unique values: {ground_truth_info['unique_values']}")

    # Extract tile ID from SAFE directory name
    tile_id = os.path.basename(safe_dir).split("_")[5]
    print(f"Processing tile {tile_id} with {pansharpening_method} pansharpening...")

    # Step 1: Find all band files
    img_data_path = list(Path(safe_dir).glob("GRANULE/*/IMG_DATA"))[0]
    band_files = [f for f in img_data_path.glob("*.jp2") if not f.name.endswith("TCI.jp2")]

    # Define band resolution groups
    high_res_bands = ["B02", "B03", "B04", "B08"]  # 10m
    medium_res_bands = ["B05", "B06", "B07", "B8A", "B11", "B12"]  # 20m
    low_res_bands = ["B01", "B09", "B10"]  # 60m

    # Step 2: Get high-resolution band metadata and data
    high_res_band_data = {}
    high_res_profile = None
    high_res_shape = None
    high_res_transform = None
    high_res_crs = None

    # Find and read high-resolution bands
    for band_name in high_res_bands:
        band_files_filtered = [f for f in band_files if f.name.split("_")[-1].split(".")[0] == band_name]
        if band_files_filtered:
            high_res_band_file = band_files_filtered[0]
            with rasterio.open(high_res_band_file) as src:
                # Store band data
                high_res_band_data[band_name] = src.read(1)

                # Store metadata from the first high-resolution band
                if high_res_profile is None:
                    high_res_profile = src.profile.copy()
                    high_res_shape = (src.height, src.width)
                    high_res_transform = src.transform
                    high_res_crs = src.crs

    if not high_res_band_data:
        raise FileNotFoundError(f"No high-resolution bands found in {img_data_path}")

    print(f"Found {len(high_res_band_data)} high-resolution bands with shape {high_res_shape}")

    # Step 3: Apply pansharpening to all bands
    print("Applying pansharpening...")
    pansharpened_bands_paths = {}
    pansharpening_func = get_pansharpening_method(pansharpening_method)

    for band_file in tqdm(band_files, desc="Pansharpening bands"):
        band_name = band_file.name.split("_")[-1].split(".")[0]
        output_path = os.path.join(pansharpened_dir, f"{band_name}_pansharpened.tif")

        with rasterio.open(band_file) as src:
            # Determine if band needs pansharpening
            needs_pansharpening = band_name in medium_res_bands or band_name in low_res_bands

            if needs_pansharpening:
                # Read band data
                band_data = src.read(1)

                # Apply appropriate pansharpening method
                if pansharpening_method == "brovey" and len(high_res_band_data) >= 3:
                    pansharpened_data = pansharpening_func(
                        band_data, high_res_band_data, high_res_shape
                    )
                elif pansharpening_method == "hpf" and "B08" in high_res_band_data:
                    # Use NIR band (B08) as the high-resolution band for HPF
                    pansharpened_data = pansharpening_func(
                        band_data, high_res_band_data["B08"], high_res_shape
                    )
                else:
                    # Fallback to simple resize
                    from data_preparation.pansharpening import simple_pansharpening
                    pansharpened_data = simple_pansharpening(band_data, high_res_shape)
            else:
                # No pansharpening needed, just read the band
                pansharpened_data = src.read(1)

            # Create profile for output file
            output_profile = high_res_profile.copy()
            output_profile.update({
                "driver": "GTiff",
                "height": high_res_shape[0],
                "width": high_res_shape[1],
                "count": 1,
                "dtype": pansharpened_data.dtype,
                "crs": high_res_crs,
                "transform": high_res_transform
            })

            # Write pansharpened band
            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(pansharpened_data, 1)

            pansharpened_bands_paths[band_name] = output_path

    # Step 4: Align with ground truth
    print("Aligning with ground truth...")
    aligned_bands_paths = {}

    for band_name, band_path in tqdm(pansharpened_bands_paths.items(), desc="Aligning bands"):
        output_path = os.path.join(aligned_dir, f"{band_name}_aligned.tif")
        aligned_path = align_raster_to_reference(
            band_path, ground_truth_info, output_path
        )
        aligned_bands_paths[band_name] = aligned_path

    # Step 5: Create multiband image
    print("Creating multiband image...")
    multiband_path = os.path.join(output_dir, "multiband.tif")

    # Sort bands to ensure consistent order (important for model input)
    # Standard Sentinel-2 band order
    band_order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    
    # Filter to only include available bands
    available_bands = [b for b in band_order if b in aligned_bands_paths]
    
    # Create multiband GeoTIFF
    with rasterio.open(aligned_bands_paths[available_bands[0]]) as src:
        profile = src.profile.copy()
        profile.update({
            "count": len(available_bands),
            "driver": "GTiff",
        })
        
        with rasterio.open(multiband_path, "w", **profile) as dst:
            for i, band_name in enumerate(available_bands, 1):
                with rasterio.open(aligned_bands_paths[band_name]) as src_band:
                    dst.write(src_band.read(1), i)
                    # Write band name as metadata
                    dst.set_band_description(i, band_name)
            
            # Add band names to metadata
            dst.update_tags(band_names=",".join(available_bands))
    
    print(f"Multiband image created with {len(available_bands)} bands")
    print(f"Band order: {available_bands}")

    return {
        "multiband_path": multiband_path,
        "ground_truth_info": ground_truth_info,
        "band_order": available_bands,
    }


def load_model_and_mapping(model_path, dataset_dir, class_mapping_file=None):
    """
    Load trained model and class mapping.

    Parameters:
    -----------
    model_path : str
        Path to trained model file
    dataset_dir : str
        Path to dataset directory (for class mapping)
    class_mapping_file : str, optional
        Path to class mapping file. If not provided, will look in dataset_dir.

    Returns:
    --------
    tuple
        (model, class_mapping, inverse_mapping, normalizer)
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    config = checkpoint.get("config", {})
    
    # Determine number of classes
    num_classes = checkpoint.get("num_classes", config.get("num_classes", None))
    
    if num_classes is None:
        # Try to get from model_state_dict shape
        if "model_state_dict" in checkpoint:
            # Look for the final layer weights
            for key, value in checkpoint["model_state_dict"].items():
                if "final_conv.weight" in key:
                    num_classes = value.shape[0]
                    print(f"Detected {num_classes} classes from model weights")
                    break
        
        if num_classes is None:
            raise ValueError("Could not determine number of classes from model")
    
    print(f"Model has {num_classes} output classes")
    
    # Create model
    in_channels = config.get("in_channels", 13)  # Default for Sentinel-2
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_type=config.get("encoder_type", "resnet50"),
        use_batchnorm=config.get("use_batchnorm", True),
        skip_connections=config.get("skip_connections", 4),
    ).to(device)
    
    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    
    # Load class mapping if available
    class_mapping = None
    inverse_mapping = None
    
    # First check if provided in checkpoint
    if "remapper" in checkpoint:
        remapper_info = checkpoint["remapper"]
        class_mapping = remapper_info.get("class_mapping", None)
        inverse_mapping = remapper_info.get("reverse_mapping", None)
        print("Loaded class mapping from checkpoint")
    
    # Then check if a specific mapping file is provided
    elif class_mapping_file and os.path.exists(class_mapping_file):
        with open(class_mapping_file, "r") as f:
            class_mapping = json.load(f)
        
        # Create inverse mapping
        inverse_mapping = {str(v): int(k) for k, v in class_mapping.items()}
        print(f"Loaded class mapping from {class_mapping_file}")
    
    # Finally check in dataset directory
    elif dataset_dir:
        # Try standard locations
        mapping_files = [
            os.path.join(dataset_dir, "class_mapping.json"),
            os.path.join(dataset_dir, "model_class_config.json"),
        ]
        
        for mapping_file in mapping_files:
            if os.path.exists(mapping_file):
                with open(mapping_file, "r") as f:
                    mapping_data = json.load(f)
                
                if "class_mapping" in mapping_data:
                    class_mapping = mapping_data["class_mapping"]
                    inverse_mapping = mapping_data.get("reverse_mapping")
                    print(f"Loaded class mapping from {mapping_file}")
                    break
        
        # If we found class_mapping but not inverse_mapping, create it
        if class_mapping and not inverse_mapping:
            inverse_mapping = {str(v): int(k) for k, v in class_mapping.items()}
    
    # Load normalizer
    normalizer = None
    
    if dataset_dir:
        normalizer_path = os.path.join(dataset_dir, "normalizer.pkl")
        if os.path.exists(normalizer_path):
            try:
                with open(normalizer_path, "rb") as f:
                    normalizer = pickle.load(f)
                print(f"Loaded normalizer from {normalizer_path}")
            except Exception as e:
                print(f"Error loading normalizer: {e}")
                print("Will use default normalization")
    
    return model, class_mapping, inverse_mapping, normalizer


def predict_patches(model, image_path, ground_truth, num_classes, patch_size, overlap, batch_size, normalizer=None):
    """
    Memory-efficient prediction with direct co-occurrence matrix calculation.
    """
    print("Initializing memory-efficient prediction...")
    
    # Open the multiband image
    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()
        
        # Calculate patch stride
        stride = patch_size - overlap
        
        # Calculate number of patches
        n_h = (height - overlap) // stride
        n_w = (width - overlap) // stride
        
        # Adjust for edge cases
        if (n_h * stride + overlap) < height:
            n_h += 1
        if (n_w * stride + overlap) < width:
            n_w += 1
        
        # Gather patch coordinates
        patches_coords = []
        for i in range(n_h):
            for j in range(n_w):
                x_start = j * stride
                y_start = i * stride
                
                # Adjust for right and bottom edges
                if x_start + patch_size > width:
                    x_start = width - patch_size
                if y_start + patch_size > height:
                    y_start = height - patch_size
                
                patches_coords.append((y_start, x_start))
        
        # Create empty prediction map and count map
        prediction_map = np.zeros((height, width), dtype=np.int32)
        counts = np.zeros((height, width), dtype=np.int32)
        
        # Create empty co-occurrence matrix
        gt_unique = np.unique(ground_truth)
        co_occurrence = np.zeros((num_classes, len(gt_unique)), dtype=np.int64)
        
        print(f"Processing {len(patches_coords)} patches...")
        
        # Process patches in batches
        batch_idx = 0
        for i in range(0, len(patches_coords), batch_size):
            batch_coords = patches_coords[i:i+batch_size]
            batch_images = []
            
            # Load batch patches
            for y_start, x_start in batch_coords:
                # Read window
                window = Window(x_start, y_start, patch_size, patch_size)
                patch = src.read(window=window)
                batch_images.append(patch)
            
            # Stack and convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch_images)).float()
            
            # Apply normalization
            if normalizer is not None:
                batch_tensor = normalize_batch(batch_tensor, normalizer, device)
            else:
                # Simple default normalization
                batch_tensor = batch_tensor / 10000.0
            
            # Move to device
            batch_tensor = batch_tensor.to(device)
            
            # Make predictions
            with torch.no_grad():
                outputs = model(batch_tensor)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get class predictions
                _, predictions = torch.max(outputs, dim=1)
                predictions = predictions.cpu().numpy()
                
                # Update the co-occurrence matrix directly
                for idx, (y_start, x_start) in enumerate(batch_coords):
                    # Get ground truth patch
                    gt_patch = ground_truth[y_start:y_start+patch_size, x_start:x_start+patch_size]
                    
                    # For each GT class in the patch
                    for gt_idx, gt_cls in enumerate(gt_unique):
                        gt_mask = gt_patch == gt_cls
                        if not np.any(gt_mask):
                            continue
                            
                        # For each prediction class
                        pred_patch = predictions[idx]
                        for pred_cls in range(num_classes):
                            pred_mask = pred_patch == pred_cls
                            overlap = np.sum(gt_mask & pred_mask)
                            co_occurrence[pred_cls, gt_idx] += overlap
            
            # Add predictions to the map
            for idx, (y_start, x_start) in enumerate(batch_coords):
                prediction_map[y_start:y_start+patch_size, x_start:x_start+patch_size] += predictions[idx]
                counts[y_start:y_start+patch_size, x_start:x_start+patch_size] += 1
            
            # Print progress
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size}/{len(patches_coords)} patches")
        
        # Average overlapping predictions
        valid_mask = counts > 0
        prediction_map[valid_mask] = prediction_map[valid_mask] // counts[valid_mask]
    
    return prediction_map, profile, co_occurrence, gt_unique


def remap_prediction(prediction_map, inverse_mapping=None):
    """
    Remap predictions back to original class values.

    Parameters:
    -----------
    prediction_map : numpy.ndarray
        Prediction map with model output class indices
    inverse_mapping : dict or None
        Dictionary mapping model indices to original class values

    Returns:
    --------
    numpy.ndarray
        Remapped prediction map
    """
    if inverse_mapping is None:
        return prediction_map
    
    # Create remapped prediction map
    remapped_prediction = np.zeros_like(prediction_map)
    
    # Apply inverse mapping
    for model_idx_str, orig_class in inverse_mapping.items():
        model_idx = int(model_idx_str)
        remapped_prediction[prediction_map == model_idx] = orig_class
    
    return remapped_prediction


def save_prediction(prediction_map, profile, output_path, colormap=None):
    """
    Save prediction map as GeoTIFF and visualize it.

    Parameters:
    -----------
    prediction_map : numpy.ndarray
        Prediction map
    profile : dict
        Rasterio profile for geospatial metadata
    output_path : str
        Output path for the prediction GeoTIFF
    colormap : dict or None
        Colormap for visualization

    Returns:
    --------
    str
        Path to saved prediction GeoTIFF
    """
    # Update profile for the prediction map
    profile.update({
        "count": 1,
        "driver": "GTiff",
        "dtype": prediction_map.dtype,
        "nodata": 0,  # Assuming 0 is background/nodata
    })
    
    # Save prediction map
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction_map, 1)
    
    print(f"Prediction saved to {output_path}")
    
    return output_path


def create_prediction_visualization(prediction_path, ground_truth_path, output_dir, class_values=None):
    """
    Create visualization of the prediction compared to ground truth.

    Parameters:
    -----------
    prediction_path : str
        Path to prediction GeoTIFF
    ground_truth_path : str
        Path to ground truth GeoTIFF
    output_dir : str
        Output directory for visualizations
    class_values : list or None
        List of class values for legend

    Returns:
    --------
    list
        Paths to visualization images
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Read prediction and ground truth
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
        profile = src.profile
    
    with rasterio.open(ground_truth_path) as src:
        ground_truth = src.read(1)
    
    # Get unique values for colormap
    unique_values = np.unique(np.concatenate([np.unique(prediction), np.unique(ground_truth)]))
    
    # Create colormap
    if class_values is None:
        class_values = unique_values
    
    # Generate colors - using a predefined colormap
    import matplotlib.colors as mcolors
    colormap = plt.cm.get_cmap("tab20", len(unique_values))
    colors = {val: mcolors.to_hex(colormap(i)) for i, val in enumerate(unique_values)}
    
    # Create visualization of prediction
    output_vis_path = os.path.join(vis_dir, "prediction_visualization.png")
    
    # Convert prediction to RGB using colormap
    height, width = prediction.shape
    rgb_prediction = np.zeros((height, width, 3), dtype=np.uint8)
    
    for val in unique_values:
        mask = (prediction == val)
        if np.any(mask):
            color = mcolors.to_rgb(colors[val])
            for i in range(3):
                rgb_prediction[mask, i] = int(color[i] * 255)
    
    # Save prediction visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(rgb_prediction)
    plt.title("Prediction")
    plt.axis("off")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[val], label=f"Class {val}") for val in unique_values]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_vis_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # Create comparison visualization
    output_comparison_path = os.path.join(vis_dir, "prediction_vs_ground_truth.png")
    
    # Convert ground truth to RGB
    rgb_ground_truth = np.zeros((height, width, 3), dtype=np.uint8)
    
    for val in unique_values:
        mask = (ground_truth == val)
        if np.any(mask):
            color = mcolors.to_rgb(colors[val])
            for i in range(3):
                rgb_ground_truth[mask, i] = int(color[i] * 255)
    
    # Create comparison figure
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_ground_truth)
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_prediction)
    plt.title("Prediction")
    plt.axis("off")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[val], label=f"Class {val}") for val in unique_values]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_comparison_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # Create confusion matrix
    output_confusion_path = os.path.join(vis_dir, "confusion_matrix.png")
    
    # Mask out areas where ground truth is 0 (typically background/nodata)
    valid_mask = ground_truth > 0
    
    # Calculate confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(
        ground_truth[valid_mask].flatten(),
        prediction[valid_mask].flatten(),
        labels=unique_values,
        normalize='true'
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"{val}" for val in unique_values]
    )
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_confusion_path, dpi=200)
    plt.close()
    
    return [output_vis_path, output_comparison_path, output_confusion_path]


def calculate_metrics(prediction, ground_truth):
    """
    Calculate performance metrics.

    Parameters:
    -----------
    prediction : numpy.ndarray
        Prediction map
    ground_truth : numpy.ndarray
        Ground truth map

    Returns:
    --------
    dict
        Dictionary with various metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        jaccard_score,
        precision_score,
        recall_score,
    )
    
    # Mask out areas where ground truth is 0 (background/nodata)
    valid_mask = ground_truth > 0
    
    # Flatten arrays
    y_true = ground_truth[valid_mask].flatten()
    y_pred = prediction[valid_mask].flatten()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "iou": jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
    }
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return metrics


def main():
    """Run the prediction pipeline."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("Enhanced Sentinel-2 Land Cover Classification".center(80))
    print("=" * 80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Extract and validate SAFE directory
    print("Step 1: Processing SAFE directory...")
    safe_dir = extract_safe_dir(args.safe_dir)
    
    # Step 2: Process Sentinel-2 data
    print("\nStep 2: Processing Sentinel-2 data...")
    processed_data = process_sentinel_data(
        safe_dir,
        args.ground_truth,
        args.output_dir,
        args.pansharpening
    )
    
    # Step 3: Load model and mapping
    print("\nStep 3: Loading model and class mapping...")
    model, class_mapping, inverse_mapping, normalizer = load_model_and_mapping(
        args.model_path,
        args.dataset_dir,
        args.class_mapping_file
    )
    
    # Make sure inverse_mapping is loaded
    if inverse_mapping is None:
        print("ERROR: No inverse mapping found! Please provide a valid inverse mapping file.")
        print("Checking for inverse_class_mapping.json in the current directory...")
        if os.path.exists("inverse_class_mapping.json"):
            try:
                with open("inverse_class_mapping.json", "r") as f:
                    inverse_mapping = json.load(f)
                print("Successfully loaded inverse_class_mapping.json from current directory.")
            except Exception as e:
                print(f"Error loading inverse_class_mapping.json: {e}")
                print("Exiting...")
                return
        else:
            print("inverse_class_mapping.json not found in current directory.")
            print("Please provide a valid inverse mapping file using --class_mapping_file")
            print("Exiting...")
            return
    
    # Print loaded inverse mapping for verification
    print("\nLoaded inverse mapping:")
    for model_idx, gt_class in sorted([(int(k), int(v)) for k, v in inverse_mapping.items()]):
        print(f"  Model index {model_idx} -> Ground truth class {gt_class}")
    
    # Step 4: Read ground truth first
    print("\nStep 4: Reading ground truth...")
    with rasterio.open(args.ground_truth) as src:
        ground_truth = src.read(1)
        
    # Step 5: Making improved memory-efficient predictions
    print("\nStep 5: Making improved memory-efficient predictions...")
    prediction_map, profile, co_occurrence, gt_unique = predict_patches_improved(
        model,
        processed_data["multiband_path"],
        ground_truth,
        num_classes=model.final_conv.out_channels,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        normalizer=normalizer
    )

    # Step 6: SKIP refined mapping creation and FORCE using the inverse mapping
    print("\nStep 6: Using loaded inverse mapping (skipping co-occurrence mapping)")
    
    # Convert inverse_mapping to the format expected by post_process_prediction
    # The key difference: using the loaded inverse_mapping directly
    refined_mapping = {int(k): int(v) for k, v in inverse_mapping.items()}
    
    # Print the mapping that will be used
    print("\nUsing the following mapping:")
    for model_idx, gt_class in sorted(refined_mapping.items()):
        print(f"  Model index {model_idx} -> Ground truth class {gt_class}")

    # Step 7: Apply post-processing if requested
    if args.post_process:
        print("\nStep 7: Applying advanced spatial post-processing...")
        final_prediction = post_process_prediction(
            prediction_map,
            refined_mapping,
            window_size=args.window_size,
            ground_truth_info=processed_data["ground_truth_info"]
        )
    else:
        print("\nStep 7: Applying basic remapping without spatial post-processing...")
        print(f"Using provided inverse mapping with {len(refined_mapping)} classes")
        final_prediction = remap_prediction(prediction_map, refined_mapping)

    # Step 8: Save prediction
    print("\nStep 8: Saving prediction results...")
    prediction_path = os.path.join(args.output_dir, "prediction.tif")
    save_prediction(final_prediction, profile, prediction_path)
    
    # Step 7: Calculate metrics and visualize
    print("\nStep 7: Evaluating and visualizing results...")
    
    # Read ground truth for comparison
    with rasterio.open(args.ground_truth) as src:
        ground_truth = src.read(1)
    
    # Calculate metrics
    metrics = calculate_metrics(final_prediction, ground_truth)
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Create visualizations if requested
    if args.visualize:
        # Get unique classes for legend
        if inverse_mapping:
            class_values = sorted(list(map(int, inverse_mapping.values())))
        else:
            class_values = sorted(list(np.unique(final_prediction)))
        
        # Create visualizations
        vis_paths = create_prediction_visualization(
            prediction_path,
            args.ground_truth,
            args.output_dir,
            class_values
        )
        
        print(f"Created visualizations:")
        for path in vis_paths:
            print(f"  - {path}")
    
    # Step 8: Save prediction information
    info = {
        "model_path": args.model_path,
        "safe_dir": args.safe_dir,
        "ground_truth": args.ground_truth,
        "prediction_path": prediction_path,
        "metrics": metrics,
        "band_order": processed_data["band_order"],
        "class_mapping": class_mapping,
        "inverse_mapping": inverse_mapping
    }
    
    info_path = os.path.join(args.output_dir, "prediction_info.json")
    with open(info_path, "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        json_info = {}
        
        def convert_numpy_types(obj):
            """Helper function to convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8, np.uint8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            else:
                return obj
        
        # Apply conversion to all items
        json_info = convert_numpy_types(info)
        
        # Save to JSON
        json.dump(json_info, f, indent=4)
    
    print("\n" + "=" * 80)
    print("Prediction completed successfully!".center(80))
    print("=" * 80 + "\n")
    
    print(f"Prediction results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()