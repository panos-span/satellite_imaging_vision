"""
Enhanced prediction script for Sentinel-2 imagery using a pre-trained U-Net model.
This version includes robust handling of class remapping for accurate prediction results.
"""

import os
import sys
import zipfile
import tempfile
import json
import numpy as np
import rasterio
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage

# Import the model
from models.unet import UNet
from dataset.normalizers import Sentinel2Normalizer, load_sentinel2_normalizer, normalize_batch
from data_preparation.pansharpening import simple_pansharpening, brovey_pansharpening

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_mapping(dataset_dir):
    """
    Load class mapping from a dataset directory.
    
    Parameters:
    -----------
    dataset_dir : str
        Path to the dataset directory containing mapping files
    
    Returns:
    --------
    dict or None
        Class mapping dictionary if found, None otherwise
    """
    mapping_files = [
        os.path.join(dataset_dir, "inverse_class_mapping.json"),
        os.path.join(dataset_dir, "class_mapping.json"),
        os.path.join(dataset_dir, "model_class_config.json")
    ]
    
    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            print(f"Found mapping file: {mapping_file}")
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                print(f"Keys in mapping file: {list(mapping_data.keys())}")
                
                # Process based on file structure
                if mapping_file.endswith("inverse_class_mapping.json"):
                    # This file typically has model_class -> original_class mapping
                    reverse_mapping = {str(k): int(v) for k, v in mapping_data.items()}
                    class_mapping = {str(v): int(k) for k, v in reverse_mapping.items()}
                    
                    return {
                        'reverse_mapping': reverse_mapping,
                        'class_mapping': class_mapping
                    }
                
                elif mapping_file.endswith("class_mapping.json"):
                    # This file typically has original_class -> model_class mapping
                    class_mapping = {str(k): int(v) for k, v in mapping_data.items()}
                    reverse_mapping = {str(v): int(k) for k, v in class_mapping.items()}
                    
                    return {
                        'class_mapping': class_mapping,
                        'reverse_mapping': reverse_mapping
                    }
                
                elif 'class_mapping' in mapping_data and 'reverse_mapping' in mapping_data:
                    # This file has both mappings
                    return {
                        'class_mapping': mapping_data['class_mapping'],
                        'reverse_mapping': mapping_data['reverse_mapping']
                    }
                
                elif 'num_classes' in mapping_data and 'class_values' in mapping_data:
                    # This is a model_class_config.json file
                    if 'class_mapping' in mapping_data:
                        class_mapping = mapping_data['class_mapping']
                        if 'reverse_mapping' in mapping_data:
                            reverse_mapping = mapping_data['reverse_mapping']
                        else:
                            # Create reverse mapping
                            reverse_mapping = {str(v): k for k, v in class_mapping.items()}
                        
                        return {
                            'class_mapping': class_mapping,
                            'reverse_mapping': reverse_mapping
                        }
            
            except Exception as e:
                print(f"Error loading mapping file {mapping_file}: {e}")
    
    print("No valid mapping files found!")
    return None


def debug_data_statistics(image_tensor, name="image"):
    """Print statistics about a tensor to help debugging."""
    if isinstance(image_tensor, torch.Tensor):
        print(f"\n{name} statistics:")
        print(f"  Shape: {image_tensor.shape}")
        print(f"  Min: {image_tensor.min().item():.4f}, Max: {image_tensor.max().item():.4f}")
        print(f"  Mean: {image_tensor.mean().item():.4f}, Std: {image_tensor.std().item():.4f}")
        # Sample a few values
        sample = image_tensor.flatten()[:5]
        print(f"  Sample values: {[f'{x.item():.4f}' for x in sample]}")
    elif isinstance(image_tensor, np.ndarray):
        print(f"\n{name} statistics:")
        print(f"  Shape: {image_tensor.shape}")
        print(f"  Min: {image_tensor.min():.4f}, Max: {image_tensor.max():.4f}")
        print(f"  Mean: {image_tensor.mean():.4f}, Std: {image_tensor.std():.4f}")
        # Sample a few values
        sample = image_tensor.flatten()[:5]
        print(f"  Sample values: {[f'{x:.4f}' for x in sample]}")


def extract_sentinel_zip(zip_path):
    """Extract Sentinel-2 zip file to temporary directory"""
    print(f"Extracting {zip_path}...")
    temp_dir = tempfile.mkdtemp(prefix="sentinel_extract_")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find SAFE directories
    safe_dirs = glob.glob(os.path.join(temp_dir, "*.SAFE"))
    
    if not safe_dirs:
        raise ValueError(f"No .SAFE directories found in {zip_path}")
    
    print(f"Found {len(safe_dirs)} SAFE directories: {[os.path.basename(d) for d in safe_dirs]}")
    
    return safe_dirs[0]  # Return the first SAFE directory


def load_model(model_path, device):
    """Load a trained U-Net model from checkpoint and extract class mapping info"""
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Debug: print all checkpoint keys to help with troubleshooting
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get number of classes
    config = checkpoint.get('config', {})
    num_classes = checkpoint.get('num_classes', config.get('num_classes', None))
    
    if num_classes is None:
        raise ValueError("Could not determine number of classes from model checkpoint")
    
    # Create model
    model = UNet(
        in_channels=13,  # Sentinel-2 has 13 bands
        num_classes=num_classes,
        encoder_type="best_performance",
        use_batchnorm=True,
        skip_connections=4,
        freeze_backbone=False  # Not important for inference
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with {num_classes} output classes")
    print(f"Best validation IoU: {checkpoint.get('val_iou', 'Unknown')}")
    
    # Check for class mapping information in the checkpoint
    class_mapping = None
    
    # Option 1: Direct 'remapper' key
    if 'remapper' in checkpoint:
        class_mapping = checkpoint['remapper']
        print("Found class mapping in checkpoint['remapper']")
    
    return model, num_classes, class_mapping


def create_multiband_image(safe_dir, output_path):
    """
    Create a 13-band Sentinel-2 image at 10m resolution using Brovey pansharpening
    to match training data preparation.
    """
    print(f"Processing Sentinel-2 data from {os.path.basename(safe_dir)}")
    
    # Define band groups by resolution
    high_res_bands = ['B02', 'B03', 'B04', 'B08']  # 10m
    medium_res_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # 20m
    low_res_bands = ['B01', 'B09', 'B10']  # 60m
    
    # Find all band files
    img_data_path = list(Path(safe_dir).glob('GRANULE/*/IMG_DATA'))[0]
    all_band_files = list(img_data_path.glob('*.jp2'))
    
    # Group files by band name
    band_files = {}
    for f in all_band_files:
        if 'TCI' not in f.name:  # Skip true color image
            # Extract band name (e.g., B01, B02)
            for band in high_res_bands + medium_res_bands + low_res_bands:
                if f.name.endswith(f'{band}.jp2'):
                    band_files[band] = str(f)
                    break
    
    print(f"Found {len(band_files)} bands")
    
    # Step 1: Read all high-resolution bands first
    high_res_band_data = {}
    high_res_shape = None
    high_res_profile = None
    high_res_transform = None
    high_res_crs = None
    
    # Find and read high-resolution bands
    for band_name in high_res_bands:
        if band_name in band_files:
            with rasterio.open(band_files[band_name]) as src:
                # Store band data
                high_res_band_data[band_name] = src.read(1)
                
                # Store metadata from the first high-resolution band
                if high_res_profile is None:
                    high_res_profile = src.profile.copy()
                    high_res_shape = (src.height, src.width)
                    high_res_transform = src.transform
                    high_res_crs = src.crs
    
    # Step 2: Process each band with proper pansharpening
    all_bands_data = []
    
    # Process each band
    for band_name in tqdm(high_res_bands + medium_res_bands + low_res_bands, desc="Processing bands"):
        if band_name in band_files:
            with rasterio.open(band_files[band_name]) as src:
                band_data = src.read(1)
                
                # Apply pansharpening for low/medium resolution bands
                if band_name in medium_res_bands or band_name in low_res_bands:
                    if len(high_res_band_data) >= 3:
                        # Use Brovey pansharpening to match training data preparation
                        band_data = brovey_pansharpening(band_data, high_res_band_data, high_res_shape)
                    else:
                        # Fallback to simple if not enough high-res bands
                        band_data = simple_pansharpening(band_data, high_res_shape)
                
                all_bands_data.append(band_data)
        else:
            # Create an empty band if missing
            print(f"Warning: Creating empty band for {band_name}")
            all_bands_data.append(np.zeros(high_res_shape, dtype=np.uint16))
    
    # Step 3: Stack all bands and save
    stacked_bands = np.stack(all_bands_data)
    
    # Update profile for multiband output
    output_profile = high_res_profile.copy()
    output_profile.update({
        'count': len(all_bands_data),
        'driver': 'GTiff',
        'dtype': 'uint16',
    })
    
    # Write multiband image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(stacked_bands)
        
        # Add band names as metadata
        dst.update_tags(band_names=','.join(high_res_bands + medium_res_bands + low_res_bands))
    
    print(f"Created multiband image with {len(all_bands_data)} bands at {output_path}")
    
    return output_path


class PredictionDataset(Dataset):
    """Dataset for making predictions using sliding window"""
    
    def __init__(self, image_path, patch_size=512, overlap=128, normalizer=None):
        self.image_path = image_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.normalizer = normalizer
        
        # Read image metadata
        with rasterio.open(image_path) as src:
            self.height = src.height
            self.width = src.width
            self.count = src.count
            self.profile = src.profile.copy()
        
        # Calculate stride and patches
        self.stride = patch_size - overlap
        self.patches = []
        
        # Generate grid of patches
        for y in range(0, self.height - self.patch_size + 1, self.stride):
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                self.patches.append((y, x))
        
        # Add edge patches
        if (self.height - self.patch_size) % self.stride != 0:
            y = self.height - self.patch_size
            for x in range(0, self.width - self.patch_size + 1, self.stride):
                if (y, x) not in self.patches:
                    self.patches.append((y, x))
        
        if (self.width - self.patch_size) % self.stride != 0:
            x = self.width - self.patch_size
            for y in range(0, self.height - self.patch_size + 1, self.stride):
                if (y, x) not in self.patches:
                    self.patches.append((y, x))
        
        # Add bottom-right corner
        if (self.height - self.patch_size) % self.stride != 0 and (self.width - self.patch_size) % self.stride != 0:
            y = self.height - self.patch_size
            x = self.width - self.patch_size
            if (y, x) not in self.patches:
                self.patches.append((y, x))
        
        print(f"Created {len(self.patches)} patches of size {patch_size}x{patch_size} with {overlap} overlap")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Get coordinates
        y, x = self.patches[idx]
        
        # Read patch
        with rasterio.open(self.image_path) as src:
            patch = src.read(window=Window(x, y, self.patch_size, self.patch_size))
        
        # Convert to torch tensor
        patch_tensor = torch.from_numpy(patch).float()
        
        # Apply normalization
        if self.normalizer is not None:
            # Debug data before normalization
            if idx == 0:  # Just for the first patch
                debug_data_statistics(patch_tensor, "Before normalization")
                
            # Use normalize_batch for consistent handling
            patch_tensor = normalize_batch(patch_tensor.unsqueeze(0), self.normalizer, None).squeeze(0)
            
            # Debug after normalization
            if idx == 0:
                debug_data_statistics(patch_tensor, "After normalization")
        else:
            # Simple normalization
            if patch_tensor.max() > 100:
                patch_tensor = patch_tensor / 10000.0
                if idx == 0:
                    print(f"Applied simple normalization (division by 10000)")
        
        # Return tensor and coordinates
        return patch_tensor, torch.tensor([y, x, self.patch_size])


def apply_class_remapping(model_class_indices, class_mapping):
    """
    Apply class remapping to convert model output classes to original classes.
    
    Parameters:
    -----------
    model_class_indices : numpy.ndarray
        Model prediction output class indices
    class_mapping : dict
        Class mapping dictionary
        
    Returns:
    --------
    numpy.ndarray
        Remapped class indices
    """
    if class_mapping is None:
        print("WARNING: No class mapping available. Using model output classes directly.")
        return model_class_indices
    
    # Create output array
    class_indices = np.zeros_like(model_class_indices)
    
    # Get the unique model output classes in this prediction
    unique_model_classes = np.unique(model_class_indices)
    print(f"Model predicted these classes: {unique_model_classes}")
    
    # Check if reverse_mapping is available
    if 'reverse_mapping' not in class_mapping:
        print("WARNING: No reverse_mapping found in class_mapping!")
        return model_class_indices
    
    reverse_mapping = class_mapping['reverse_mapping']
    
    # Map each model class to its original class value
    remapped_classes = 0
    for model_class in unique_model_classes:
        model_class_str = str(int(model_class))
        
        if model_class_str in reverse_mapping:
            original_class = reverse_mapping[model_class_str]
            
            # Convert to int if it's a string
            if isinstance(original_class, str):
                try:
                    original_class = int(original_class)
                except ValueError:
                    print(f"Warning: Could not convert '{original_class}' to int, using as is")
            
            # Apply mapping
            class_indices[model_class_indices == model_class] = original_class
            remapped_classes += 1
            print(f"  Remapped class {model_class} → {original_class}")
        else:
            print(f"Warning: No mapping found for model class {model_class}, keeping original")
            class_indices[model_class_indices == model_class] = model_class
    
    # Check if any classes were successfully remapped
    if remapped_classes == 0:
        print("WARNING: Failed to remap any classes! Using original model output.")
        return model_class_indices
    
    print(f"After remapping, classes are: {np.unique(class_indices)}")
    return class_indices


def predict_image(model, image_path, output_path, patch_size=256, overlap=128, batch_size=4, 
                  num_classes=None, model_path=None, class_mapping=None, data_dir=None):
    """
    Predict land cover for a Sentinel-2 image using sliding window approach.
    """
    # Try to load the normalizer from multiple possible locations
    normalizer = None
    normalizer_paths = [
        r"F:\\processed_data\\training_dataset_256\\normalizer.pkl",
        os.path.join(os.path.dirname(model_path), "normalizer.pkl")
    ]
    
    # Add data_dir to search paths if provided
    if data_dir:
        normalizer_paths.insert(0, os.path.join(data_dir, "normalizer.pkl"))
    
    for normalizer_path in normalizer_paths:
        if os.path.exists(normalizer_path):
            print(f"Loading Sentinel-2 normalizer from {normalizer_path}")
            try:
                normalizer = load_sentinel2_normalizer(normalizer_path)
                print(f"Loaded normalizer with method: {normalizer.method}")
                break
            except Exception as e:
                print(f"Error loading normalizer from {normalizer_path}: {e}")
    
    if normalizer is None:
        # Fallback to default normalizer
        print("No saved normalizer found, creating default pretrained normalizer")
        normalizer = Sentinel2Normalizer(method="pretrained")
    
    # Create dataset and dataloader
    dataset = PredictionDataset(
        image_path=image_path,
        patch_size=patch_size,
        overlap=overlap,
        normalizer=normalizer
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get image shape
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
    
    # Create arrays for prediction
    if num_classes > 1:
        # Multiclass segmentation
        prediction_array = np.zeros((num_classes, height, width), dtype=np.float32)
    else:
        # Binary segmentation
        prediction_array = np.zeros((height, width), dtype=np.float32)
    
    counter_array = np.zeros((height, width), dtype=np.uint8)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        for images, coords in tqdm(dataloader, desc="Predicting"):
            # Move to device
            images = images.to(device)
            
            # Debug data statistics occasionally
            if counter_array.sum() == 0:  # Just for the first batch
                debug_data_statistics(images, "Model input")
                
            # Forward pass
            outputs = model(images)
            
            # Debug output statistics occasionally
            if counter_array.sum() == 0:  # Just for the first batch
                debug_data_statistics(outputs, "Model output")
            
            # Get probabilities
            if num_classes > 1:
                probs = torch.softmax(outputs, dim=1)
            else:
                probs = torch.sigmoid(outputs)
            
            # Move to CPU and numpy
            probs = probs.cpu().numpy()
            
            # Store predictions
            for i in range(len(coords)):
                y, x, size = coords[i]
                y, x, size = int(y), int(x), int(size)
                
                if num_classes > 1:
                    # Multiclass segmentation
                    prediction_array[:, y:y+size, x:x+size] += probs[i]
                else:
                    # Binary segmentation
                    prediction_array[y:y+size, x:x+size] += probs[i, 0]
                
                counter_array[y:y+size, x:x+size] += 1
    
    # Average predictions
    mask = counter_array > 0
    if num_classes > 1:
        # Multiclass segmentation
        for c in range(num_classes):
            prediction_array[c][mask] /= counter_array[mask]
        
        # Get class with highest probability (model output classes, typically 0-N)
        model_class_indices = np.argmax(prediction_array, axis=0).astype(np.uint8)
        
        # Apply a median filter to smooth the predictions
        print("Applying median filter to smooth predictions...")
        model_class_indices = ndimage.median_filter(model_class_indices, size=5)
        
        # Apply class remapping if available (convert from model indices to original values)
        class_indices = apply_class_remapping(model_class_indices, class_mapping)
    else:
        # Binary segmentation
        prediction_array[mask] /= counter_array[mask]
        # Threshold probabilities
        class_indices = (prediction_array > 0.5).astype(np.uint8)
        
        # Apply a median filter to smooth the predictions
        print("Applying median filter to smooth predictions...")
        class_indices = ndimage.median_filter(class_indices, size=5)
    
    # Save prediction as GeoTIFF
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255
    })
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(class_indices.astype('uint8'), 1)
    
    print(f"Saved prediction to {output_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(class_indices, cmap='viridis')
    plt.title('Land Cover Prediction')
    plt.colorbar(label='Class')
    plt.axis('off')
    
    # Save visualization
    vis_path = os.path.splitext(output_path)[0] + '_viz.png'
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {vis_path}")
    
    return output_path, vis_path


def main():
    """Main function for prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Land cover prediction with trained U-Net model")
    
    parser.add_argument('--model_path', type=str, 
                        default="models/experiments/cosine_annealing/unet_sentinel2_best.pth",
                        help="Path to the trained model")
    parser.add_argument('--zip_path', type=str, required=True,
                        help="Path to Sentinel-2 ZIP file")
    parser.add_argument('--output_dir', type=str, default="predictions",
                        help="Directory to save output files")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Path to dataset directory containing mapping files")
    parser.add_argument('--mapping_file', type=str, default=None,
                        help="Path to specific mapping file to use")
    parser.add_argument('--patch_size', type=int, default=256,
                        help="Size of patches for prediction")
    parser.add_argument('--overlap', type=int, default=128,
                        help="Overlap between patches")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for prediction")
    parser.add_argument('--force_remapping', action='store_true',
                        help="Force class remapping even if not found in model")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Print verbose debug information")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Extract ZIP file
        safe_dir = extract_sentinel_zip(args.zip_path)
        
        # Get tile ID
        tile_id = None
        safe_name = os.path.basename(safe_dir)
        parts = safe_name.split('_')
        for part in parts:
            if part.startswith('T') and len(part) == 6:
                tile_id = part
                break
        
        if not tile_id:
            tile_id = "T34SEH"  # Default to this tile ID if not found
        
        print(f"Processing tile {tile_id}")
        
        # Create multiband image
        processed_dir = os.path.join(args.output_dir, f"processed_{tile_id}")
        os.makedirs(processed_dir, exist_ok=True)
        multiband_path = os.path.join(processed_dir, "multiband.tif")
        
        if not os.path.exists(multiband_path):
            multiband_path = create_multiband_image(safe_dir, multiband_path)
        else:
            print(f"Using existing processed image: {multiband_path}")
        
        # Load model
        model, num_classes, class_mapping = load_model(args.model_path, device)
        
        # Try to load mapping from dataset directory if specified
        if args.data_dir is not None:
            print(f"\nLooking for class mapping in data directory: {args.data_dir}")
            data_dir_mapping = load_class_mapping(args.data_dir)
            if data_dir_mapping is not None:
                print("Successfully loaded class mapping from data directory!")
                class_mapping = data_dir_mapping
            else:
                print("Could not find valid mapping in data directory.")
        
        # If mapping file is specified, load it directly
        if args.mapping_file is not None and os.path.exists(args.mapping_file):
            print(f"\nLoading class mapping from specified file: {args.mapping_file}")
            with open(args.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if 'reverse_mapping' in mapping_data:
                class_mapping = mapping_data
            elif all(k.isdigit() for k in mapping_data.keys()):
                # This is just a reverse mapping
                reverse_mapping = {k: v for k, v in mapping_data.items()}
                class_mapping = {
                    'reverse_mapping': reverse_mapping,
                    'class_mapping': {str(v): k for k, v in reverse_mapping.items()}
                }
            print("Successfully loaded class mapping from specified file")
        
        # Make prediction
        prediction_path = os.path.join(args.output_dir, f"{tile_id}_land_cover_prediction.tif")
        
        # Predict
        prediction_path, vis_path = predict_image(
            model=model,
            image_path=multiband_path,
            output_path=prediction_path,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_classes=num_classes,
            model_path=args.model_path,
            class_mapping=class_mapping,
            data_dir=args.data_dir
        )
        
        # Save class mapping if available
        if class_mapping:
            mapping_path = os.path.join(args.output_dir, f"{tile_id}_class_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump(class_mapping, f, indent=2)
            print(f"Saved class mapping to: {mapping_path}")
        
        print("\nPrediction complete!")
        print(f"Prediction saved to: {prediction_path}")
        print(f"Visualization saved to: {vis_path}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()