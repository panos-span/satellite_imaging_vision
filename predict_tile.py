"""
Script to predict land cover classes for Sentinel-2 imagery using trained UNet model.

This script implements multi-scale overlapping tile prediction for large Sentinel-2 images,
optimizing for memory efficiency while ensuring seamless predictions by leveraging the 
model's receptive field.
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
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.merge import merge
from pathlib import Path
import pickle
from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap
import glob
import json

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing modules
from models.unet import UNet
from dataset.normalizers import load_sentinel2_normalizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_best_model(model_path, num_classes, in_channels=13, encoder_type="best_performance"):
    """Load the best trained model"""
    print(f"Loading model from {model_path}")
    
    # Create model with same architecture
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
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with val_iou: {checkpoint.get('val_iou', 'unknown'):.4f}")
    
    # Set model to evaluation mode
    model.eval()
    return model, checkpoint

def prepare_s2_data(image_path, output_path, apply_pansharpening=True):
    """Prepare Sentinel-2 data for prediction by aligning all bands to 10m resolution"""
    if os.path.exists(output_path):
        print(f"Using existing preprocessed data at {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(), src.meta.copy()
    
    print(f"Preparing Sentinel-2 data from {image_path}")
    
    # Check if input is a directory containing band files or a single file
    if os.path.isdir(image_path):
        # Find individual band files
        band_files = {}
        for band_name in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 
                          'B8A', 'B09', 'B10', 'B11', 'B12']:
            pattern = os.path.join(image_path, f"**/*{band_name}*.jp2")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                band_files[band_name] = matches[0]
        
        # Sort bands in correct order
        band_order = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        available_bands = [band_files[band] for band in band_order if band in band_files]
        
        # Read bands and stack them
        print(f"Reading {len(available_bands)} bands...")
        band_data = []
        band_meta = None
        for band_file in available_bands:
            with rasterio.open(band_file) as src:
                band_data.append(src.read(1))
                if band_meta is None:
                    band_meta = src.meta.copy()
                    # Check resolution
                    pixel_size_x = src.transform[0]
                    pixel_size_y = abs(src.transform[4])
                    print(f"Band resolution: {pixel_size_x}m x {pixel_size_y}m")
        
        # Stack bands
        stacked_data = np.stack(band_data)
        
        # Apply pansharpening if requested
        if apply_pansharpening:
            # Identify high-res bands (10m) and low-res bands (20m, 60m)
            high_res_indices = [1, 2, 3, 7]  # B02, B03, B04, B08 (0-indexed)
            low_res_indices = [4, 5, 6, 8, 9, 10, 11, 12]  # 20m and 60m bands
            
            # Pansharpen low-res bands to 10m
            print("Applying pansharpening to bring all bands to 10m resolution...")
            pansharpened_data = np.zeros((len(band_data), band_data[0].shape[0], band_data[0].shape[1]), 
                                         dtype=np.float32)
            
            # Copy high-res bands directly
            for i in high_res_indices:
                if i < len(band_data):
                    pansharpened_data[i] = band_data[i]
            
            # Resample low-res bands to high-res
            for i in low_res_indices:
                if i < len(band_data):
                    # Simple bilinear upsampling
                    lr_band = band_data[i]
                    hr_shape = band_data[high_res_indices[0]].shape
                    
                    # Use rasterio to resample
                    with rasterio.Env():
                        # Calculate transform for the low-res band
                        src_transform = band_meta['transform']
                        dst_transform = rasterio.transform.Affine(
                            src_transform[0] / 2, src_transform[1], src_transform[2],
                            src_transform[3], src_transform[4] / 2, src_transform[5]
                        )
                        
                        # Perform the resampling
                        hr_band = np.zeros(hr_shape, dtype=np.float32)
                        rasterio.warp.reproject(
                            source=lr_band,
                            destination=hr_band,
                            src_transform=src_transform,
                            src_crs=band_meta['crs'],
                            dst_transform=dst_transform,
                            dst_crs=band_meta['crs'],
                            resampling=rasterio.warp.Resampling.bilinear
                        )
                        
                        pansharpened_data[i] = hr_band
            
            stacked_data = pansharpened_data
            
            # Update metadata for the pansharpened data
            band_meta.update({
                'count': len(band_data),
                'height': stacked_data.shape[1],
                'width': stacked_data.shape[2],
                'dtype': 'float32'
            })
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **band_meta) as dst:
            for i in range(stacked_data.shape[0]):
                dst.write(stacked_data[i].astype(rasterio.float32), i+1)
        
        print(f"Preprocessed data saved to {output_path}")
        return stacked_data, band_meta
    
    else:
        # Single file case
        with rasterio.open(image_path) as src:
            data = src.read()
            meta = src.meta.copy()
            return data, meta

def predict_with_overlap(model, data, patch_size, overlap, batch_size=4):
    """Make predictions using overlapping patches and merge results"""
    height, width = data.shape[1], data.shape[2]
    n_channels = data.shape[0]
    n_classes = model.num_classes
    
    # Calculate effective patch size (area that will be kept after removing overlap regions)
    effective_patch = patch_size - 2 * overlap
    
    # Calculate number of patches in height and width dimensions
    n_h = int(np.ceil((height - 2 * overlap) / effective_patch))
    n_w = int(np.ceil((width - 2 * overlap) / effective_patch))
    
    # Pad the image to fit exactly n_h x n_w patches
    pad_h = n_h * effective_patch + 2 * overlap - height
    pad_w = n_w * effective_patch + 2 * overlap - width
    
    if pad_h > 0 or pad_w > 0:
        padded_data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded_data = data
        
    # Create prediction array
    padded_height, padded_width = padded_data.shape[1], padded_data.shape[2]
    prediction = np.zeros((n_classes, padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)
    
    # Create batches of patches
    batches = []
    batch_indices = []
    
    # Extract overlapping patches
    print(f"Extracting patches with size {patch_size} and overlap {overlap}")
    print(f"Total patches: {n_h * n_w}")
    
    # Create a soft importance mask that gives higher weight to the center of each patch
    importance = np.ones((patch_size, patch_size), dtype=np.float32)
    y, x = np.ogrid[:patch_size, :patch_size]
    center_y, center_x = patch_size / 2, patch_size / 2
    # Create a radial mask that decreases weight towards the edges
    mask = (1 - 0.5 * np.sqrt(((y - center_y) / (patch_size / 2)) ** 2 + 
                           ((x - center_x) / (patch_size / 2)) ** 2))
    # Clip the mask to ensure values are reasonable
    mask = np.clip(mask, 0.4, 1.0)
    importance = mask
    
    with tqdm(total=n_h * n_w) as pbar:
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
                            prediction[c, y_start:y_start+patch_size, x_start:x_start+patch_size] += output[c] * importance
                        
                        # Update the count matrix
                        counts[y_start:y_start+patch_size, x_start:x_start+patch_size] += importance
                    
                    # Clear batches
                    batches = []
                    batch_indices = []
                    
                    # Update progress bar
                    pbar.update(len(batch_indices))
    
    # Normalize predictions by counts
    for c in range(n_classes):
        prediction[c] /= np.maximum(counts, 1e-8)
    
    # Crop back to original size
    prediction = prediction[:, :height, :width]
    
    # Convert the class probabilities to class labels
    if n_classes > 1:
        prediction = np.argmax(prediction, axis=0).astype(np.uint8)
    else:
        prediction = (prediction[0] > 0.5).astype(np.uint8)
        
    return prediction

def predict_large_tile(model, tile_data, normalizer, patch_size=256, batch_size=8):
    """Make prediction on large tile using overlapping patches based on receptive field"""
    # Get receptive field info from model to calculate overlap
    rf_info = model.get_receptive_field_size()
    rf_size = rf_info['effective_rf']
    
    print(f"Model receptive field: {rf_size} pixels")
    
    # Calculate required overlap to ensure seamless predictions
    # We use half the effective receptive field as overlap
    overlap = rf_size // 2
    
    # Ensure overlap is smaller than patch size
    if overlap >= patch_size // 2:
        overlap = patch_size // 4
        
    print(f"Using overlap of {overlap} pixels for predictions")
    
    # Normalize the data if normalizer is provided
    if normalizer is not None:
        print("Applying normalization...")
        norm_data = np.zeros_like(tile_data, dtype=np.float32)
        for i in range(tile_data.shape[0]):
            band = tile_data[i].copy()
            # Use normalizer.normalize_band if it exists, otherwise do simple standardization
            if hasattr(normalizer, 'normalize_band'):
                norm_data[i] = normalizer.normalize_band(band, i)
            else:
                mean, std = np.mean(band[band > 0]), np.std(band[band > 0])
                norm_data[i] = (band - mean) / (std + 1e-8)
    else:
        # Basic normalization
        print("No normalizer provided, applying simple standardization")
        norm_data = np.zeros_like(tile_data, dtype=np.float32)
        for i in range(tile_data.shape[0]):
            band = tile_data[i].copy()
            if np.max(band) > 100:  # Likely raw Sentinel-2 reflectance values
                band = band / 10000.0
            mean, std = np.mean(band[band > 0]), np.std(band[band > 0])
            norm_data[i] = (band - mean) / (std + 1e-8)
    
    # Make predictions with overlap
    prediction = predict_with_overlap(model, norm_data, patch_size, overlap, batch_size)
    
    return prediction

def evaluate_prediction(prediction, reference_path, num_classes):
    """Evaluate the prediction against reference data"""
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
    if num_classes > 1:
        # Multi-class metrics
        # Jaccard index (IoU) per class and mean
        try:
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
            overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            
            # Results
            print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Mean F1 Score: {mean_f1:.4f}")
            print("\nPer-class metrics:")
            for i in range(num_classes):
                if i < len(iou_per_class):
                    print(f"  Class {i}: IoU={iou_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, Acc={class_accuracy[i]:.4f}")
        
            metrics = {
                "overall_accuracy": float(overall_accuracy),
                "mean_iou": float(mean_iou),
                "mean_f1": float(mean_f1),
                "class_iou": iou_per_class.tolist(),
                "class_f1": f1_per_class.tolist(),
                "class_accuracy": class_accuracy.tolist(),
                "confusion_matrix": cm.tolist()
            }
            
        except Exception as e:
            print(f"Error calculating multi-class metrics: {e}")
            # Fallback to basic metrics
            accuracy = np.mean(pred_flat == ref_flat)
            print(f"Basic accuracy: {accuracy:.4f}")
            metrics = {"accuracy": float(accuracy)}
    else:
        # Binary metrics
        try:
            iou = jaccard_score(ref_flat, pred_flat)
            f1 = f1_score(ref_flat, pred_flat)
            cm = confusion_matrix(ref_flat, pred_flat)
            
            # True positive, false positive, etc.
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"IoU: {iou:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            
            metrics = {
                "accuracy": float(accuracy),
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "confusion_matrix": cm.tolist()
            }
        except Exception as e:
            print(f"Error calculating binary metrics: {e}")
            # Fallback to basic metrics
            accuracy = np.mean(pred_flat == ref_flat)
            print(f"Basic accuracy: {accuracy:.4f}")
            metrics = {"accuracy": float(accuracy)}
    
    return metrics, reference_data, reference_meta

def visualize_results(image, prediction, reference=None, num_classes=9, class_names=None, save_path=None):
    """Visualize the prediction results with optional reference comparison"""
    
    # Create a false-color RGB composite from the Sentinel-2 image
    # Using bands 4 (red), 3 (green), 2 (blue) for natural color
    rgb_indices = [3, 2, 1]  # 0-indexed (B04, B03, B02)
    
    # Extract and scale the RGB bands for visualization
    rgb = np.zeros((3, image.shape[1], image.shape[2]), dtype=np.float32)
    for i, band_idx in enumerate(rgb_indices):
        if band_idx < image.shape[0]:
            band = image[band_idx].copy()
            # Check if values need scaling (Sentinel-2 reflectance values)
            if np.max(band) > 100:
                band = band / 10000.0
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            rgb[i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # Transpose for matplotlib display
    rgb = np.transpose(rgb, (1, 2, 0))
    
    # Create color maps for land cover classes
    # Spectral colormap for prediction
    cmap = plt.cm.get_cmap('tab10', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
    prediction_cmap = ListedColormap(colors)
    
    # Use class names if provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Set up the figure for visualization
    if reference is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot RGB composite
        axs[0].imshow(rgb)
        axs[0].set_title('Sentinel-2 RGB Composite')
        axs[0].axis('off')
        
        # Plot prediction
        pred_img = axs[1].imshow(prediction, cmap=prediction_cmap, vmin=0, vmax=num_classes-1)
        axs[1].set_title('Predicted Land Cover')
        axs[1].axis('off')
        
        # Plot reference
        ref_img = axs[2].imshow(reference, cmap=prediction_cmap, vmin=0, vmax=num_classes-1)
        axs[2].set_title('Reference Land Cover')
        axs[2].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pred_img, cax=cbar_ax)
        cbar.set_ticks(np.arange(0, num_classes, 1) + 0.5)
        cbar.set_ticklabels(class_names)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot RGB composite
        axs[0].imshow(rgb)
        axs[0].set_title('Sentinel-2 RGB Composite')
        axs[0].axis('off')
        
        # Plot prediction
        pred_img = axs[1].imshow(prediction, cmap=prediction_cmap, vmin=0, vmax=num_classes-1)
        axs[1].set_title('Predicted Land Cover')
        axs[1].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pred_img, cax=cbar_ax)
        cbar.set_ticks(np.arange(0, num_classes, 1) + 0.5)
        cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix for better visualization
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def save_prediction_to_geotiff(prediction, reference_meta, output_path):
    """Save the prediction as a GeoTIFF with same geospatial information as reference"""
    # Update metadata for the prediction
    meta = reference_meta.copy()
    meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255  # Optional: set a nodata value
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write prediction to file
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction.astype(rasterio.uint8), 1)
    
    print(f"Prediction saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Predict land cover from Sentinel-2 imagery")
    
    # Input and output paths
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the Sentinel-2 L1C tile (GeoTIFF or directory)")
    parser.add_argument("--reference_path", type=str, default=None,
                        help="Path to reference data for evaluation")
    parser.add_argument("--normalizer_path", type=str, default=None,
                        help="Path to the normalizer pickle file")
    parser.add_argument("--output_dir", type=str, default="prediction_results",
                        help="Directory to save results")
    
    # Model parameters
    parser.add_argument("--num_classes", type=int, default=9,
                        help="Number of land cover classes")
    parser.add_argument("--in_channels", type=int, default=13,
                        help="Number of input channels in the model")
    parser.add_argument("--encoder_type", type=str, default="best_performance",
                        help="Encoder type used in the model")
    
    # Prediction parameters
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Size of patches for prediction")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for prediction")
    parser.add_argument("--processed_data_path", type=str, default=None,
                        help="Path to save/load preprocessed data")
    
    # Visualization parameters
    parser.add_argument("--class_names_file", type=str, default=None,
                        help="JSON file with class names for visualization")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the best model
    model, checkpoint = load_best_model(
        args.model_path, 
        num_classes=args.num_classes, 
        in_channels=args.in_channels,
        encoder_type=args.encoder_type
    )
    
    # Get class names if provided
    class_names = None
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    
    # Load normalizer if available
    normalizer = None
    if args.normalizer_path and os.path.exists(args.normalizer_path):
        print(f"Loading normalizer from {args.normalizer_path}")
        normalizer = load_sentinel2_normalizer(args.normalizer_path)
        if hasattr(normalizer, 'method'):
            print(f"Loaded normalizer with method: {normalizer.method}")
    
    # Process Sentinel-2 input data
    processed_data_path = args.processed_data_path
    if processed_data_path is None:
        processed_data_path = os.path.join(args.output_dir, "processed_tile.tif")
    
    # Prepare/load Sentinel-2 data
    tile_data, tile_meta = prepare_s2_data(args.input_path, processed_data_path)
    
    # Make predictions
    print("Making predictions on tile...")
    prediction = predict_large_tile(
        model, 
        tile_data, 
        normalizer, 
        patch_size=args.patch_size,
        batch_size=args.batch_size
    )
    
    # Save prediction
    save_prediction_to_geotiff(
        prediction, 
        tile_meta, 
        os.path.join(args.output_dir, "prediction.tif")
    )
    
    # If reference data is provided, calculate metrics
    if args.reference_path and os.path.exists(args.reference_path):
        metrics, reference_data, reference_meta = evaluate_prediction(
            prediction, args.reference_path, args.num_classes
        )
        
        # Save metrics to JSON
        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot confusion matrix
        if 'confusion_matrix' in metrics:
            names = class_names if class_names else [f"Class {i}" for i in range(args.num_classes)]
            plot_confusion_matrix(
                np.array(metrics['confusion_matrix']), 
                names,
                save_path=os.path.join(args.output_dir, "confusion_matrix.png")
            )
        
        # Visualize results
        visualize_results(
            tile_data, 
            prediction, 
            reference=reference_data,
            num_classes=args.num_classes,
            class_names=class_names,
            save_path=os.path.join(args.output_dir, "visualization.png")
        )
    else:
        # Visualize without reference
        visualize_results(
            tile_data, 
            prediction,
            num_classes=args.num_classes,
            class_names=class_names,
            save_path=os.path.join(args.output_dir, "visualization.png")
        )
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()