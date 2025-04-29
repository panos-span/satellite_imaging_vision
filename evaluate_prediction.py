"""
Enhanced evaluation script for land cover predictions with improved class remapping.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd
import seaborn as sns
import json
import glob

"""
python evaluate_prediction.py --prediction "F:\output\predictions\T34SEH_land_cover_prediction.tif"     --ground_truth 
"GBDA24_ex2_34SEH_ref_data.tif"     --mapping_file "F:\processed_data\training_dataset\inverse_class_mapping.json"     --output_dir "F:\output\predictions\evaluation_fixed"
Loaded F:\output\predictions\T34SEH_land_cover_prediction.tif
"""

def load_raster(file_path):
    """Load a raster file and return its data and metadata"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            profile = src.profile
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            
            # Get unique values
            unique_values = np.unique(data)
            
            print(f"Loaded {file_path}")
            print(f"Shape: {data.shape}")
            print(f"Unique values: {unique_values}")
            if nodata is not None:
                print(f"Nodata value: {nodata}")
            
            return data, profile, transform, crs, unique_values, nodata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def find_class_mapping_file(prediction_path):
    """
    Search for class mapping files in the prediction directory or parent directories.
    Looks for files like class_mapping.json, inverse_class_mapping.json, model_class_config.json.
    """
    # Start with the directory containing the prediction file
    search_dir = os.path.dirname(os.path.abspath(prediction_path))
    
    # Try different depths of parent directories (up to 3 levels)
    for _ in range(4):
        # Look for common mapping file names
        for filename in ['inverse_class_mapping.json', 'class_mapping.json', 'model_class_config.json']:
            mapping_path = os.path.join(search_dir, filename)
            if os.path.exists(mapping_path):
                print(f"Found mapping file: {mapping_path}")
                return mapping_path
                
        # Also try to find files with these patterns
        patterns = ['*class*mapping*.json', '*model*config*.json']
        for pattern in patterns:
            matches = glob.glob(os.path.join(search_dir, pattern))
            if matches:
                print(f"Found potential mapping file: {matches[0]}")
                return matches[0]
        
        # Move up one directory
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # We've reached the root
            break
        search_dir = parent_dir
    
    return None

def load_class_mapping_from_file(mapping_path):
    """
    Load class mapping from a file.
    Handles different types of mapping files and formats.
    """
    try:
        with open(mapping_path, 'r') as f:
            content = json.load(f)
        
        # Different mapping files have different structures
        mapping = {}
        
        if 'inverse' in os.path.basename(mapping_path):
            # This is already an inverse mapping (model class → original class)
            for model_class, original_class in content.items():
                mapping[int(model_class)] = int(original_class)
        
        elif 'class_mapping' in content:
            # This is from model_class_config.json - need to invert it
            for original_class, model_class in content['class_mapping'].items():
                mapping[int(model_class)] = int(original_class)
        
        elif 'reverse_mapping' in content:
            # This is a reverse mapping from model_class_config.json
            for model_class, original_class in content['reverse_mapping'].items():
                mapping[int(model_class)] = int(original_class)
        
        else:
            # Regular mapping (original class → model class) - need to invert it
            for original_class, model_class in content.items():
                mapping[int(model_class)] = int(original_class)
        
        print(f"Loaded class mapping from {mapping_path}")
        print(f"Mapping: {mapping}")
        return mapping
    
    except Exception as e:
        print(f"Error loading mapping file {mapping_path}: {e}")
        return None

def create_class_mapping(pred_classes, gt_classes):
    """
    Create a mapping between prediction and ground truth classes.
    This is used if no mapping file is found.
    
    Parameters:
    -----------
    pred_classes : array-like
        List of unique class values in the prediction
    gt_classes : array-like
        List of unique class values in the ground truth
        
    Returns:
    --------
    dict
        Mapping from prediction classes to ground truth classes
    """
    # Initialize the mapping
    mapping = {}
    
    # Remove nodata values (255 for predictions, 0 for ground truth if it's nodata)
    pred_classes = sorted([c for c in pred_classes if c != 255])
    
    # Check if 0 is nodata or a valid class in ground truth
    gt_has_zero = 0 in gt_classes
    gt_classes_filtered = sorted([c for c in gt_classes if c != 0]) if not gt_has_zero else sorted(gt_classes)
    
    # If ground truth has 0 as a class, map prediction 0 to ground truth 0
    if gt_has_zero and 0 in pred_classes:
        mapping[0] = 0
        # Remove 0 from both lists to process the rest
        if 0 in pred_classes:
            pred_classes.remove(0)
        if 0 in gt_classes_filtered:
            gt_classes_filtered.remove(0)
    
    # Check if the number of remaining classes match
    if len(pred_classes) != len(gt_classes_filtered):
        print(f"Warning: Number of classes differs between prediction ({len(pred_classes)}) "
              f"and ground truth ({len(gt_classes_filtered)})")
        print("This might lead to incorrect mapping.")
    
    # Map prediction classes to ground truth classes in order
    for i, pred_class in enumerate(pred_classes):
        if i < len(gt_classes_filtered):
            mapping[pred_class] = gt_classes_filtered[i]
        else:
            # If more prediction classes than ground truth classes,
            # map extras to the last ground truth class
            mapping[pred_class] = gt_classes_filtered[-1]
    
    print(f"Created class mapping: {mapping}")
    return mapping

def remap_prediction(prediction, mapping):
    """
    Remap prediction classes according to mapping.
    
    Parameters:
    -----------
    prediction : numpy.ndarray
        Prediction array with model output classes
    mapping : dict
        Dictionary mapping from model classes to original classes
        
    Returns:
    --------
    numpy.ndarray
        Remapped prediction with original class values
    """
    if mapping is None:
        return prediction

    # Create a remapped copy of the prediction
    remapped = np.zeros_like(prediction)
    
    # Apply mapping
    for pred_class, gt_class in mapping.items():
        remapped[prediction == pred_class] = gt_class
    
    # Keep nodata values (usually 255)
    remapped[prediction == 255] = 255
    
    print(f"Remapped prediction classes: {np.unique(remapped)}")
    return remapped

def create_rgb_from_multiband(multiband_path, output_dir):
    """Create an RGB visualization from a multiband Sentinel-2 image"""
    try:
        with rasterio.open(multiband_path) as src:
            # Try to find RGB bands (B04, B03, B02)
            band_names = src.tags().get('band_names', '').split(',')
            rgb_indices = []
            
            for band_name in ['B04', 'B03', 'B02']:
                if band_name in band_names:
                    rgb_indices.append(band_names.index(band_name) + 1)  # +1 for 1-indexed bands
            
            # If RGB bands not found, use first three bands
            if len(rgb_indices) != 3:
                rgb_indices = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            
            # Read bands
            red = src.read(rgb_indices[0])
            green = src.read(rgb_indices[1])
            blue = src.read(rgb_indices[2])
            
            # Create RGB array
            rgb = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.float32)
            rgb[:,:,0] = red
            rgb[:,:,1] = green
            rgb[:,:,2] = blue
            
            # Scale each band to [0, 1]
            for i in range(3):
                p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
                rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2) if p98 > p2 else 0, 0, 1)
            
            print("Created RGB visualization from multiband image")
            return rgb
    except Exception as e:
        print(f"Error creating RGB visualization: {e}")
        return None

def create_visualizations(prediction, ground_truth, rgb_image, output_dir, subsample=10):
    """Create visualizations for qualitative assessment"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subsampled version for visualization (to reduce memory usage)
    if max(prediction.shape) > 5000:
        print(f"Subsampling data by factor {subsample} for visualization...")
        pred_vis = prediction[::subsample, ::subsample]
        gt_vis = ground_truth[::subsample, ::subsample] if ground_truth is not None else None
        rgb_vis = rgb_image[::subsample, ::subsample] if rgb_image is not None else None
    else:
        pred_vis = prediction
        gt_vis = ground_truth
        rgb_vis = rgb_image
    
    # Get unique classes
    pred_classes = np.unique(pred_vis)
    gt_classes = np.unique(gt_vis) if gt_vis is not None else []
    
    # Create a set of unique classes across both datasets
    unique_classes = sorted(list(set(pred_classes) | set(gt_classes)))
    
    # Remove nodata values (255)
    unique_classes = [c for c in unique_classes if c != 255]
    
    # Create colormap
    cmap = plt.cm.get_cmap('tab10', len(unique_classes))
    
    # Create prediction visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(pred_vis, cmap=cmap)
    plt.colorbar(label='Class')
    plt.title('Land Cover Prediction')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'prediction_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ground truth visualization if available
    if gt_vis is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(gt_vis, cmap=cmap)
        plt.colorbar(label='Class')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'ground_truth_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create side-by-side comparison
        plt.figure(figsize=(24, 12))
        
        plt.subplot(1, 2, 1)
        plt.imshow(gt_vis, cmap=cmap)
        plt.colorbar(label='Class')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(pred_vis, cmap=cmap)
        plt.colorbar(label='Class')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create RGB visualization if available
    if rgb_vis is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_vis)
        plt.title('RGB Image')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'rgb_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create RGB with prediction overlay
        plt.figure(figsize=(12, 12))
        plt.imshow(rgb_vis)
        plt.imshow(pred_vis, cmap=cmap, alpha=0.5)
        plt.title('RGB with Prediction Overlay')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'rgb_prediction_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # If ground truth is available, create RGB with ground truth overlay
        if gt_vis is not None:
            plt.figure(figsize=(12, 12))
            plt.imshow(rgb_vis)
            plt.imshow(gt_vis, cmap=cmap, alpha=0.5)
            plt.title('RGB with Ground Truth Overlay')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'rgb_ground_truth_overlay.png'), dpi=300, bbox_inches='tight')
            plt.close()

def calculate_metrics(prediction, ground_truth, output_dir):
    """Calculate evaluation metrics"""
    if ground_truth is None:
        print("No ground truth data available for quantitative evaluation")
        return None
        
    print("Calculating metrics...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # First, remove pixels where either prediction or ground truth is nodata
    valid_mask = (prediction != 255) & (ground_truth != 0)
    
    if np.sum(valid_mask) == 0:
        print("No valid pixels for comparison")
        return None
    
    # Get valid pixels
    y_true = ground_truth[valid_mask]
    y_pred = prediction[valid_mask]
    
    # Get unique classes
    classes = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    
    # Calculate IoU for each class
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    iou = np.zeros(len(classes))
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        iou[i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Print metrics
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"F1 score (micro): {f1_micro:.4f}")
    print(f"F1 score (macro): {f1_macro:.4f}")
    print(f"Mean IoU: {np.mean(iou):.4f}")
    
    # Create and display a table of per-class metrics
    metrics_df = pd.DataFrame({
        'class': classes,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    })
    
    # Print per-class metrics
    print("Per-class metrics:")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.6f}"))
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'), index=False)
    
    # Create confusion matrix plot
    try:
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar chart of class metrics
        metrics_melted = pd.melt(metrics_df, id_vars=['class'], value_vars=['precision', 'recall', 'f1_score', 'iou'])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='class', y='value', hue='variable', data=metrics_melted)
        plt.title('Per-Class Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, 'class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating metric visualizations: {e}")
    
    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'mean_iou': np.mean(iou),
        'per_class': metrics_df.to_dict('records')
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate land cover predictions")
    
    # Required arguments
    parser.add_argument('--prediction', type=str, required=True,
                        help="Path to prediction GeoTIFF")
    
    # Optional arguments
    parser.add_argument('--ground_truth', type=str, default=None,
                        help="Path to ground truth GeoTIFF")
    parser.add_argument('--multiband', type=str, default=None,
                        help="Path to multiband Sentinel-2 image for RGB visualization")
    parser.add_argument('--output_dir', type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument('--subsample', type=int, default=10,
                        help="Subsample factor for visualizations")
    parser.add_argument('--mapping_file', type=str, default=None,
                        help="Path to class mapping file (will be auto-detected if not provided)")
    parser.add_argument('--save_remapped', action='store_true',
                        help="Save the remapped prediction file")
    parser.add_argument('--auto_remap', action='store_true', default=True,
                        help="Automatically remap prediction classes to match ground truth (default: True)")
                        
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load prediction
    pred_data, pred_profile, pred_transform, pred_crs, pred_classes, pred_nodata = load_raster(args.prediction)
    
    # Step 2: Load ground truth if available
    gt_data = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        gt_data, gt_profile, gt_transform, gt_crs, gt_classes, gt_nodata = load_raster(args.ground_truth)
        
        # Check if we need to remap predictions to match ground truth classes
        if args.auto_remap:
            print("\nChecking for class remapping needs...")
            
            # Check for differences in class values
            pred_valid_classes = [c for c in pred_classes if c != 255]
            gt_valid_classes = [c for c in gt_classes if c != 0 or 0 in pred_valid_classes]
            
            need_remapping = set(pred_valid_classes) != set(gt_valid_classes)
            
            if need_remapping:
                print("Class values differ between prediction and ground truth. Remapping needed.")
                print(f"Prediction classes: {pred_valid_classes}")
                print(f"Ground truth classes: {gt_valid_classes}")
                
                # Try to find and load mapping file
                mapping = None
                
                # Check if mapping file is provided
                if args.mapping_file and os.path.exists(args.mapping_file):
                    mapping = load_class_mapping_from_file(args.mapping_file)
                else:
                    # Try to find mapping file automatically
                    mapping_path = find_class_mapping_file(args.prediction)
                    if mapping_path:
                        mapping = load_class_mapping_from_file(mapping_path)
                
                # If no mapping file found, create mapping based on class values
                if mapping is None:
                    print("No mapping file found. Creating mapping based on class values...")
                    mapping = create_class_mapping(pred_classes, gt_classes)
                
                # Apply mapping to prediction
                remapped_pred = remap_prediction(pred_data, mapping)
                
                # Save remapped prediction if requested
                if args.save_remapped:
                    remapped_path = os.path.join(args.output_dir, 'remapped_prediction.tif')
                    with rasterio.open(remapped_path, 'w', **pred_profile) as dst:
                        dst.write(remapped_pred, 1)
                    print(f"Saved remapped prediction to {remapped_path}")
            else:
                print("No remapping needed. Prediction classes match ground truth classes.")
                remapped_pred = pred_data
        else:
            # Skip remapping if auto_remap is disabled
            print("Automatic remapping disabled. Using original prediction classes.")
            remapped_pred = pred_data
    else:
        print("No ground truth data provided, skipping quantitative evaluation")
        remapped_pred = pred_data
    
    # Step 3: Create RGB visualization if multiband image is provided
    rgb_image = None
    if args.multiband and os.path.exists(args.multiband):
        rgb_image = create_rgb_from_multiband(args.multiband, args.output_dir)
    
    # Step 4: Create visualizations
    create_visualizations(remapped_pred, gt_data, rgb_image, args.output_dir, args.subsample)
    
    # Step 5: Calculate metrics if ground truth is available
    if gt_data is not None:
        metrics = calculate_metrics(remapped_pred, gt_data, args.output_dir)
        
        # Save metrics to JSON
        if metrics:
            import json
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()