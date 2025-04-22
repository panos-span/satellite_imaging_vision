"""
Script to run prediction on Sentinel-2 tile T34SEH (Achaia) using the best trained model.

This script will:
1. Use the best model from the frozen_backbone_onecycle experiment
2. Apply the model to predict land cover for T34SEH tile
3. Evaluate against the reference data (GBDA24_ex2_34SEH_ref_data.tif)
4. Generate visualizations and metrics

Usage:
    python run_prediction_achaia.py --data_dir <path_to_dataset> --model_dir <path_to_trained_model>
"""

import os
import sys
import argparse
import subprocess
import glob
import json
import torch
from pathlib import Path

def find_best_model(experiments_dir):
    """Find the best model from the experiment results"""
    print("Searching for best model in experiment results...")
    
    # Look for results.json to find best configuration
    results_path = os.path.join(experiments_dir, "results.json")
    best_config_name = None
    best_model_path = None
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            # Find config with highest validation IoU
            best_config = max(results, key=lambda x: x.get('val_iou', 0))
            best_config_name = best_config.get('config', {}).get('name')
            best_val_iou = best_config.get('val_iou', 0)
            print(f"Found best configuration: {best_config_name} with val_iou: {best_val_iou:.4f}")
            
            # Set the path to the best model
            best_model_path = os.path.join(experiments_dir, best_config_name, "unet_sentinel2_best.pth")
            if not os.path.exists(best_model_path):
                print(f"Warning: Best model file not found at {best_model_path}")
                best_model_path = None
        except Exception as e:
            print(f"Error reading results.json: {e}")
    
    # If results.json doesn't exist or didn't contain valid data, try to find the best model manually
    if not best_model_path:
        print("Searching for models in experiment subdirectories...")
        best_val_iou = -1
        best_config_name = None
        
        for exp_dir in glob.glob(os.path.join(experiments_dir, "*")):
            if os.path.isdir(exp_dir):
                model_file = os.path.join(exp_dir, "unet_sentinel2_best.pth")
                if os.path.exists(model_file):
                    config_name = os.path.basename(exp_dir)
                    print(f"Found model in {config_name}")
                    
                    # Try to load the model checkpoint to get its validation IoU
                    try:
                        checkpoint = torch.load(model_file, map_location='cpu')
                        val_iou = checkpoint.get('val_iou', 0)
                        print(f"  - Validation IoU: {val_iou:.4f}")
                        
                        if val_iou > best_val_iou:
                            best_val_iou = val_iou
                            best_config_name = config_name
                            best_model_path = model_file
                            print(f"  - New best model found!")
                    except Exception as e:
                        print(f"  - Error loading model: {e}")
        
        if best_model_path:
            print(f"Selected best model: {best_config_name} with IoU: {best_val_iou:.4f}")
        else:
            print("No models found in experiment directories, checking main directory...")
            model_path = os.path.join(experiments_dir, "..", "unet_sentinel2_best.pth")
            if os.path.exists(model_path):
                best_model_path = model_path
                print(f"Using model from main directory: {best_model_path}")
            else:
                raise FileNotFoundError("No trained models found in the specified directories")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file not found at {best_model_path}")
    
    print(f"Using model: {best_model_path}")
    return best_model_path

def find_normalizer(data_dir):
    """Find normalizer.pkl in the data directory"""
    normalizer_path = os.path.join(data_dir, "normalizer.pkl")
    if os.path.exists(normalizer_path):
        print(f"Found normalizer at {normalizer_path}")
        return normalizer_path
    
    # Try to find normalizer in parent directories
    parent_dir = os.path.dirname(data_dir)
    normalizer_path = os.path.join(parent_dir, "normalizer.pkl")
    if os.path.exists(normalizer_path):
        print(f"Found normalizer in parent directory: {normalizer_path}")
        return normalizer_path
    
    print("Normalizer not found, prediction will use simple normalization")
    return None

def find_sentinel2_tile(data_dir, tile_id="T34SEH"):
    """Find Sentinel-2 tile data for the specified tile ID"""
    # Look for preprocessed tile or SAFE directory
    for pattern in [
        f"*{tile_id}*.tif",
        f"*{tile_id}*.SAFE",
        f"*{tile_id}*",
        f"S2*{tile_id}*.SAFE",
        f"S2*_{tile_id}_*.SAFE"
    ]:
        matches = list(Path(data_dir).glob(pattern))
        if matches:
            print(f"Found Sentinel-2 data: {matches[0]}")
            return str(matches[0])
    
    # Look in subdirectories
    for pattern in [
        f"**/S2*{tile_id}*.SAFE",
        f"**/S2*_{tile_id}_*.SAFE",
        f"**/*{tile_id}*.tif"
    ]:
        matches = list(Path(data_dir).glob(pattern))
        if matches:
            print(f"Found Sentinel-2 data in subdirectory: {matches[0]}")
            return str(matches[0])
    
    print(f"WARNING: No Sentinel-2 data found for tile {tile_id}")
    return None

def find_reference_data(data_dir, reference_filename="GBDA24_ex2_34SEH_ref_data.tif"):
    """Find reference data for evaluation"""
    # Check directly in data directory
    reference_path = os.path.join(data_dir, reference_filename)
    if os.path.exists(reference_path):
        print(f"Found reference data: {reference_path}")
        return reference_path
    
    # Look in common subdirectories
    for subdir in ["reference", "ground_truth", "validation", "eval"]:
        reference_path = os.path.join(data_dir, subdir, reference_filename)
        if os.path.exists(reference_path):
            print(f"Found reference data in {subdir}: {reference_path}")
            return reference_path
    
    # Recursive search if still not found
    for pattern in [f"**/{reference_filename}", "**/*reference*.tif", "**/*ground_truth*.tif"]:
        matches = list(Path(data_dir).glob(pattern))
        if matches:
            print(f"Found reference data: {matches[0]}")
            return str(matches[0])
    
    print("WARNING: No reference data found for evaluation")
    return None

def get_num_classes(model_dir, data_dir):
    """Determine the number of classes from model or dataset info"""
    # Try to read from experiment results
    results_path = os.path.join(model_dir, "experiments", "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
                if results and 'config' in results[0]:
                    return results[0]['config'].get('num_classes', 9)
        except Exception:
            pass
    
    # Check dataset_info.json in data directory
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, 'r') as f:
                info = json.load(f)
                return info.get('num_classes', 9)
        except Exception:
            pass
    
    # Default to 9 classes (common for land cover)
    print("Using default number of classes: 9")
    return 9

def get_class_names(data_dir, num_classes=9):
    """Get class names for visualization if available"""
    # Check for class_names.json in data directory
    for filename in ["class_names.json", "classes.json", "labels.json"]:
        class_names_path = os.path.join(data_dir, filename)
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
    
    # If not found, create default class names
    class_names = [f"Class {i}" for i in range(num_classes)]
    print(f"Using default class names: {class_names}")
    
    # Save default names for future use
    class_names_path = os.path.join(data_dir, "class_names.json")
    try:
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f, indent=2)
    except Exception:
        pass
    
    return class_names

def main():
    parser = argparse.ArgumentParser(description="Run prediction on Achaia (T34SEH) Sentinel-2 tile")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing dataset and Sentinel-2 data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained models (with experiments subdirectory)")
    parser.add_argument("--output_dir", type=str, default="achaia_prediction",
                        help="Directory to save prediction results")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size for prediction")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for prediction")
    
    args = parser.parse_args()
    
    # Find the best model from experiments
    experiment_dir = os.path.join(args.model_dir, "experiments")
    if not os.path.exists(experiment_dir):
        print(f"Experiments directory not found at {experiment_dir}")
        experiment_dir = args.model_dir
    
    try:
        model_path = find_best_model(experiment_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Looking for model in main directory...")
        model_path = os.path.join(args.model_dir, "unet_sentinel2_best.pth")
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            sys.exit(1)
    
    # Find normalizer
    normalizer_path = find_normalizer(args.data_dir)
    
    # Find Sentinel-2 tile data for T34SEH
    tile_path = find_sentinel2_tile(args.data_dir)
    if not tile_path:
        print("Error: Sentinel-2 tile T34SEH not found")
        sys.exit(1)
    
    # Find reference data
    reference_path = find_reference_data(args.data_dir)
    
    # Get number of classes
    num_classes = get_num_classes(args.model_dir, args.data_dir)
    
    # Get class names
    class_names = get_class_names(args.data_dir, num_classes)
    class_names_path = os.path.join(args.output_dir, "class_names.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Prepare prediction command
    cmd = [
        "python", "predict_tile.py",
        "--model_path", model_path,
        "--input_path", tile_path,
        "--output_dir", args.output_dir,
        "--num_classes", str(num_classes),
        "--patch_size", str(args.patch_size),
        "--batch_size", str(args.batch_size),
        "--processed_data_path", os.path.join(args.output_dir, "processed_sentinel2.tif"),
        "--class_names_file", class_names_path
    ]
    
    if normalizer_path:
        cmd.extend(["--normalizer_path", normalizer_path])
    
    if reference_path:
        cmd.extend(["--reference_path", reference_path])
    
    # Run prediction
    print("\nRunning prediction with command:")
    print(" ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()