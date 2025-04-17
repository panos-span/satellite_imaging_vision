#!/usr/bin/env python3
"""
Script to explicitly remap dataset class values to consecutive indices.

This script is designed for datasets where masks are stored as NumPy .npy files.
It remaps the class values in a dataset directly by creating a new set of masks
with properly remapped class indices.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from glob import glob

def remap_class_values(mask, class_mapping):
    """
    Remap class values in a mask using the provided mapping.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Mask with original class values
    class_mapping : dict
        Dictionary mapping original class values to new indices
        
    Returns:
    --------
    numpy.ndarray
        Mask with remapped class indices
    """
    # Create a new mask with the same shape
    remapped = np.zeros_like(mask)
    
    # Apply the mapping
    for orig_val, new_idx in class_mapping.items():
        remapped[mask == orig_val] = new_idx
        
    return remapped

def create_class_mapping(unique_values, force_include_zero=True):
    """
    Create a mapping from original class values to consecutive indices.
    
    Parameters:
    -----------
    unique_values : list
        List of unique class values found in the dataset
    force_include_zero : bool
        If True, ensure that 0 is included in the mapping
        
    Returns:
    --------
    dict
        Dictionary mapping original values to new indices
    """
    # Sort unique values
    sorted_values = sorted(unique_values)
    
    # Force include 0 if requested and not already present
    if force_include_zero and 0 not in sorted_values:
        sorted_values = [0] + sorted_values
        print("Forcing inclusion of class 0 (background) in the mapping")
    
    # Create mapping
    mapping = {val: idx for idx, val in enumerate(sorted_values)}
    
    print(f"Created class mapping:")
    for val, idx in mapping.items():
        print(f"  Original class {val} -> New class {idx}")
    
    return mapping

def find_unique_class_values(dataset_dir, mask_pattern="**/masks/*.npy", sample_limit=None):
    """
    Find all unique class values in masks in a dataset directory.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing mask files
    mask_pattern : str
        Pattern to match mask files
    sample_limit : int or None
        Maximum number of masks to analyze, None for all
        
    Returns:
    --------
    list
        List of unique class values
    """
    print(f"Analyzing masks in {dataset_dir}...")
    
    # Find all mask files using glob with recursive=True
    mask_paths = glob(os.path.join(dataset_dir, mask_pattern), recursive=True)
    
    if not mask_paths:
        # Try with a simpler pattern
        mask_pattern = "**/*.npy"
        mask_paths = glob(os.path.join(dataset_dir, mask_pattern), recursive=True)
        
    if not mask_paths:
        raise ValueError(f"No mask files found in {dataset_dir} with pattern {mask_pattern}")
    
    print(f"Found {len(mask_paths)} mask files")
    
    # Limit the number of files to analyze if requested
    if sample_limit and len(mask_paths) > sample_limit:
        np.random.shuffle(mask_paths)
        mask_paths = mask_paths[:sample_limit]
        print(f"Analyzing a sample of {sample_limit} mask files")
    
    # Find unique values
    unique_values = set()
    
    # For detecting and displaying mask value distribution
    value_counts = {}
    
    for mask_path in tqdm(mask_paths, desc="Analyzing masks"):
        try:
            # Load the NumPy array
            mask = np.load(mask_path)
            
            # Count occurrences of each value
            values, counts = np.unique(mask, return_counts=True)
            
            # Update counts
            for val, count in zip(values, counts):
                val_int = int(val)
                if val_int in value_counts:
                    value_counts[val_int] += count
                else:
                    value_counts[val_int] = count
            
            # Update unique values set
            unique_values.update(values.tolist())
        except Exception as e:
            print(f"Error reading {mask_path}: {e}")
    
    # Display value distribution
    print("\nClass distribution:")
    total_pixels = sum(value_counts.values())
    for val, count in sorted(value_counts.items()):
        percentage = (count / total_pixels) * 100
        print(f"  Class {val}: {count} pixels ({percentage:.2f}%)")
    
    unique_list = sorted(list(unique_values))
    print(f"\nFound {len(unique_list)} unique class values: {unique_list}")
    
    return unique_list

def remap_dataset(dataset_dir, output_dir, class_mapping, mask_pattern="**/masks/*.npy", copy_images=True):
    """
    Remap class values in masks and save to a new directory.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing original dataset
    output_dir : str
        Directory to save remapped dataset
    class_mapping : dict
        Dictionary mapping original class values to new indices
    mask_pattern : str
        Pattern to match mask files
    copy_images : bool
        Whether to copy the images to the output directory
        
    Returns:
    --------
    None
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mask files
    mask_paths = glob(os.path.join(dataset_dir, mask_pattern), recursive=True)
    
    if not mask_paths:
        # Try with a simpler pattern
        mask_pattern = "**/*.npy"
        mask_paths = glob(os.path.join(dataset_dir, mask_pattern), recursive=True)
    
    print(f"Remapping {len(mask_paths)} mask files...")
    
    # Process each mask
    for mask_path in tqdm(mask_paths, desc="Remapping masks"):
        try:
            # Read mask
            mask = np.load(mask_path)
            
            # Remap class values
            remapped = remap_class_values(mask, class_mapping)
            
            # Create output directory structure
            rel_path = os.path.relpath(mask_path, dataset_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save remapped mask
            np.save(output_path, remapped)
                
            # Copy corresponding image if requested
            if copy_images:
                # Get the images directory by assuming masks are in a 'masks' directory
                base_dir = os.path.dirname(mask_path)
                if 'masks' in base_dir:
                    images_dir = base_dir.replace('masks', 'images')
                    mask_name = os.path.basename(mask_path)
                    image_name = mask_name  # Assuming same name pattern
                    image_path = os.path.join(images_dir, image_name)
                    
                    if os.path.exists(image_path):
                        # Create output image path
                        rel_dir = os.path.relpath(images_dir, dataset_dir)
                        output_image_dir = os.path.join(output_dir, rel_dir)
                        os.makedirs(output_image_dir, exist_ok=True)
                        output_image_path = os.path.join(output_image_dir, image_name)
                        
                        # Copy image
                        shutil.copy2(image_path, output_image_path)
                        
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
    
    # Copy the splits structure if it exists
    for split_dir in ['train', 'val', 'test']:
        src_split_dir = os.path.join(dataset_dir, split_dir)
        if os.path.exists(src_split_dir) and os.path.isdir(src_split_dir):
            dst_split_dir = os.path.join(output_dir, split_dir)
            if not os.path.exists(dst_split_dir):
                # Only copy if it doesn't already exist (avoid duplication)
                shutil.copytree(src_split_dir, dst_split_dir, ignore=shutil.ignore_patterns('*.npy'))
                print(f"Copied {split_dir} directory structure")
    
    # Copy the normalizer.pkl and splits_info.json if they exist
    for file_name in ['normalizer.pkl', 'splits_info.json']:
        src_path = os.path.join(dataset_dir, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, file_name)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file_name} to output directory")
    
    # Copy other important directories
    for dir_name in ['samples']:
        src_path = os.path.join(dataset_dir, dir_name)
        if os.path.exists(src_path) and os.path.isdir(src_path):
            dst_path = os.path.join(output_dir, dir_name)
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
                print(f"Copied {dir_name} directory to output directory")
    
    # Save class mapping
    import json
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, 'w') as f:
        # Convert keys to strings for JSON
        str_mapping = {str(k): int(v) for k, v in class_mapping.items()}
        json.dump(str_mapping, f, indent=2)
    
    # Also save inverse mapping for reference
    inverse_mapping = {v: k for k, v in class_mapping.items()}
    inverse_mapping_path = os.path.join(output_dir, "inverse_class_mapping.json")
    with open(inverse_mapping_path, 'w') as f:
        # Convert keys and values to strings for JSON
        str_inverse_mapping = {str(k): int(v) for k, v in inverse_mapping.items()}
        json.dump(str_inverse_mapping, f, indent=2)
    
    print(f"Class mapping saved to {mapping_path}")
    print(f"Inverse class mapping saved to {inverse_mapping_path}")
    print(f"Remapped dataset saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Remap dataset class values to consecutive indices")
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing the original dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the remapped dataset")
    parser.add_argument("--mask_pattern", type=str, default="**/masks/*.npy",
                        help="Pattern to match mask files")
    parser.add_argument("--no_copy_images", action="store_true",
                        help="Do not copy images to the output directory")
    parser.add_argument("--force_include_zero", action="store_true", default=True,
                        help="Force inclusion of class 0 in the mapping")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Force specific number of classes")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit number of mask files to analyze for class detection")
    
    args = parser.parse_args()
    
    # Find unique class values
    unique_values = find_unique_class_values(args.dataset_dir, args.mask_pattern, args.sample_limit)
    
    # Create class mapping
    class_mapping = create_class_mapping(unique_values, args.force_include_zero)
    
    # Override number of classes if specified
    if args.num_classes is not None:
        if args.num_classes != len(class_mapping):
            print(f"Warning: Overriding detected number of classes ({len(class_mapping)}) with specified value ({args.num_classes})")
    num_classes = args.num_classes if args.num_classes is not None else len(class_mapping)
    
    # Remap dataset
    remap_dataset(
        args.dataset_dir, 
        args.output_dir, 
        class_mapping, 
        args.mask_pattern,
        not args.no_copy_images
    )
    
    print("\nDone! Now you can train on the remapped dataset with exactly", 
          num_classes, "classes.")
    print("Use the following command to train:")
    print(f"python train.py --data_dir {args.output_dir} --save_dir results --num_classes {num_classes}")

if __name__ == "__main__":
    main()