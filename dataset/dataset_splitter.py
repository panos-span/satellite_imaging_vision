"""
Utilities for splitting datasets into training, validation, and testing subsets.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split


def create_stratified_sample_indices(mask_path, validity_mask_path=None, 
                                    val_split=0.2, test_split=0.1, 
                                    stratify=True, random_state=42):
    """
    Create stratified sample indices for dataset splitting.
    
    Parameters:
    -----------
    mask_path : str
        Path to ground truth mask
    validity_mask_path : str, optional
        Path to validity mask (1=valid, 0=invalid)
    val_split : float, optional
        Fraction of data to use for validation
    test_split : float, optional
        Fraction of data to use for testing
    stratify : bool, optional
        Whether to use stratified sampling based on class distribution
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    indices_dict : dict
        Dictionary with train, val, and test indices
    """
    # Read mask to get class distribution
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band
        
        # Get mask dimensions
        height, width = mask.shape
        
        # Get unique classes
        unique_classes = np.unique(mask)
    
    # Read validity mask if provided
    if validity_mask_path and os.path.exists(validity_mask_path):
        with rasterio.open(validity_mask_path) as src:
            validity_mask = src.read(1)
    else:
        # If no validity mask, consider all pixels valid
        validity_mask = np.ones_like(mask)
    
    # Get valid pixel locations
    valid_locations = np.where(validity_mask == 1)
    valid_y, valid_x = valid_locations
    
    # Create array of valid indices
    valid_indices = np.arange(len(valid_y))
    
    # Get corresponding classes for stratification
    valid_classes = mask[valid_y, valid_x]
    
    # Split indices
    if stratify and len(unique_classes) > 1:
        # Stratified split to maintain class distribution
        train_val_indices, test_indices = train_test_split(
            valid_indices, test_size=test_split, random_state=random_state,
            stratify=valid_classes
        )
        
        # Calculate adjusted validation split
        adjusted_val_split = val_split / (1 - test_split)
        
        # Further split train_val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=adjusted_val_split, random_state=random_state,
            stratify=valid_classes[train_val_indices]
        )
    else:
        # Random split without stratification
        train_val_indices, test_indices = train_test_split(
            valid_indices, test_size=test_split, random_state=random_state
        )
        
        # Calculate adjusted validation split
        adjusted_val_split = val_split / (1 - test_split)
        
        # Further split train_val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=adjusted_val_split, random_state=random_state
        )
    
    # Get actual coordinates
    train_coords = [(valid_y[i], valid_x[i]) for i in train_indices]
    val_coords = [(valid_y[i], valid_x[i]) for i in val_indices]
    test_coords = [(valid_y[i], valid_x[i]) for i in test_indices]
    
    # Display statistics
    print(f"Total valid pixels: {len(valid_indices)}")
    print(f"Train set: {len(train_indices)} ({len(train_indices)/len(valid_indices)*100:.2f}%)")
    print(f"Validation set: {len(val_indices)} ({len(val_indices)/len(valid_indices)*100:.2f}%)")
    print(f"Test set: {len(test_indices)} ({len(test_indices)/len(valid_indices)*100:.2f}%)")
    
    # Check class distribution in each split
    if stratify and len(unique_classes) > 1:
        print("\nClass distribution:")
        
        for cls in unique_classes:
            total_count = np.sum(valid_classes == cls)
            train_count = np.sum(valid_classes[train_indices] == cls)
            val_count = np.sum(valid_classes[val_indices] == cls)
            test_count = np.sum(valid_classes[test_indices] == cls)
            
            print(f"  Class {cls}: "
                  f"Total: {total_count} ({total_count/len(valid_indices)*100:.2f}%), "
                  f"Train: {train_count} ({train_count/len(train_indices)*100:.2f}%), "
                  f"Val: {val_count} ({val_count/len(val_indices)*100:.2f}%), "
                  f"Test: {test_count} ({test_count/len(test_indices)*100:.2f}%)")
    
    # Return indices dictionary
    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'train_coords': train_coords,
        'val_coords': val_coords,
        'test_coords': test_coords,
        'valid_y': valid_y,
        'valid_x': valid_x
    }


def create_patch_based_splits(image_path, mask_path, validity_mask_path=None,
                             output_dir='dataset_splits', patch_size=256, 
                             val_split=0.2, test_split=0.1, overlap=0, 
                             stratify=True, random_state=42):
    """
    Create patch-based dataset splits for training, validation, and testing.
    
    Parameters:
    -----------
    image_path : str
        Path to multiband image
    mask_path : str
        Path to ground truth mask
    validity_mask_path : str, optional
        Path to validity mask
    output_dir : str, optional
        Output directory for saving splits
    patch_size : int, optional
        Size of patches to extract
    val_split : float, optional
        Fraction of data to use for validation
    test_split : float, optional
        Fraction of data to use for testing
    overlap : int, optional
        Overlap between patches
    stratify : bool, optional
        Whether to use stratified sampling based on class distribution
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    splits_info : dict
        Dictionary with information about the created splits
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'masks'), exist_ok=True)
    
    # Read image metadata
    with rasterio.open(image_path) as src:
        image_height, image_width = src.height, src.width
        num_bands = src.count
    
    # Read mask metadata
    with rasterio.open(mask_path) as src:
        mask_height, mask_width = src.height, src.width
        unique_classes = np.unique(src.read(1))
    
    # Verify image and mask dimensions match
    if image_height != mask_height or image_width != mask_width:
        raise ValueError(f"Image dimensions ({image_height}x{image_width}) do not match mask dimensions ({mask_height}x{mask_width})")
    
    # Calculate number of patches
    stride = patch_size - overlap
    num_patches_h = 1 + (image_height - patch_size) // stride if image_height > patch_size else 1
    num_patches_w = 1 + (image_width - patch_size) // stride if image_width > patch_size else 1
    
    # Create list of patch coordinates
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch coordinates
            y_start = i * stride
            x_start = j * stride
            
            # Adjust last row/column to fit within image
            if y_start + patch_size > image_height:
                y_start = image_height - patch_size
            if x_start + patch_size > image_width:
                x_start = image_width - patch_size
            
            patches.append((y_start, x_start))
    
    print(f"Created {len(patches)} patches of size {patch_size}x{patch_size} with {overlap} overlap")
    
    # Determine valid patches based on validity mask
    valid_patches = []
    patch_classes = []
    
    if validity_mask_path and os.path.exists(validity_mask_path):
        with rasterio.open(validity_mask_path) as src:
            validity_mask = src.read(1)
        
        # A patch is valid if it contains a minimum percentage of valid pixels
        min_valid_percent = 0.7
        
        for y_start, x_start in patches:
            # Extract validity mask patch
            patch_mask = validity_mask[y_start:y_start+patch_size, x_start:x_start+patch_size]
            valid_percent = np.mean(patch_mask == 1)
            
            if valid_percent >= min_valid_percent:
                valid_patches.append((y_start, x_start))
                
                # Read corresponding mask patch for stratification
                with rasterio.open(mask_path) as src:
                    mask_patch = src.read(1, window=((y_start, y_start+patch_size), 
                                                    (x_start, x_start+patch_size)))
                
                # Use most common class for stratification
                patch_class = np.bincount(mask_patch.flatten()).argmax()
                patch_classes.append(patch_class)
    else:
        # If no validity mask, consider all patches valid
        valid_patches = patches
        
        # Read mask patches for stratification
        for y_start, x_start in valid_patches:
            with rasterio.open(mask_path) as src:
                mask_patch = src.read(1, window=((y_start, y_start+patch_size), 
                                                (x_start, x_start+patch_size)))
            
            # Use most common class for stratification
            patch_class = np.bincount(mask_patch.flatten()).argmax()
            patch_classes.append(patch_class)
    
    patch_classes = np.array(patch_classes)
    print(f"Found {len(valid_patches)} valid patches")
    
    # Split patches into train/val/test
    if stratify and len(unique_classes) > 1:
        # Use stratified sampling
        indices = np.arange(len(valid_patches))
        
        # Split into train+val and test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=random_state,
            stratify=patch_classes
        )
        
        # Calculate adjusted validation split
        adjusted_val_split = val_split / (1 - test_split)
        
        # Split train+val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=adjusted_val_split, random_state=random_state,
            stratify=patch_classes[train_val_indices]
        )
    else:
        # Use random sampling
        indices = np.arange(len(valid_patches))
        
        # Split into train+val and test
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=random_state
        )
        
        # Calculate adjusted validation split
        adjusted_val_split = val_split / (1 - test_split)
        
        # Split train+val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=adjusted_val_split, random_state=random_state
        )
    
    # Get patch coordinates for each split
    train_patches = [valid_patches[i] for i in train_indices]
    val_patches = [valid_patches[i] for i in val_indices]
    test_patches = [valid_patches[i] for i in test_indices]
    
    # Display split statistics
    print(f"Train patches: {len(train_patches)} ({len(train_patches)/len(valid_patches)*100:.2f}%)")
    print(f"Validation patches: {len(val_patches)} ({len(val_patches)/len(valid_patches)*100:.2f}%)")
    print(f"Test patches: {len(test_patches)} ({len(test_patches)/len(valid_patches)*100:.2f}%)")
    
    # Extract and save patches
    splits_info = {
        'train_patches': len(train_patches),
        'val_patches': len(val_patches),
        'test_patches': len(test_patches),
        'patch_size': patch_size,
        'num_bands': num_bands,
        'class_values': unique_classes.tolist()
    }
    
    # Extract and save train patches
    for i, (y_start, x_start) in enumerate(train_patches):
        # Extract image patch
        with rasterio.open(image_path) as src:
            image_patch = src.read(window=((y_start, y_start+patch_size), 
                                           (x_start, x_start+patch_size)))
        
        # Extract mask patch
        with rasterio.open(mask_path) as src:
            mask_patch = src.read(1, window=((y_start, y_start+patch_size), 
                                            (x_start, x_start+patch_size)))
        
        # Save image patch
        patch_filename = f"patch_{i:05d}"
        image_out_path = os.path.join(train_dir, 'images', f"{patch_filename}.npy")
        mask_out_path = os.path.join(train_dir, 'masks', f"{patch_filename}.npy")
        
        np.save(image_out_path, image_patch)
        np.save(mask_out_path, mask_patch)
    
    # Extract and save validation patches
    for i, (y_start, x_start) in enumerate(val_patches):
        # Extract image patch
        with rasterio.open(image_path) as src:
            image_patch = src.read(window=((y_start, y_start+patch_size), 
                                           (x_start, x_start+patch_size)))
        
        # Extract mask patch
        with rasterio.open(mask_path) as src:
            mask_patch = src.read(1, window=((y_start, y_start+patch_size), 
                                            (x_start, x_start+patch_size)))
        
        # Save image patch
        patch_filename = f"patch_{i:05d}"
        image_out_path = os.path.join(val_dir, 'images', f"{patch_filename}.npy")
        mask_out_path = os.path.join(val_dir, 'masks', f"{patch_filename}.npy")
        
        np.save(image_out_path, image_patch)
        np.save(mask_out_path, mask_patch)
    
    # Extract and save test patches
    for i, (y_start, x_start) in enumerate(test_patches):
        # Extract image patch
        with rasterio.open(image_path) as src:
            image_patch = src.read(window=((y_start, y_start+patch_size), 
                                           (x_start, x_start+patch_size)))
        
        # Extract mask patch
        with rasterio.open(mask_path) as src:
            mask_patch = src.read(1, window=((y_start, y_start+patch_size), 
                                            (x_start, x_start+patch_size)))
        
        # Save image patch
        patch_filename = f"patch_{i:05d}"
        image_out_path = os.path.join(test_dir, 'images', f"{patch_filename}.npy")
        mask_out_path = os.path.join(test_dir, 'masks', f"{patch_filename}.npy")
        
        np.save(image_out_path, image_patch)
        np.save(mask_out_path, mask_patch)
    
    # Save split information
    with open(os.path.join(output_dir, 'splits_info.json'), 'w') as f:
        json.dump(splits_info, f)
    
    return splits_info