"""
Utility functions for handling and validating segmentation masks.

This module provides functions for cleaning, validating, and inspecting
segmentation masks to handle issues with invalid class indices.
"""

import torch
import numpy as np


def print_mask_stats(mask, name="mask"):
    """Print statistics about a mask to help with debugging."""
    if isinstance(mask, torch.Tensor):
        unique_values = torch.unique(mask).cpu().numpy()
        min_val = mask.min().item()
        max_val = mask.max().item()
    else:
        unique_values = np.unique(mask)
        min_val = mask.min()
        max_val = mask.max()
        
    print(f"{name} statistics:")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  Min: {min_val}, Max: {max_val}")
    print(f"  Unique values: {unique_values}")
    

def validate_mask(mask, num_classes, name="mask"):
    """
    Validate a segmentation mask for valid class indices.
    
    Parameters:
    -----------
    mask : torch.Tensor
        The segmentation mask to validate
    num_classes : int
        The number of valid classes
    name : str
        Name of the mask for logging purposes
        
    Returns:
    --------
    bool
        True if mask is valid, False otherwise
    """
    if isinstance(mask, torch.Tensor):
        min_val = mask.min().item()
        max_val = mask.max().item()
    else:
        min_val = mask.min()
        max_val = mask.max()
    
    is_valid = (min_val >= 0) and (max_val < num_classes)
    
    if not is_valid:
        print(f"WARNING: Invalid {name} detected with values outside range [0, {num_classes-1}]!")
        print(f"  Min: {min_val}, Max: {max_val}")
        
        if isinstance(mask, torch.Tensor):
            unique_values = torch.unique(mask).cpu().numpy()
        else:
            unique_values = np.unique(mask)
            
        print(f"  Unique values: {unique_values}")
        
    return is_valid


def clean_mask(mask, num_classes, ignore_index=-100):
    """
    Clean a segmentation mask by mapping invalid indices to the ignore_index.
    
    Parameters:
    -----------
    mask : torch.Tensor
        The segmentation mask to clean
    num_classes : int
        The number of valid classes
    ignore_index : int
        The index to use for invalid pixels
        
    Returns:
    --------
    torch.Tensor
        The cleaned mask
    """
    # Make a copy to avoid modifying the original
    cleaned = mask.clone() if isinstance(mask, torch.Tensor) else mask.copy()
    
    # Map invalid indices to ignore_index
    if isinstance(cleaned, torch.Tensor):
        cleaned[cleaned < 0] = ignore_index
        cleaned[cleaned >= num_classes] = ignore_index
    else:
        cleaned[cleaned < 0] = ignore_index
        cleaned[cleaned >= num_classes] = ignore_index
        
    return cleaned


def inspect_dataset_masks(dataset, num_samples=10, num_classes=None):
    """
    Inspect masks in a dataset to detect issues with class indices.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset to inspect
    num_samples : int
        Number of samples to inspect
    num_classes : int
        Number of expected classes
        
    Returns:
    --------
    dict
        Dictionary with inspection results
    """
    results = {
        "total_samples": min(num_samples, len(dataset)),
        "invalid_masks": 0,
        "unique_values": set(),
        "min_value": float('inf'),
        "max_value": float('-inf')
    }
    
    # Inspect random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        _, mask = dataset[idx]
        
        # Convert to numpy if it's a tensor
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4 and mask.shape[0] == 1:  # BCHW
                mask = mask[0]
                
            if mask.dim() == 3 and mask.shape[0] == 1:  # CHW
                mask = mask[0]
                
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
            
        # Update statistics
        unique_vals = set(np.unique(mask_np).tolist())
        results["unique_values"] = results["unique_values"].union(unique_vals)
        results["min_value"] = min(results["min_value"], np.min(mask_np))
        results["max_value"] = max(results["max_value"], np.max(mask_np))
        
        # Check for invalid values
        if num_classes is not None:
            if np.min(mask_np) < 0 or np.max(mask_np) >= num_classes:
                results["invalid_masks"] += 1
    
    # Print summary
    print(f"Dataset mask inspection results:")
    print(f"  Inspected: {results['total_samples']} samples")
    print(f"  Invalid masks: {results['invalid_masks']}")
    print(f"  Unique values: {sorted(list(results['unique_values']))}")
    print(f"  Value range: [{results['min_value']}, {results['max_value']}]")
    
    # Recommendation for num_classes
    if num_classes is None:
        recommended = max(results["max_value"] + 1, len(results["unique_values"]))
        results["recommended_num_classes"] = int(recommended)
        print(f"  Recommended num_classes: {recommended}")
    
    return results