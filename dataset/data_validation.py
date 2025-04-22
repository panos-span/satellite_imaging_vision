"""
Validation functions for segmentation dataset.

This module provides functions to validate and fix common issues in segmentation datasets,
such as invalid class indices and missing values.
"""

import os
import torch
import numpy as np
from tqdm import tqdm


def validate_and_fix_class_indices(dataset, num_classes, ignore_index=-100):
    """
    Validate and fix class indices in a segmentation dataset.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset to validate
    num_classes : int
        The number of classes in the model
    ignore_index : int
        Value to use for pixels that should be ignored in the loss calculation
        
    Returns:
    --------
    dict
        Dictionary with validation statistics
    """
    print(f"Validating class indices in dataset (num_classes={num_classes})...")
    
    # Statistics
    stats = {
        'total_samples': len(dataset),
        'samples_with_invalid_indices': 0,
        'invalid_indices_found': set(),
        'valid_indices_found': set(),
        'pixels_fixed': 0,
        'total_pixels': 0
    }
    
    for i in tqdm(range(len(dataset)), desc="Validating dataset"):
        try:
            _, mask = dataset[i]
            
            # Convert to numpy for easier manipulation
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = np.array(mask)
            
            # Get unique values
            unique_values = np.unique(mask_np)
            
            # Check for invalid indices
            invalid_mask = (mask_np < 0) | (mask_np >= num_classes)
            
            if np.any(invalid_mask):
                stats['samples_with_invalid_indices'] += 1
                stats['invalid_indices_found'].update(unique_values[unique_values < 0].tolist())
                stats['invalid_indices_found'].update(unique_values[unique_values >= num_classes].tolist())
                stats['pixels_fixed'] += np.sum(invalid_mask)
                
                # Fix invalid indices by setting them to ignore_index
                if hasattr(dataset, 'masks') and isinstance(dataset.masks, list):
                    # If dataset stores masks directly
                    mask_np[invalid_mask] = ignore_index
                    if isinstance(dataset.masks[i], torch.Tensor):
                        dataset.masks[i] = torch.from_numpy(mask_np)
                    else:
                        dataset.masks[i] = mask_np
            
            # Record valid indices
            stats['valid_indices_found'].update(unique_values[(unique_values >= 0) & (unique_values < num_classes)].tolist())
            stats['total_pixels'] += mask_np.size
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    # Convert sets to sorted lists for better readability
    stats['invalid_indices_found'] = sorted(list(stats['invalid_indices_found']))
    stats['valid_indices_found'] = sorted(list(stats['valid_indices_found']))
    
    # Print summary
    print("\nDataset Validation Summary:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with invalid indices: {stats['samples_with_invalid_indices']} ({stats['samples_with_invalid_indices']/stats['total_samples']*100:.2f}%)")
    print(f"Valid class indices found: {stats['valid_indices_found']}")
    print(f"Invalid class indices found: {stats['invalid_indices_found']}")
    print(f"Pixels with invalid indices: {stats['pixels_fixed']} ({stats['pixels_fixed']/stats['total_pixels']*100:.5f}%)")
    
    return stats


def create_ignore_mask_dataset_wrapper(dataset, num_classes, ignore_index=-100):
    """
    Create a dataset wrapper that filters invalid class indices on-the-fly.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The original dataset
    num_classes : int
        The number of classes in the model
    ignore_index : int
        Value to use for pixels that should be ignored in the loss calculation
        
    Returns:
    --------
    torch.utils.data.Dataset
        Wrapped dataset that filters invalid indices
    """
    class IgnoreMaskDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, num_classes, ignore_index):
            self.dataset = dataset
            self.num_classes = num_classes
            self.ignore_index = ignore_index
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            image, mask = self.dataset[idx]
            
            # Convert to tensor if not already
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask)
            
            # Find invalid indices
            invalid_mask = (mask < 0) | (mask >= self.num_classes)
            
            # Set invalid indices to ignore_index
            if invalid_mask.any():
                mask = mask.clone()  # Create a copy to avoid modifying the original
                mask[invalid_mask] = self.ignore_index
            
            return image, mask
    
    return IgnoreMaskDatasetWrapper(dataset, num_classes, ignore_index)