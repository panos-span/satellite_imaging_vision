"""
Utility functions for working with segmentation datasets.

This module provides helper functions for working with segmentation datasets,
including functions to detect the number of classes and extract class information.
"""

import os
import glob
import numpy as np
import rasterio
import torch
from tqdm import tqdm
from collections import Counter


def detect_classes_from_masks(masks_dir, sample_size=100, mask_suffix="_mask.tif"):
    """
    Detect the number of unique classes in segmentation masks.

    Parameters:
    -----------
    masks_dir : str
        Directory containing mask files
    sample_size : int
        Number of mask files to sample for class detection
    mask_suffix : str
        Suffix for mask files

    Returns:
    --------
    dict
        Dictionary with class information:
        {'num_classes': int, 'class_values': list, 'class_distribution': dict}
    """
    print(f"Detecting classes from masks in {masks_dir}...")

    # Find all mask files
    mask_files = glob.glob(os.path.join(masks_dir, f"*{mask_suffix}"))
    if not mask_files:
        # Try to find any TIFF/TIF files if specific suffix not found
        mask_files = glob.glob(os.path.join(masks_dir, "*.tif")) + glob.glob(
            os.path.join(masks_dir, "*.tiff")
        )

    if not mask_files:
        raise ValueError(f"No mask files found in {masks_dir}")

    # Sample files if there are too many
    if len(mask_files) > sample_size:
        mask_files = np.random.choice(mask_files, sample_size, replace=False)

    # Extract class values from mask files
    class_values = set()
    class_counter = Counter()

    for mask_file in tqdm(mask_files, desc="Reading mask files"):
        try:
            with rasterio.open(mask_file) as src:
                mask = src.read(1)  # Read first band

                # Count unique values
                unique_values, counts = np.unique(mask, return_counts=True)
                for val, count in zip(unique_values, counts):
                    class_values.add(int(val))
                    class_counter[int(val)] += int(count)
        except Exception as e:
            print(f"Error reading mask file {mask_file}: {e}")

    # Convert class values to a sorted list
    class_values = sorted(list(class_values))

    # Calculate total pixels
    total_pixels = sum(class_counter.values())

    # Calculate class distribution as percentages
    class_distribution = {
        str(cls): f"{count / total_pixels * 100:.2f}%"
        for cls, count in class_counter.items()
    }

    # Determine if it's a binary or multiclass problem
    num_classes = len(class_values)
    if num_classes == 2 and 0 in class_values and 1 in class_values:
        # Binary segmentation (background=0, foreground=1)
        is_binary = True
        num_classes = 1  # For binary segmentation, we use 1 output channel with sigmoid
    else:
        # Multiclass segmentation
        is_binary = False

    result = {
        "num_classes": num_classes,
        "class_values": class_values,
        "class_distribution": class_distribution,
        "is_binary": is_binary,
    }

    print(f"Detected {len(class_values)} unique class values: {class_values}")
    print(f"Class distribution: {class_distribution}")
    print(f"Task type: {'Binary' if is_binary else 'Multiclass'} segmentation")
    print(f"Recommended num_classes for model: {num_classes}")

    return result


def detect_classes_from_dataset(dataset, sample_size=100):
    """
    Detect the number of unique classes from a PyTorch dataset.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        PyTorch dataset with mask targets
    sample_size : int
        Number of samples to use for class detection

    Returns:
    --------
    dict
        Dictionary with class information
    """
    print("Detecting classes from dataset...")

    # Extract class values from masks
    class_values = set()
    class_counter = Counter()

    for i in tqdm(range(sample_size), desc="Examining dataset samples"):
        _, mask = dataset[i]

        # Convert to numpy array if it's a tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Handle both single-channel and multi-channel masks
        if mask.ndim > 2 and mask.shape[0] in [1, 3]:
            # Likely in CHW format, take the first channel
            mask = mask[0]

        # Count unique values
        unique_values, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique_values, counts):
            class_values.add(int(val))
            class_counter[int(val)] += int(count)

    # Convert class values to a sorted list
    class_values = sorted(list(class_values))

    # Calculate total pixels
    total_pixels = sum(class_counter.values())

    # Calculate class distribution as percentages
    class_distribution = {
        str(cls): f"{count / total_pixels * 100:.2f}%"
        for cls, count in class_counter.items()
    }

    # Determine if it's a binary or multiclass problem
    num_classes = len(class_values)
    if num_classes == 2 and 0 in class_values and 1 in class_values:
        # Binary segmentation (background=0, foreground=1)
        is_binary = True
        num_classes = 1  # For binary segmentation, we use 1 output channel with sigmoid
    else:
        # Multiclass segmentation
        is_binary = False

    result = {
        "num_classes": num_classes,
        "class_values": class_values,
        "class_distribution": class_distribution,
        "is_binary": is_binary,
    }

    print(f"Detected {len(class_values)} unique class values: {class_values}")
    print(f"Class distribution: {class_distribution}")
    print(f"Task type: {'Binary' if is_binary else 'Multiclass'} segmentation")
    print(f"Recommended num_classes for model: {num_classes}")

    return result
