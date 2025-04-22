"""
Utilities for remapping class indices in segmentation masks.

This module provides functions for remapping sparse or non-consecutive
class indices to dense indices starting from 0, which is more efficient
for neural network training.
"""

import numpy as np
import torch
from tqdm import tqdm


class ClassRemapper:
    """
    Class for remapping sparse class indices to dense indices.

    This is useful when your class labels are not consecutive integers
    starting from 0, which is the expected format for most loss functions
    and metrics in deep learning.
    """

    def __init__(self, class_values=None):
        """
        Initialize the class remapper.

        Parameters:
        -----------
        class_values : list or None
            List of unique class values to remap. If None, will be determined
            from the first batch of data.
        """
        self.class_values = class_values
        self.class_mapping = None
        self.reverse_mapping = None

        if class_values is not None:
            self._create_mapping(class_values)

    def _create_mapping(self, class_values):
        """Create the mapping dictionaries."""
        # Sort unique values to ensure consistent mapping
        unique_values = sorted(list(set(class_values)))

        # Create mapping from original values to new indices
        self.class_mapping = {val: idx for idx, val in enumerate(unique_values)}

        # Create reverse mapping for converting back
        self.reverse_mapping = {idx: val for val, idx in self.class_mapping.items()}

        print(f"Created class mapping with {len(unique_values)} classes:")
        print(f"  Original values: {unique_values}")
        print(f"  Mapped to: {list(range(len(unique_values)))}")

    def fit(self, dataset, num_samples=100):
        """
        Analyze dataset to determine unique class values.

        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            Dataset containing masks to analyze
        num_samples : int
            Number of samples to analyze

        Returns:
        --------
        self
        """
        if self.class_mapping is not None:
            print("Class mapping already initialized, skipping fit.")
            return self

        # Collect unique values from dataset
        unique_values = set()

        # Limit samples to analyze
        indices = np.random.choice(
            len(dataset), min(num_samples, len(dataset)), replace=False
        )

        for idx in tqdm(indices, desc="Analyzing masks for class mapping"):
            _, mask = dataset[idx]

            # Convert tensor to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            # Add unique values to set
            unique_values.update(np.unique(mask_np).tolist())

        # Create mapping
        self._create_mapping(unique_values)

        return self

    def remap_mask(self, mask):
        """
        Remap a mask to use dense class indices.

        Parameters:
        -----------
        mask : torch.Tensor or numpy.ndarray
            Mask with original class values

        Returns:
        --------
        torch.Tensor or numpy.ndarray
            Mask with remapped class indices
        """
        if self.class_mapping is None:
            raise ValueError("Class mapping not initialized. Call fit() first.")

        # Check if tensor or numpy array
        is_tensor = isinstance(mask, torch.Tensor)
        device = mask.device if is_tensor else None

        # Convert to numpy for mapping
        if is_tensor:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask.copy()

        # Create remapped mask
        remapped = np.zeros_like(mask_np)

        # Apply mapping
        for orig_val, new_idx in self.class_mapping.items():
            remapped[mask_np == orig_val] = new_idx

        # Convert back to tensor if needed
        if is_tensor:
            remapped = torch.tensor(remapped, device=device)

        return remapped

    def reverse_remap(self, mask):
        """
        Remap a mask back to original class values.

        Parameters:
        -----------
        mask : torch.Tensor or numpy.ndarray
            Mask with remapped class indices

        Returns:
        --------
        torch.Tensor or numpy.ndarray
            Mask with original class values
        """
        if self.reverse_mapping is None:
            raise ValueError("Class mapping not initialized. Call fit() first.")

        # Check if tensor or numpy array
        is_tensor = isinstance(mask, torch.Tensor)
        device = mask.device if is_tensor else None

        # Convert to numpy for mapping
        if is_tensor:
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask.copy()

        # Create remapped mask
        original = np.zeros_like(mask_np)

        # Apply reverse mapping
        for new_idx, orig_val in self.reverse_mapping.items():
            original[mask_np == new_idx] = orig_val

        # Convert back to tensor if needed
        if is_tensor:
            original = torch.tensor(original, device=device)

        return original

    @property
    def num_classes(self):
        """Number of classes after remapping."""
        if self.class_mapping is None:
            return 0
        return len(self.class_mapping)

    def save(self, path):
        """Save the class remapper to a file."""
        if self.class_mapping is None:
            raise ValueError("Class mapping not initialized. Call fit() first.")

        import json

        # Convert keys to strings for JSON serialization
        class_mapping = {str(k): int(v) for k, v in self.class_mapping.items()}
        reverse_mapping = {str(k): int(v) for k, v in self.reverse_mapping.items()}

        mapping_data = {
            "class_mapping": class_mapping,
            "reverse_mapping": reverse_mapping,
            "class_values": sorted(list(self.class_mapping.keys())),
        }

        with open(path, "w") as f:
            json.dump(mapping_data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load a class remapper from a file."""
        import json

        with open(path, "r") as f:
            mapping_data = json.load(f)

        # Convert keys back from strings
        class_mapping = {int(k): v for k, v in mapping_data["class_mapping"].items()}
        reverse_mapping = {
            int(k): v for k, v in mapping_data["reverse_mapping"].items()
        }

        remapper = cls()
        remapper.class_mapping = class_mapping
        remapper.reverse_mapping = reverse_mapping
        remapper.class_values = sorted(list(class_mapping.keys()))

        return remapper


def create_class_remapper(dataset, num_samples=100):
    """
    Create and fit a class remapper from a dataset.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        Dataset containing masks to analyze
    num_samples : int
        Number of samples to analyze

    Returns:
    --------
    ClassRemapper
        Fitted class remapper
    """
    remapper = ClassRemapper()
    remapper.fit(dataset, num_samples)
    return remapper


def remap_dataset_classes(dataset, remapper):
    """
    Apply a class remapper to a full dataset.

    Warning: This modifies the dataset in-place!

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        Dataset containing masks to remap
    remapper : ClassRemapper
        Class remapper to use

    Returns:
    --------
    dataset
        The modified dataset
    """
    # Check if dataset supports indexing
    try:
        dataset[0]
    except (TypeError, IndexError):
        raise ValueError("Dataset must support indexing")

    # Check if dataset returns a tuple
    if not isinstance(dataset[0], tuple):
        raise ValueError("Dataset items must be tuples of (image, mask)")

    # Apply remapping to each mask
    for i in tqdm(range(len(dataset)), desc="Remapping dataset classes"):
        img, mask = dataset[i]

        # Remap mask
        remapped_mask = remapper.remap_mask(mask)

        # Update dataset
        # Note: This only works if the dataset supports item assignment,
        # otherwise you'll need to create a new dataset
        dataset.update_item(i, img, remapped_mask)

    return dataset
