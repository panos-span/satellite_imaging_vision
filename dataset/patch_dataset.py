"""
Dataset classes for handling patch-based data for Sentinel-2 imagery segmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import pickle
import json

def calculate_class_pixel_counts(dataset, num_samples=500):
    """
    Calculate class distribution from dataset for loss weighting.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        Dataset to analyze
    num_samples : int
        Number of samples to analyze for efficiency
    
    Returns:
    --------
    dict, int
        Dictionary with class counts and total pixels analyzed
    """
    # Limit samples for efficiency
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Initialize counts
    class_counts = {}
    total_pixels = 0
    
    # Process samples
    for i in tqdm(indices, desc="Analyzing class distribution"):
        _, mask = dataset[i]
        
        # Convert to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        # Count total valid pixels
        total_pixels += mask_np.size
        
        # Count per class
        unique_values, counts = np.unique(mask_np, return_counts=True)
        for cls, count in zip(unique_values, counts):
            cls_int = int(cls)
            if cls_int not in class_counts:
                class_counts[cls_int] = 0
            class_counts[cls_int] += int(count)
    
    return class_counts, total_pixels


class SentinelPatchDataset(Dataset):
    """
    Dataset class for handling pre-extracted image and mask patches.

    Parameters:
    -----------
    patches_dir : str
        Directory containing image and mask patches
    split : str
        Dataset split ('train', 'val', or 'test')
    transform : callable, optional
        Transform to apply to images and masks
    normalize : bool, optional
        Whether to normalize the images
    """

    def __init__(self, patches_dir, split="train", transform=None, normalize=True):
        self.patches_dir = patches_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize

        # Define paths for images and masks
        self.images_dir = os.path.join(patches_dir, split, "images")
        self.masks_dir = os.path.join(patches_dir, split, "masks")

        # Get list of patch files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(self.masks_dir, "*.npy")))

        # Sanity check: ensure same number of images and masks
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(
                f"Number of image patches ({len(self.image_files)}) does not match "
                f"number of mask patches ({len(self.mask_files)})"
            )

        # Load class information if available
        self.class_values = None
        info_file = os.path.join(patches_dir, "splits_info.json")
        if os.path.exists(info_file):
            with open(info_file, "r") as f:
                info = json.load(f)
                if "class_values" in info:
                    self.class_values = info["class_values"]

        # If class values not found in info file, try to determine from mask files
        if self.class_values is None:
            print(
                "Class values not found in splits_info.json. Determining from mask files..."
            )

            # Load a sample mask to determine classes
            if self.mask_files:
                sample_mask = np.load(self.mask_files[0])
                self.class_values = np.unique(sample_mask).tolist()
                print(f"Determined class values: {self.class_values}")

        print(f"Initialized {split} dataset with {len(self.image_files)} patches")

        # Determine normalization parameters
        if self.normalize:
            # Use ImageNet mean and std for RGB bands, 0.5 for other bands
            self.mean = [0.485, 0.456, 0.406] + [0.5] * 10  # Assuming 13 bands total
            self.std = [0.229, 0.224, 0.225] + [0.5] * 10  # Assuming 13 bands total

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a patch and its mask by index."""
        # Load image and mask
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Apply normalization if requested
        if self.normalize:
            # Reshape mean and std to match image dimensions
            mean = torch.tensor(self.mean[: image.shape[0]], dtype=torch.float32).view(
                -1, 1, 1
            )
            std = torch.tensor(self.std[: image.shape[0]], dtype=torch.float32).view(
                -1, 1, 1
            )

            # Apply normalization
            image = (image - mean) / std

        # Apply transforms if provided
        if self.transform:
            # For albumentations, we need HWC format
            image_np = image.permute(1, 2, 0).numpy()

            mask_np = mask.numpy()

            transformed = self.transform(image=image_np, mask=mask_np)
            image = torch.from_numpy(transformed["image"]).permute(
                2, 0, 1
            )  # Back to CHW
            mask = torch.from_numpy(transformed["mask"])

        return image, mask


def create_patch_data_loaders(
    patches_dir,
    batch_size=4,
    train_transform=None,
    val_transform=None,
    num_workers=os.cpu_count(),
    use_rare_class_sampler=True,
    use_improved_sampler=True,  # Added parameter to choose between samplers
    oversample_factor=3,
):
    """
    Create DataLoaders for patch-based datasets with improved class balancing.
    
    Parameters:
    -----------
    patches_dir : str
        Directory containing image and mask patches
    batch_size : int, optional
        Batch size for DataLoader
    train_transform : callable, optional
        Transform to apply to training samples
    val_transform : callable, optional
        Transform to apply to validation and test samples
    num_workers : int, optional
        Number of workers for DataLoader
    use_rare_class_sampler : bool, optional
        Whether to use any rare class sampler (either RareClassSampler or ImprovedRareClassSampler)
    use_improved_sampler : bool, optional
        Whether to use the ImprovedRareClassSampler instead of RareClassSampler
    oversample_factor : int, optional
        Oversampling factor for RareClassSampler (not used for ImprovedRareClassSampler)

    Returns:
    --------
    data_loaders : dict
        Dictionary with train, val, and test DataLoaders
    """
    # Create datasets
    train_dataset = SentinelPatchDataset(
        patches_dir=patches_dir,
        split="train",
        transform=train_transform,
        normalize=True,
    )
    
    val_dataset = SentinelPatchDataset(
        patches_dir=patches_dir, 
        split="val", 
        transform=val_transform, 
        normalize=True
    )
    
    test_dataset = SentinelPatchDataset(
        patches_dir=patches_dir,
        split="test",
        transform=val_transform,  # Use same transform as validation
        normalize=True,
    )
    
    def precompute_rare_class_indices(dataset, rare_classes=[0 , 2, 5, 6, 8]):
        """Pre-compute and cache indices of samples containing rare classes"""
        cache_file = os.path.join(os.path.dirname(dataset.patches_dir), "rare_class_cache.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading pre-computed rare class indices from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Pre-computing rare class indices for faster training...")
        rare_class_indices = {cls: [] for cls in rare_classes}
        
        for i in tqdm(range(len(dataset))):
            _, mask = dataset[i]
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            
            for cls in rare_classes:
                if (mask_np == cls).any():
                    rare_class_indices[cls].append(i)
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(rare_class_indices, f)
        
        return rare_class_indices
    
    # Add this code right after creating the train_dataset
    rare_classes = [0 , 2, 5, 6, 8]  # Adjust based on your dataset
    rare_class_indices = precompute_rare_class_indices(train_dataset, rare_classes)
    print(f"Found {sum(len(indices) for indices in rare_class_indices.values())} samples with rare classes")
    
    # For the train loader, use our custom sampler if requested
    if use_rare_class_sampler:
        if use_improved_sampler:
            # Calculate class distribution for the improved sampler
            print("Calculating class distribution for ImprovedRareClassSampler...")
            class_counts, total_pixels = calculate_class_pixel_counts(train_dataset, num_samples=500)
            print(f"Total pixels analyzed: {total_pixels}")
            
            for cls, count in sorted(class_counts.items()):
                print(f"  Class {cls}: {count:,} pixels ({count/total_pixels*100:.2f}%)")
            
            # Create the improved sampler
            sampler = ImprovedRareClassSampler(
                train_dataset,
                class_counts,
                min_factor=1,
                max_factor=6,
                precomputed_rare_indices=rare_class_indices
            )
            print("Using ImprovedRareClassSampler for better class balance")
        else:
            # Use the original RareClassSampler
            sampler = ImprovedRareClassSampler(
                train_dataset,
                oversample_factor=oversample_factor,
            )
            print(f"Using RareClassSampler with oversample_factor={oversample_factor}")
        
        # Create dataloader with the selected sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use the selected sampler
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # Original train_loader with shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        pin_memory=True,
    )
    
    # Display dataset information
    print(f"Train set: {len(train_dataset)} patches")
    print(f"Validation set: {len(val_dataset)} patches")
    print(f"Test set: {len(test_dataset)} patches")
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}

def find_rare_class_patches(dataset_path, rare_classes=[0, 2, 5, 6, 8], output_file="rare_patches.json"):
    """Run this once to identify and save rare class patch indices."""
    import json
    from tqdm import tqdm
    import numpy as np
    
    # Get paths to all mask files
    mask_dir = os.path.join(dataset_path, "train", "masks")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
    
    rare_class_indices = []
    
    # Process each mask file
    for i, mask_file in enumerate(tqdm(mask_files, desc="Finding rare patches")):
        # Load mask
        mask = np.load(mask_file)
        
        # Check if mask contains any rare class
        for cls in rare_classes:
            if (mask == cls).any():
                rare_class_indices.append(i)
                break
    
    # Save indices to file
    with open(output_file, "w") as f:
        json.dump({"rare_class_indices": rare_class_indices}, f)
    
    #print(f"Found {len(rare_class_indices)} patches with rare classes out of {len(mask_files)} total")
    return rare_class_indices

# Run this once before training

# Add this class to dataset/patch_dataset.py
class ImprovedRareClassSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, class_counts, min_factor=1, max_factor=5, precomputed_rare_indices=None):
        """
        Class-aware sampler with logarithmic scaling for better handling of extreme imbalances.
        
        Parameters:
        -----------
        dataset : torch.utils.data.Dataset
            Dataset to sample from
        class_counts : dict
            Dictionary of class pixel counts from analysis
        min_factor : int
            Minimum oversampling factor
        max_factor : int
            Maximum oversampling factor
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        
        # Get all classes
        all_classes = sorted(class_counts.keys())
        
        # Calculate total pixels
        total_pixels = sum(class_counts.values())
        
        # Calculate class frequencies
        class_freqs = {cls: count/total_pixels for cls, count in class_counts.items()}
        
        # NEW: Use logarithmic scaling for better handling of extreme imbalances
        log_freqs = {cls: -np.log10(freq) for cls, freq in class_freqs.items()}
        min_log, max_log = min(log_freqs.values()), max(log_freqs.values())
        range_log = max_log - min_log
        
        # Map to sampling factors using log scale
        self.sampling_factors = {}
        for cls, log_val in log_freqs.items():
            # Normalize to [0,1] range
            if range_log > 0:
                normalized = (log_val - min_log) / range_log
                # Map to [min_factor, max_factor]
                factor = min_factor + normalized * (max_factor - min_factor)
            else:
                factor = min_factor
            self.sampling_factors[cls] = int(np.ceil(factor))
        
        print("Class sampling factors:")
        for cls, factor in sorted(self.sampling_factors.items()):
            print(f"  Class {cls}: {factor}x")
            
        # Store precomputed indices if provided
        self.precomputed_rare_indices = precomputed_rare_indices
        
        # Create class-specific indices
        self.class_indices = self._find_class_indices(dataset, all_classes)
        
    # The rest of the class remains unchanged
    def _find_class_indices(self, dataset, all_classes):
        """Find indices of samples containing each class"""
        class_indices = {cls: [] for cls in all_classes}
        
        # If we have precomputed indices for rare classes, use them
        if self.precomputed_rare_indices:
            # First, add the precomputed rare class indices
            for cls, indices in self.precomputed_rare_indices.items():
                if cls in class_indices:
                    class_indices[cls] = indices
        
        # For classes without precomputed indices, find them as before
        classes_to_find = [cls for cls in all_classes if cls not in self.precomputed_rare_indices]
        
        if classes_to_find:
            for i in tqdm(range(len(dataset)), desc="Analyzing dataset for sampling"):
                _, mask = dataset[i]
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.numpy()
                else:
                    mask_np = mask
                
                # Find classes in this mask
                unique_classes = np.unique(mask_np)
                
                # Add index to each class's list
                for cls in unique_classes:
                    cls_int = int(cls)
                    if cls_int in classes_to_find and cls_int in class_indices:
                        class_indices[cls_int].append(i)
        
        return class_indices
    
    def __iter__(self):
        # Create combined indices with appropriate oversampling
        combined_indices = list(self.indices)  # Start with all indices
        
        # Add oversampled rare classes
        for cls, factor in self.sampling_factors.items():
            if factor > 1 and cls in self.class_indices:
                # Add this class's indices multiple times based on factor
                indices_to_add = self.class_indices[cls] * (factor - 1)
                combined_indices.extend(indices_to_add)
        
        # Shuffle to avoid training on the same samples consecutively
        np.random.shuffle(combined_indices)
        return iter(combined_indices)
    
    def __len__(self):
        total_len = len(self.indices)
        for cls, factor in self.sampling_factors.items():
            if factor > 1 and cls in self.class_indices:
                total_len += len(self.class_indices[cls]) * (factor - 1)
        return total_len