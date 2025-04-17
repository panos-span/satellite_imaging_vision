"""
Dataset classes for handling patch-based data for Sentinel-2 imagery segmentation.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json

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
    def __init__(self, patches_dir, split='train', transform=None, normalize=True):
        self.patches_dir = patches_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Define paths for images and masks
        self.images_dir = os.path.join(patches_dir, split, 'images')
        self.masks_dir = os.path.join(patches_dir, split, 'masks')
        
        # Get list of patch files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.npy')))
        self.mask_files = sorted(glob.glob(os.path.join(self.masks_dir, '*.npy')))
        
        # Sanity check: ensure same number of images and masks
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of image patches ({len(self.image_files)}) does not match " 
                            f"number of mask patches ({len(self.mask_files)})")
        
        # Load class information if available
        self.class_values = None
        info_file = os.path.join(patches_dir, 'splits_info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = json.load(f)
                if 'class_values' in info:
                    self.class_values = info['class_values']
        
        # If class values not found in info file, try to determine from mask files
        if self.class_values is None:
            print("Class values not found in splits_info.json. Determining from mask files...")
            
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
            self.std = [0.229, 0.224, 0.225] + [0.5] * 10   # Assuming 13 bands total
        
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
            mean = torch.tensor(self.mean[:image.shape[0]], dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(self.std[:image.shape[0]], dtype=torch.float32).view(-1, 1, 1)
            
            # Apply normalization
            image = (image - mean) / std
        
        # Apply transforms if provided
        if self.transform:
            # For albumentations, we need HWC format
            image_np = image.permute(1, 2, 0).numpy()

            mask_np = mask.numpy()
            
            transformed = self.transform(image=image_np, mask=mask_np)
            image = torch.from_numpy(transformed['image']).permute(2, 0, 1)  # Back to CHW
            mask = torch.from_numpy(transformed['mask'])
        
        return image, mask


def create_patch_data_loaders(patches_dir, batch_size=8, 
                             train_transform=None, val_transform=None,
                             num_workers=4):
    """
    Create DataLoaders for patch-based datasets.
    
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
    
    Returns:
    --------
    data_loaders : dict
        Dictionary with train, val, and test DataLoaders
    """
    # Create datasets
    train_dataset = SentinelPatchDataset(
        patches_dir=patches_dir,
        split='train',
        transform=train_transform,
        normalize=True
    )
    
    val_dataset = SentinelPatchDataset(
        patches_dir=patches_dir,
        split='val',
        transform=val_transform,
        normalize=True
    )
    
    test_dataset = SentinelPatchDataset(
        patches_dir=patches_dir,
        split='test',
        transform=val_transform,  # Use same transform as validation
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Display dataset information
    print(f"Train set: {len(train_dataset)} patches")
    print(f"Validation set: {len(val_dataset)} patches")
    print(f"Test set: {len(test_dataset)} patches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }