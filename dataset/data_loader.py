"""
Data loading utilities for Sentinel-2 image segmentation.
"""
import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SentinelDataset(Dataset):
    """
    Dataset class for Sentinel-2 imagery and land cover segmentation.
    
    Parameters:
    -----------
    image_path : str
        Path to the multiband Sentinel-2 image
    mask_path : str
        Path to the ground truth mask
    validity_mask_path : str, optional
        Path to the validity mask (1=valid, 0=invalid)
    transform : callable, optional
        Optional transform to be applied to the image and mask
    normalize : bool, optional
        Whether to normalize the image according to ResNet pretrained stats
    subset_indices : list, optional
        Indices for subsetting the dataset
    patch_size : int, optional
        Size of the patches to extract (default: 128)
    """
    def __init__(self, image_path, mask_path, validity_mask_path=None, 
                 transform=None, normalize=True, subset_indices=None,
                 patch_size=128):
        self.image_path = image_path
        self.mask_path = mask_path
        self.validity_mask_path = validity_mask_path
        self.transform = transform
        self.normalize = normalize
        self.subset_indices = subset_indices
        self.patch_size = patch_size
        
        # Open files to get metadata
        with rasterio.open(image_path) as src:
            self.image_shape = (src.height, src.width)
            self.num_bands = src.count
            self.image_profile = src.profile
        
        with rasterio.open(mask_path) as src:
            self.mask_shape = (src.height, src.width)
            self.mask_profile = src.profile
            
            # Read unique class values
            self.unique_classes = np.unique(src.read(1))
            self.num_classes = len(self.unique_classes)
        
        if validity_mask_path is not None and os.path.exists(validity_mask_path):
            with rasterio.open(validity_mask_path) as src:
                # Read entire validity mask
                self.validity_mask = src.read(1)
                
                # Get indices of valid pixels
                valid_indices = np.where(self.validity_mask == 1)
                self.valid_coords = list(zip(valid_indices[0], valid_indices[1]))
                
                if subset_indices is not None:
                    # Use provided subset indices
                    self.valid_coords = [self.valid_coords[i] for i in subset_indices]
        else:
            # If no validity mask, consider all pixels valid
            self.validity_mask = None
            y_coords, x_coords = np.meshgrid(
                np.arange(self.mask_shape[0]), 
                np.arange(self.mask_shape[1]), 
                indexing='ij'
            )
            self.valid_coords = list(zip(y_coords.flatten(), x_coords.flatten()))
            
            if subset_indices is not None:
                # Use provided subset indices
                self.valid_coords = [self.valid_coords[i] for i in subset_indices]
        
        # Create normalization transform
        if self.normalize:
            # Use ImageNet mean and std as default 
            # (common for pre-trained models like ResNet)
            self.norm_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406] + [0.5] * (self.num_bands - 3),  # Use 0.5 for non-RGB bands
                std=[0.229, 0.224, 0.225] + [0.5] * (self.num_bands - 3)
            )
        
        print(f"Dataset initialized with {len(self.valid_coords)} valid pixels")
        print(f"Image shape: {self.image_shape}, Mask shape: {self.mask_shape}")
        print(f"Number of bands: {self.num_bands}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class values: {self.unique_classes}")
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.valid_coords)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Get coordinates for this index
        y, x = self.valid_coords[idx]
        
        # Use the patch size from initialization
        patch_size = self.patch_size
        half_size = patch_size // 2
        
        # Calculate patch boundaries with padding for edge cases
        y_min = max(0, y - half_size)
        y_max = min(self.image_shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(self.image_shape[1], x + half_size)
        
        # Read image patch
        with rasterio.open(self.image_path) as src:
            # Read window
            image = src.read(
                window=((y_min, y_max), (x_min, x_max))
            )
        
        # Read mask patch
        with rasterio.open(self.mask_path) as src:
            mask = src.read(
                1,  # Read first band
                window=((y_min, y_max), (x_min, x_max))
            )
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        # Handle any padding needed to get consistent patch size
        if image.shape[1] < patch_size or image.shape[2] < patch_size:
            # Create tensors of the right size filled with zeros
            padded_image = torch.zeros((self.num_bands, patch_size, patch_size), dtype=torch.float32)
            padded_mask = torch.zeros((patch_size, patch_size), dtype=torch.int64)
            
            # Copy the actual data
            h, w = image.shape[1], image.shape[2]
            padded_image[:, :h, :w] = image
            padded_mask[:h, :w] = mask
            
            image = padded_image
            mask = padded_mask
        
        # Apply normalization
        if self.normalize:
            image = self.norm_transform(image)
        
        # Apply transforms if provided
        if self.transform:
            # For albumentations, we need HWC format
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            
            transformed = self.transform(image=image_np, mask=mask_np)
            image = torch.from_numpy(transformed['image']).permute(2, 0, 1)  # Back to CHW
            mask = torch.from_numpy(transformed['mask'])
        
        return image, mask


def create_data_loaders(image_path, mask_path, validity_mask_path=None,
                        batch_size=8, val_split=0.2, test_split=0.1,
                        train_transform=None, val_transform=None,
                        num_workers=4, seed=42, patch_size=128):
    """
    Create DataLoaders for training, validation, and testing.
    
    Parameters:
    -----------
    image_path : str
        Path to the multiband Sentinel-2 image
    mask_path : str
        Path to the ground truth mask
    validity_mask_path : str, optional
        Path to the validity mask (1=valid, 0=invalid)
    batch_size : int, optional
        Batch size for DataLoader
    val_split : float, optional
        Fraction of data to use for validation
    test_split : float, optional
        Fraction of data to use for testing
    train_transform : callable, optional
        Transform to apply to training samples
    val_transform : callable, optional
        Transform to apply to validation and test samples
    num_workers : int, optional
        Number of workers for DataLoader
    seed : int, optional
        Random seed for reproducibility
    patch_size : int, optional
        Size of the patches to extract (default: 128)
    
    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training set
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation set
    test_loader : torch.utils.data.DataLoader
        DataLoader for test set
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a dummy dataset to get valid pixel indices
    dummy_dataset = SentinelDataset(
        image_path=image_path,
        mask_path=mask_path,
        validity_mask_path=validity_mask_path,
        transform=None,
        normalize=False,
        patch_size=patch_size
    )
    
    # Get the total number of valid pixels
    total_indices = list(range(len(dummy_dataset)))
    
    # Split indices into train/val/test
    if test_split > 0:
        # First, split into train+val and test
        train_val_indices, test_indices = train_test_split(
            total_indices, test_size=test_split, random_state=seed
        )
        
        # Then split train+val into train and val
        if val_split > 0:
            # Calculate adjusted validation split
            adjusted_val_split = val_split / (1 - test_split)
            
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=adjusted_val_split, random_state=seed
            )
        else:
            train_indices = train_val_indices
            val_indices = []
    else:
        # No test set, just train/val split
        if val_split > 0:
            train_indices, val_indices = train_test_split(
                total_indices, test_size=val_split, random_state=seed
            )
            test_indices = []
        else:
            train_indices = total_indices
            val_indices = []
            test_indices = []
    
    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    # Create datasets using the split indices
    train_dataset = SentinelDataset(
        image_path=image_path,
        mask_path=mask_path,
        validity_mask_path=validity_mask_path,
        transform=train_transform,
        normalize=True,
        subset_indices=train_indices,
        patch_size=patch_size
    )
    
    val_dataset = SentinelDataset(
        image_path=image_path,
        mask_path=mask_path,
        validity_mask_path=validity_mask_path,
        transform=val_transform,
        normalize=True,
        subset_indices=val_indices,
        patch_size=patch_size
    ) if val_indices else None
    
    test_dataset = SentinelDataset(
        image_path=image_path,
        mask_path=mask_path,
        validity_mask_path=validity_mask_path,
        transform=val_transform,  # Use same transform as validation
        normalize=True,
        subset_indices=test_indices,
        patch_size=patch_size
    ) if test_indices else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader