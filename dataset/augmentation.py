"""
Data augmentation utilities for satellite imagery segmentation.
"""
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.core.transforms_interface import DualTransform
from typing import Dict, List, Optional, Tuple, Union
import skimage.measure


class RGBTransformWrapper(DualTransform):
    """
    Apply RGB-specific transformations to only the RGB channels of multispectral imagery.
    """
    def __init__(self, rgb_transform, rgb_indices=[0, 1, 2], always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.rgb_transform = rgb_transform
        self.rgb_indices = rgb_indices
        
    def apply(self, img, **params):
        # Check if input is multispectral
        if img.ndim != 3 or img.shape[2] < 3:
            # Not enough channels, return original
            return img
        
        # Check if RGB indices are valid
        valid_indices = [idx for idx in self.rgb_indices if idx < img.shape[2]]
        if len(valid_indices) < 3:
            # Not enough bands for RGB transform, return original
            return img
        
        # Extract RGB channels
        rgb_channels = img[:, :, valid_indices].copy()
        
        # Apply RGB transform
        transformed_rgb = self.rgb_transform(image=rgb_channels)['image']
        
        # Replace RGB channels in original image
        result = img.copy()
        for i, idx in enumerate(valid_indices[:3]):  # Limit to first 3 indices
            result[:, :, idx] = transformed_rgb[:, :, i]
                
        return result
    
    def apply_to_mask(self, mask, **params):
        # Don't modify the mask
        return mask
    
    def get_transform_init_args_names(self):
        return ("rgb_transform", "rgb_indices")


def get_train_transform(p=0.5, patch_size=256, use_copy_paste=False, object_classes=None, rgb_indices=[0, 1, 2]):
    """
    Get a set of data augmentations for training.
    """
    # Import CopyPaste only when needed
    if use_copy_paste:
        try:
            from dataset.copy_paste_augmentation import LandCoverCopyPaste
        except ImportError:
            try:
                from copy_paste_augmentation import LandCoverCopyPaste
            except ImportError:
                print("Warning: CopyPaste augmentation not available. Continuing without it.")
                LandCoverCopyPaste = None
        
        if object_classes is None:
            # Default to classes 1+ (assuming 0 is background)
            object_classes = list(range(1, 10))  # Adjust based on your dataset
    
    # Create separate RGB-specific transforms
    rgb_only_transforms = A.Compose([
        A.CLAHE(p=0.5, clip_limit=4.0),
        A.HueSaturationValue(p=0.3, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
    ])
    
    # Geometric augmentations (safe for any number of channels)
    transforms_list = [
        # Flip and rotate
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        
        # Size transformations
        A.Resize(height=patch_size, width=patch_size),
        
        # Elastic-type transformations
        A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05),
        A.GridDistortion(p=0.2),
        
        # Distortion and rotation
        A.Affine(p=0.3, scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45)),
        
        # Generic radiometric augmentations (work with any number of channels)
        A.RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
        A.GaussNoise(p=0.2, std_range=(0.01, 0.05)),
        A.GaussianBlur(p=0.2, blur_limit=(3, 5)),
        A.RandomGamma(p=0.2, gamma_limit=(80, 120)),
        
        # Channel operations for multispectral
        A.ChannelShuffle(p=0.1),
        A.ChannelDropout(p=0.1, channel_drop_range=(1, 2), fill=0),
        
        # RGB-specific transformations (using our wrapper)
        RGBTransformWrapper(
            rgb_transform=rgb_only_transforms,
            rgb_indices=rgb_indices,
            p=0.5
        ),
    ]
    
    # Add CopyPaste augmentation if requested
    if use_copy_paste:
        try:
            from dataset.copy_paste_augmentation import LandCoverCopyPaste
        except ImportError:
            try:
                from copy_paste_augmentation import LandCoverCopyPaste
            except ImportError:
                print("Warning: CopyPaste augmentation not available. Continuing without it.")
                LandCoverCopyPaste = None
                
            # Define rare classes that need more augmentation
            rare_classes = [0, 2, 3, 5, 6, 8]  # Corresponding to 0, 20, 30, 50, 60, 90
            
            transforms_list.append(
                LandCoverCopyPaste(
                    object_classes=rare_classes,  # Focus on rare classes
                    p=0.7,  # Increase from 0.5
                    max_objects_per_image=5,  # Increase from 3
                    min_object_area=50,  # Decrease from 100
                    blend_mode='gaussian'
                )
            )
    
    # Combine all transforms
    transform = A.Compose(transforms_list)
    
    return transform


def get_val_transform(patch_size=256):
    """
    Get a set of minimal transforms for validation/testing.
    """
    # For validation, we typically don't apply heavy augmentations
    # Just ensure the size is consistent
    transform = A.Compose([
        A.CenterCrop(height=patch_size, width=patch_size, p=1.0),
    ])
    
    return transform


def custom_multispectral_augmentation(image, mask=None, bands_to_augment=None):
    """
    Apply custom spectral augmentations to multispectral satellite imagery.
    """
    # Convert to HWC format if needed
    if image.shape[0] < image.shape[-1]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    # Default to all bands if not specified
    if bands_to_augment is None:
        bands_to_augment = list(range(image.shape[-1]))
    
    # Create a copy of the image to avoid modifying the original
    augmented_image = image.copy()
    
    # Apply band-specific radiometric augmentations
    for band_idx in bands_to_augment:
        # Get the band
        band = augmented_image[..., band_idx]
        
        # Skip empty bands
        if np.all(band == 0):
            continue
        
        # Apply random gamma correction
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            band = np.power(band / band.max(), gamma) * band.max() if band.max() > 0 else band
        
        # Apply random scaling
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            band = band * scale
        
        # Apply random shift
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05) * band.std() if band.std() > 0 else 0
            band = band + shift
        
        # Clip to valid range
        band = np.clip(band, 0, band.max()) if band.max() > 0 else band
        
        # Update the band in the augmented image
        augmented_image[..., band_idx] = band
    
    # Also apply global spectral transformations
    if random.random() < 0.3:
        # Simulate atmospheric effects by adjusting all bands
        atmospheric_factor = random.uniform(0.9, 1.1)
        augmented_image = augmented_image * atmospheric_factor
    
    # Normalize to [0, 1] range for easier handling
    if augmented_image.max() > 0:
        augmented_image = augmented_image / augmented_image.max()
    
    # Return the augmented image and mask
    return (augmented_image, mask) if mask is not None else augmented_image


class NormalizeToPretrainedStats:
    """
    Normalize multispectral images to match pretrained model statistics.
    """
    def __init__(self, rgb_indices=[0, 1, 2], 
                 imagenet_mean=[0.485, 0.456, 0.406],
                 imagenet_std=[0.229, 0.224, 0.225],
                 other_bands_mean=0.5, other_bands_std=0.5):
        self.rgb_indices = rgb_indices
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        self.other_bands_mean = other_bands_mean
        self.other_bands_std = other_bands_std
    
    def __call__(self, image, mask=None):
        # Make a copy of the image to avoid modifying the original
        normalized_image = image.copy()
        
        # Normalize RGB bands using ImageNet statistics
        for i, idx in enumerate(self.rgb_indices):
            if idx < normalized_image.shape[-1]:
                normalized_image[..., idx] = (normalized_image[..., idx] - self.imagenet_mean[i]) / self.imagenet_std[i]
        
        # Normalize other bands
        for i in range(normalized_image.shape[-1]):
            if i not in self.rgb_indices:
                normalized_image[..., i] = (normalized_image[..., i] - self.other_bands_mean) / self.other_bands_std
        
        return (normalized_image, mask) if mask is not None else normalized_image


def get_multispectral_augmentation_pipeline(spatial_size=256, p=0.5, use_copy_paste=False, object_classes=None, rgb_indices=[0, 1, 2]):
    """
    Get a comprehensive augmentation pipeline for multispectral satellite imagery.
    """
    # Get the train transform as a base
    base_transform = get_train_transform(
        p=p, 
        patch_size=spatial_size, 
        use_copy_paste=use_copy_paste, 
        object_classes=object_classes,
        rgb_indices=rgb_indices
    )
    
    # Create custom multispectral normalizer
    normalizer = NormalizeToPretrainedStats(rgb_indices=rgb_indices)
    
    def pipeline(image, mask=None):
        """
        Apply the full augmentation pipeline.
        """
        # Apply base geometric and radiometric transforms
        transformed = base_transform(image=image, mask=mask) if mask is not None else base_transform(image=image)
        
        augmented_image = transformed['image']
        augmented_mask = transformed.get('mask', None)
        
        # Apply custom multispectral augmentation with 50% probability
        if random.random() < 0.5:
            augmented_image, augmented_mask = custom_multispectral_augmentation(
                augmented_image, augmented_mask
            )
        
        # Apply normalization
        augmented_image, augmented_mask = normalizer(augmented_image, augmented_mask)
        
        # Ensure correct spatial size
        if augmented_image.shape[0] != spatial_size or augmented_image.shape[1] != spatial_size:
            # Resize if not already the correct size
            augmented_image = cv2.resize(augmented_image, (spatial_size, spatial_size))
            if augmented_mask is not None:
                augmented_mask = cv2.resize(augmented_mask, (spatial_size, spatial_size), 
                                          interpolation=cv2.INTER_NEAREST)
        
        return {'image': augmented_image, 'mask': augmented_mask} if mask is not None else {'image': augmented_image}
    
    return pipeline