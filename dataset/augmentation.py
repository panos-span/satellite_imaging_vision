"""
Data augmentation utilities for satellite imagery segmentation.
"""
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


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
    Get a memory-efficient set of data augmentations for training.
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
    
    # Create separate RGB-specific transforms - keeping these simple
    rgb_only_transforms = A.Compose([
        A.HueSaturationValue(p=0.3, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
        # Removed CLAHE which can be memory intensive
    ])
    
    # Memory-efficient geometric augmentations
    transforms_list = [
        # Keep all flips and rotations - these are fast operations
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        
        # REMOVE the resize operation entirely - patches already have correct size
        # A.Resize(height=patch_size, width=patch_size),  # Remove this
        
        # REMOVE expensive elastic/grid transforms completely
        # They're the biggest bottleneck in your pipeline
        
        # Optimize ShiftScaleRotate to be more conservative
        A.ShiftScaleRotate(p=0.2, scale_limit=0.05, rotate_limit=10),
        
        # Keep basic contrast adjustments - relatively fast
        A.RandomBrightnessContrast(p=p),
    ]
    
    # Add CopyPaste augmentation if requested, but with reduced parameters
    if use_copy_paste and LandCoverCopyPaste is not None:
        # Define rare classes that need more augmentation
        rare_classes = [0, 2, 5, 6, 8]  # Most rare classes

        # Modify LandCoverCopyPaste to use the cache
        transforms_list.append(
            LandCoverCopyPaste(
                object_classes=rare_classes,
                p=0.2,
                max_objects_per_image=2,
                min_object_area=150,
            )
        )
    
    # Use memory efficient defaults for composition
    transform = A.Compose(
        transforms_list,
        # Force boolean mask output type for efficiency
        bbox_params=None,  # No bounding box transforms
        keypoint_params=None,  # No keypoint transforms
        additional_targets=None  # No additional targets
    )
    
    return transform


def get_val_transform(patch_size=256):
    """
    Get memory-efficient validation transforms.
    """
    # For validation, use simple resizing instead of cropping
    # This is more memory efficient for large patches
    transform = A.Compose([
        A.Resize(height=patch_size, width=patch_size, p=1.0),
        # No other transforms needed for validation
    ])
    
    return transform


def custom_multispectral_augmentation(image, mask=None, bands_to_augment=None):
    """
    Memory-efficient custom spectral augmentations.
    """
    # Convert to HWC format if needed
    if image.shape[0] < image.shape[-1]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    # Limit bands to augment to save memory - only process key bands
    if bands_to_augment is None:
        # If not specified, focus on important bands (RGB + a few others)
        if image.shape[-1] > 5:
            bands_to_augment = [0, 1, 2, 7, 11]  # RGB + NIR + SWIR
        else:
            bands_to_augment = list(range(image.shape[-1]))
    
    # Create a copy of the image to avoid modifying the original
    augmented_image = image.copy()
    
    # Use a simpler augmentation approach that uses less memory
    # Apply global transformations rather than per-band to save memory
    
    # 1. Global brightness adjustment
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        augmented_image = augmented_image * scale
        
    # 2. Global gamma correction
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        max_val = augmented_image.max() if augmented_image.max() > 0 else 1.0
        augmented_image = np.power(augmented_image / max_val, gamma) * max_val
    
    # Clip to valid range to avoid NaNs and Infs
    augmented_image = np.clip(augmented_image, 0, None)
    
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


def get_multispectral_augmentation_pipeline(spatial_size=512, p=0.5, use_copy_paste=False, object_classes=None, rgb_indices=[0, 1, 2]):
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