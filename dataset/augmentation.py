"""
Data augmentation utilities for satellite imagery segmentation.

This module provides data augmentation techniques specifically suited
for satellite imagery and semantic segmentation tasks, including both
geometric and radiometric augmentations.
"""
import random
import numpy as np
import cv2
import albumentations as A


def get_train_transform(p=0.5, patch_size=256):
    """
    Get a set of data augmentations for training.
    
    This function creates a composition of geometric and radiometric
    augmentations for satellite imagery, ensuring that the mask is 
    transformed consistently with the image.
    
    Parameters:
    -----------
    p : float, optional
        Probability of applying each transform (default: 0.5)
    patch_size : int, optional
        Size of the output patches (default: 256)
    
    Returns:
    --------
    transform : albumentations.Compose
        Composition of transforms
    """
    # Geometric augmentations
    geometric_transforms = [
        # Flip and rotate
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        
        # Elastic-type transformations
        A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.2),
        
        # Distortion and cropping
        A.ShiftScaleRotate(p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=p)
    ]
    
    # Radiometric augmentations (applied only to the image, not the mask)
    radiometric_transforms = [
        # Brightness and contrast
        A.RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
        
        # Color jittering
        A.HueSaturationValue(p=p, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
        
        # Noise and blur
        A.GaussNoise(p=0.2, var_limit=(10.0, 50.0)),
        A.GaussianBlur(p=0.2, blur_limit=(3, 5)),
        
        # Gamma and color adjustments
        A.RandomGamma(p=0.2, gamma_limit=(80, 120)),
        A.CLAHE(p=0.2, clip_limit=4.0),
        
        # Channel shuffling/dropping for multispectral images
        A.ChannelShuffle(p=0.1),
        A.ChannelDropout(p=0.1, channel_drop_range=(1, 2), fill_value=0),
    ]
    
    # Combine geometric and radiometric transforms
    transform = A.Compose(geometric_transforms + radiometric_transforms)
    
    return transform


def get_val_transform(patch_size=256):
    """
    Get a set of minimal transforms for validation/testing.
    
    This typically involves only normalization with potentially
    some light augmentations to improve robustness.
    
    Parameters:
    -----------
    patch_size : int, optional
        Size of the output patches (default: 256)
    
    Returns:
    --------
    transform : albumentations.Compose
        Composition of transforms
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
    
    This function allows for band-specific augmentations, which is useful
    for specialized satellite imagery tasks.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image of shape (channels, height, width) or (height, width, channels)
    mask : numpy.ndarray, optional
        Input mask
    bands_to_augment : list, optional
        List of band indices to augment
        
    Returns:
    --------
    augmented_image : numpy.ndarray
        Augmented image
    augmented_mask : numpy.ndarray or None
        Augmented mask (or None if no mask was provided)
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
            band = np.power(band / band.max(), gamma) * band.max()
        
        # Apply random scaling
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            band = band * scale
        
        # Apply random shift
        if random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05) * band.std()
            band = band + shift
        
        # Clip to valid range
        band = np.clip(band, 0, band.max())
        
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
    
    This transformer normalizes RGB bands using ImageNet statistics
    and normalizes other bands to a standardized range.
    
    Parameters:
    -----------
    rgb_indices : list, optional
        Indices of RGB bands in the input image
    imagenet_mean : list, optional
        Mean values for RGB channels from ImageNet
    imagenet_std : list, optional
        Standard deviation values for RGB channels from ImageNet
    other_bands_mean : float, optional
        Mean value for non-RGB bands
    other_bands_std : float, optional
        Standard deviation for non-RGB bands
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
        """
        Apply normalization to the image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image of shape (height, width, channels)
        mask : numpy.ndarray, optional
            Input mask
            
        Returns:
        --------
        normalized_image : numpy.ndarray
            Normalized image
        mask : numpy.ndarray or None
            Unchanged mask (or None if no mask was provided)
        """
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


def get_multispectral_augmentation_pipeline(spatial_size=256, p=0.5):
    """
    Get a comprehensive augmentation pipeline for multispectral satellite imagery.
    
    This function combines geometric transformations, radiometric augmentations,
    and specialized multispectral transformations suitable for satellite imagery
    and semantic segmentation tasks.
    
    Parameters:
    -----------
    spatial_size : int, optional
        Spatial size for cropping/resizing (default: 256)
    p : float, optional
        Probability of applying each transform (default: 0.5)
        
    Returns:
    --------
    pipeline : callable
        Augmentation pipeline function
    """
    # Create base augmentation with Albumentations including spatial_size parameter
    base_transform = A.Compose([
        # Include a resize/crop operation to ensure consistent spatial size
        A.RandomResizedCrop(height=spatial_size, width=spatial_size, 
                          scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
        A.CenterCrop(height=spatial_size, width=spatial_size, p=1.0),
        
        # Other augmentations from the train transform
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.2),
        A.ShiftScaleRotate(p=0.3, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(p=p, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
        A.GaussNoise(p=0.2, var_limit=(10.0, 50.0)),
        A.GaussianBlur(p=0.2, blur_limit=(3, 5)),
        A.RandomGamma(p=0.2, gamma_limit=(80, 120)),
        A.CLAHE(p=0.2, clip_limit=4.0),
        A.ChannelShuffle(p=0.1),
        A.ChannelDropout(p=0.1, channel_drop_range=(1, 2), fill_value=0),
    ])
    
    # Create custom multispectral normalizer
    normalizer = NormalizeToPretrainedStats()
    
    def pipeline(image, mask=None):
        """
        Apply the full augmentation pipeline.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image
        mask : numpy.ndarray, optional
            Input mask
            
        Returns:
        --------
        augmented_image : numpy.ndarray
            Augmented image
        augmented_mask : numpy.ndarray or None
            Augmented mask (or None if no mask was provided)
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