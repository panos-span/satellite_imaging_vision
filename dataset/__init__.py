"""
Dataset module for handling Sentinel-2 imagery for land cover classification.

This module provides utilities for data loading, dataset splitting, augmentation,
and normalization for Sentinel-2 multispectral imagery in land cover classification tasks.
"""
from .data_loader import SentinelDataset, create_data_loaders
from .patch_dataset import SentinelPatchDataset, create_patch_data_loaders
from .dataset_splitter import create_stratified_sample_indices, create_patch_based_splits
from .augmentation import (
    get_train_transform,
    get_val_transform,
    custom_multispectral_augmentation,
    get_multispectral_augmentation_pipeline,
    NormalizeToPretrainedStats
)
from .normalizers import (
    MinMaxScaler,
    StandardScaler,
    Sentinel2Normalizer,
    get_sentinel2_statistics
)

__all__ = [
    # Data loading
    'SentinelDataset',
    'create_data_loaders',
    'SentinelPatchDataset',
    'create_patch_data_loaders',
    
    # Dataset splitting
    'create_stratified_sample_indices',
    'create_patch_based_splits',
    
    # Augmentation
    'get_train_transform',
    'get_val_transform',
    'custom_multispectral_augmentation',
    'get_multispectral_augmentation_pipeline',
    'NormalizeToPretrainedStats',
    
    # Normalization
    'MinMaxScaler',
    'StandardScaler',
    'Sentinel2Normalizer',
    'get_sentinel2_statistics'
]