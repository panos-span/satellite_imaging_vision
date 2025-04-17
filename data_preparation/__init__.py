"""
Data preparation module for Sentinel-2 imagery processing.

This module provides utilities for processing Sentinel-2 data for land cover classification,
including data checking, preprocessing, and visualization.
"""
from .geospatial_utils import (
    get_ground_truth_info, 
    align_raster_to_reference, 
    merge_rasters, 
    create_multiband_stack,
    create_dummy_ground_truth
)

__all__ = [
    'get_ground_truth_info',
    'align_raster_to_reference',
    'merge_rasters',
    'create_multiband_stack',
    'create_dummy_ground_truth'
]