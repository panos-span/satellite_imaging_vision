"""
Utility functions for geospatial data processing.
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box, mapping
import shutil


def get_ground_truth_info(ground_truth_path):
    """
    Get information about the ground truth data.
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to ground truth GeoTIFF file
    
    Returns:
    --------
    gt_info : dict
        Dictionary with ground truth information
    """
    with rasterio.open(ground_truth_path) as src:
        gt_info = {
            'crs': src.crs,
            'transform': src.transform,
            'shape': (src.height, src.width),
            'bounds': src.bounds,
            'profile': src.profile.copy()
        }
        
        # Try to get class names from metadata
        if 'classes' in src.tags():
            gt_info['classes'] = src.tags()['classes'].split(',')
        
        # Get unique values in mask
        mask_data = src.read(1)
        gt_info['unique_values'] = np.unique(mask_data).tolist()
    
    return gt_info


def align_raster_to_reference(input_path, reference_info, output_path, resampling=Resampling.bilinear):
    """
    Align a raster to match the CRS, transform, and dimensions of a reference.
    
    Parameters:
    -----------
    input_path : str
        Path to input raster file
    reference_info : dict
        Dictionary with reference information (crs, transform, shape)
    output_path : str
        Path to save aligned raster
    resampling : Resampling, optional
        Resampling method to use
    
    Returns:
    --------
    output_path : str
        Path to aligned raster
    """
    # Extract reference information
    ref_crs = reference_info['crs']
    ref_transform = reference_info['transform']
    ref_shape = reference_info['shape']
    
    with rasterio.open(input_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_shape = (src.height, src.width)
        
        # Check if reprojection is needed
        if src_crs != ref_crs or src_transform != ref_transform or src_shape != ref_shape:
            # Create output profile based on reference
            dst_profile = src.profile.copy()
            dst_profile.update({
                'driver': 'GTiff',
                'height': ref_shape[0],
                'width': ref_shape[1],
                'count': 1,
                'dtype': src.dtypes[0],
                'crs': ref_crs,
                'transform': ref_transform
            })
            
            # Read source data
            src_data = src.read(1)
            
            # Create destination array
            dst_data = np.zeros(ref_shape, dtype=src.dtypes[0])
            
            # Reproject
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling
            )
            
            # Write aligned band
            with rasterio.open(output_path, 'w', **dst_profile) as dst:
                dst.write(dst_data, 1)
        else:
            # No reprojection needed, just copy the file
            shutil.copy2(input_path, output_path)
    
    return output_path


def merge_rasters(raster_paths, output_path, reference_bounds=None, reference_crs=None):
    """
    Merge multiple rasters into a single file and optionally crop to reference bounds.
    
    Parameters:
    -----------
    raster_paths : list
        List of paths to raster files to merge
    output_path : str
        Path to save merged raster
    reference_bounds : tuple, optional
        Bounds to crop to (left, bottom, right, top)
    reference_crs : CRS, optional
        CRS of the reference bounds
    
    Returns:
    --------
    output_path : str
        Path to merged raster
    """
    from rasterio.merge import merge
    
    # Open all files
    src_files = [rasterio.open(path) for path in raster_paths]
    
    try:
        # Merge the files
        mosaic, out_transform = merge(src_files)
        
        # Get metadata from the first file
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform
        })
        
        # Crop to reference bounds if provided
        if reference_bounds is not None and reference_crs is not None:
            # Create shapely geometry for reference bounds
            ref_geom = box(*reference_bounds)
            geom = mapping(ref_geom)
            
            try:
                # Create a memory file for the merged mosaic
                from rasterio.io import MemoryFile
                with MemoryFile() as memfile:
                    with memfile.open(**out_meta) as temp:
                        temp.write(mosaic)
                    
                    # Open the memory file and crop to reference bounds
                    with memfile.open() as temp:
                        cropped, crop_transform = mask(temp, [geom], crop=True)
                        
                        # Update metadata
                        out_meta.update({
                            'height': cropped.shape[1],
                            'width': cropped.shape[2],
                            'transform': crop_transform
                        })
                        
                        # Write cropped mosaic
                        with rasterio.open(output_path, 'w', **out_meta) as dst:
                            dst.write(cropped)
            except Exception as e:
                print(f"Error cropping mosaic: {str(e)}")
                
                # Fallback: save the uncropped mosaic
                with rasterio.open(output_path, 'w', **out_meta) as dst:
                    dst.write(mosaic)
        else:
            # Save the uncropped mosaic
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(mosaic)
    finally:
        # Close all open files
        for src in src_files:
            src.close()
    
    return output_path


def create_multiband_stack(band_paths, output_path, band_names=None):
    """
    Create a single multiband image by stacking multiple bands.
    
    Parameters:
    -----------
    band_paths : dict or list
        Dictionary with band names as keys and file paths as values,
        or list of file paths with band names derived from filenames
    output_path : str
        Path to save the stacked image
    band_names : list, optional
        List of band names (required if band_paths is a list)
    
    Returns:
    --------
    output_path : str
        Path to the stacked image
    """
    # Convert list to dictionary if needed
    if isinstance(band_paths, list):
        if band_names is None:
            # Extract band names from filenames
            band_names = [os.path.basename(path).split('_')[0].split('.')[0] for path in band_paths]
        
        band_paths_dict = dict(zip(band_names, band_paths))
    else:
        band_paths_dict = band_paths
    
    # Sort band names for consistency
    sorted_band_names = sorted(band_paths_dict.keys())
    
    # Get metadata from the first band
    with rasterio.open(band_paths_dict[sorted_band_names[0]]) as src:
        profile = src.profile.copy()
        profile.update({
            'count': len(sorted_band_names),
            'driver': 'GTiff'
        })
    
    # Create the stacked image
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Write each band
        for i, band_name in enumerate(sorted_band_names):
            with rasterio.open(band_paths_dict[band_name]) as src:
                dst.write(src.read(1), i+1)
            
            # Set band name as description
            dst.set_band_description(i+1, band_name)
        
        # Add band names as metadata
        dst.update_tags(band_names=','.join(sorted_band_names))
    
    return output_path, sorted_band_names


def create_dummy_ground_truth(reference_path, output_path, fill_value=0):
    """
    Create a dummy ground truth mask based on a reference image.
    
    Parameters:
    -----------
    reference_path : str
        Path to reference image
    output_path : str
        Path to save dummy ground truth
    fill_value : int, optional
        Value to fill the mask with
    
    Returns:
    --------
    output_path : str
        Path to dummy ground truth
    """
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'dtype': rasterio.uint8
        })
        
        # Create dummy mask filled with the fill value
        dummy_mask = np.full((src.height, src.width), fill_value, dtype=np.uint8)
        
        # Write dummy mask
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dummy_mask, 1)
            
            # Add metadata
            dst.update_tags(classes='background')
    
    return output_path