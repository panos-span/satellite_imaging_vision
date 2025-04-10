"""
Functions for creating and working with validity masks.
"""
import os
import numpy as np
import rasterio
from rasterio.warp import Resampling
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import box, mapping


class ValidityMaskCreator:
    """
    Class for creating and managing validity masks.
    """
    
    def __init__(self, safe_dir=None, aligned_bands_paths=None):
        """
        Initialize the validity mask creator.
        
        Parameters:
        -----------
        safe_dir : str, optional
            Path to the Sentinel-2 SAFE directory
        aligned_bands_paths : dict, optional
            Dictionary with band names as keys and file paths to aligned bands as values
        """
        self.safe_dir = safe_dir
        self.aligned_bands_paths = aligned_bands_paths
        self.cloud_mask_path = None
        
        # Find cloud mask if available (in QI_DATA folder)
        if safe_dir:
            self._find_cloud_mask()
    
    def _find_cloud_mask(self):
        """Find cloud classification mask in GRANULE/*/QI_DATA."""
        try:
            qi_data_path = list(Path(self.safe_dir).glob('GRANULE/*/QI_DATA'))[0]
            cloud_mask_files = list(qi_data_path.glob('MSK_CLASSI_B00.jp2'))
            
            if cloud_mask_files:
                self.cloud_mask_path = str(cloud_mask_files[0])
                print(f"Found cloud classification mask: {os.path.basename(self.cloud_mask_path)}")
        except (IndexError, FileNotFoundError):
            print("No cloud classification mask found")
    
    def create_validity_mask(self, output_path):
        """
        Create a validity mask identifying valid pixels.
        
        Parameters:
        -----------
        output_path : str
            Path to save the validity mask
        
        Returns:
        --------
        output_path : str
            Path to the validity mask
        """
        if not self.aligned_bands_paths:
            raise ValueError("No aligned bands provided")
        
        # Get the first aligned band to use as reference for dimensions and transform
        first_band_name = next(iter(self.aligned_bands_paths))
        first_band_path = self.aligned_bands_paths[first_band_name]
        
        with rasterio.open(first_band_path) as src:
            profile = src.profile.copy()
            shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs
            
            # Initialize validity mask (1 = valid, 0 = invalid)
            validity_mask = np.ones(shape, dtype=np.uint8)
            
            # Mark no-data values as invalid
            if src.nodata is not None:
                band_data = src.read(1)
                validity_mask[band_data == src.nodata] = 0
        
        # Check for cloud coverage if cloud mask is available
        if self.cloud_mask_path:
            try:
                with rasterio.open(self.cloud_mask_path) as src:
                    # Read cloud mask and resample to match the aligned bands
                    cloud_data = src.read(
                        1, 
                        out_shape=shape,
                        resampling=Resampling.nearest
                    )
                    
                    # Interpret the cloud mask bits
                    # For Sentinel-2 L1C, the classification mask has the following bit values:
                    # Bit 10: Opaque clouds (0 = no opaque clouds, 1 = opaque clouds)
                    # Bit 11: Cirrus clouds (0 = no cirrus clouds, 1 = cirrus clouds)
                    
                    # Create cloud mask (1 = cloud, 0 = no cloud)
                    cloud_bit_10 = ((cloud_data >> 10) & 1)  # Opaque clouds
                    cloud_bit_11 = ((cloud_data >> 11) & 1)  # Cirrus clouds
                    
                    # Mark cloud pixels as invalid in the validity mask
                    validity_mask[(cloud_bit_10 == 1) | (cloud_bit_11 == 1)] = 0
                    
                    print("Applied cloud mask to validity mask")
            except Exception as e:
                print(f"Error applying cloud mask: {str(e)}")
        
        # Also check each band for extreme values or other quality issues
        for band_name, band_path in self.aligned_bands_paths.items():
            try:
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
                    
                    # Mark invalid pixels (could be no-data or extreme values)
                    if src.nodata is not None:
                        validity_mask[band_data == src.nodata] = 0
                    
                    # Optional: Mark pixels with extreme values (statistical outliers)
                    # For example, values beyond 3 standard deviations from the mean
                    # valid_pixels = band_data[validity_mask == 1]
                    # if len(valid_pixels) > 0:
                    #     mean = np.mean(valid_pixels)
                    #     std = np.std(valid_pixels)
                    #     threshold = mean + 3 * std
                    #     validity_mask[(band_data > threshold) & (validity_mask == 1)] = 0
            except Exception as e:
                print(f"Error processing band {band_name} for validity mask: {str(e)}")
        
        # Save validity mask
        # Update profile for mask
        profile.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': None,
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(validity_mask, 1)
            dst.update_tags(description="Validity mask (1=valid, 0=invalid)")
        
        print(f"Validity mask created: {output_path}")
        print(f"Valid pixels: {np.count_nonzero(validity_mask)} / {validity_mask.size} ({np.count_nonzero(validity_mask) / validity_mask.size * 100:.2f}%)")
        
        return output_path


def merge_validity_masks(validity_mask_paths, ground_truth_info, output_path):
    """
    Merge multiple validity masks into a single mask for the entire dataset.
    
    Parameters:
    -----------
    validity_mask_paths : list
        List of paths to validity masks
    ground_truth_info : dict
        Dictionary with ground truth information
    output_path : str
        Path to save the merged validity mask
    
    Returns:
    --------
    output_path : str
        Path to the merged validity mask
    """
    from rasterio.merge import merge
    from rasterio.mask import mask
    from rasterio.io import MemoryFile
    
    # Extract ground truth bounds
    gt_bounds = ground_truth_info['bounds']
    gt_crs = ground_truth_info['crs']
    
    # Create shapely geometry for ground truth bounds
    gt_geom = box(*gt_bounds)
    
    # Open all mask files
    mask_files = [rasterio.open(path) for path in validity_mask_paths]
    
    try:
        # Merge the masks
        mosaic, out_transform = merge(mask_files)
        
        # Get metadata from the first file
        out_meta = mask_files[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform
        })
        
        # Crop to ground truth bounds
        geom = mapping(gt_geom)
        
        try:
            # Create a memory file for the merged mosaic
            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as temp:
                    temp.write(mosaic)
                
                # Open the memory file and crop to ground truth bounds
                with memfile.open() as temp:
                    cropped, crop_transform = mask(temp, [geom], crop=True)
                    
                    # Update metadata
                    out_meta.update({
                        'height': cropped.shape[1],
                        'width': cropped.shape[2],
                        'transform': crop_transform
                    })
                    
                    # Write cropped mosaic
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with rasterio.open(output_path, 'w', **out_meta) as dst:
                        # For validity masks, we want the most restrictive approach - 
                        # a pixel is valid only if it's valid in all tiles
                        final_mask = (cropped[0] > 0).astype(np.uint8)
                        dst.write(final_mask, 1)
                        dst.update_tags(description="Merged validity mask (1=valid, 0=invalid)")
        except Exception as e:
            print(f"Error cropping merged validity mask: {str(e)}")
            
            # Fallback: save the uncropped mosaic
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                # For validity masks, we want the most restrictive approach
                final_mask = (mosaic[0] > 0).astype(np.uint8)
                dst.write(final_mask, 1)
                dst.update_tags(description="Merged validity mask (1=valid, 0=invalid)")
    finally:
        # Close all open files
        for src in mask_files:
            src.close()
    
    # Report statistics
    with rasterio.open(output_path) as src:
        mask_data = src.read(1)
        valid_pixels = np.count_nonzero(mask_data)
        total_pixels = mask_data.size
        valid_percentage = (valid_pixels / total_pixels) * 100
        
        print(f"Merged validity mask created: {output_path}")
        print(f"Valid pixels: {valid_pixels} / {total_pixels} ({valid_percentage:.2f}%)")
    
    return output_path