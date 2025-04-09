import os
import glob
import zipfile
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.transform import resize
import shutil
from shapely.geometry import box, mapping
import geopandas as gpd
import pandas as pd


def apply_pansharpening(low_res_band, high_res_shape):
    """
    Apply simple pansharpening by resizing the low-resolution band to match the high-resolution band shape.
    
    Parameters:
    -----------
    low_res_band : numpy.ndarray
        Low-resolution band data
    high_res_shape : tuple
        Shape of high-resolution band (height, width)
    
    Returns:
    --------
    pansharpened_band : numpy.ndarray
        Pansharpened band data
    """
    # Resize the low-resolution band to match the high-resolution band shape
    pansharpened_band = resize(
        low_res_band, 
        high_res_shape, 
        order=3,  # cubic interpolation
        preserve_range=True
    )
    
    return pansharpened_band.astype(low_res_band.dtype)


def pansharpen_sentinel_bands(safe_dir, output_dir):
    """
    Apply pansharpening to all Sentinel-2 bands to get them at 10m resolution.
    
    Parameters:
    -----------
    safe_dir : str
        Path to the Sentinel-2 SAFE directory
    output_dir : str
        Directory to save pansharpened bands
    
    Returns:
    --------
    pansharpened_bands_paths : dict
        Dictionary with band names as keys and file paths as values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get IMG_DATA directory
    img_data_path = list(Path(safe_dir).glob('GRANULE/*/IMG_DATA'))[0]
    
    # Define band resolution groups
    high_res_bands = ['B02', 'B03', 'B04', 'B08']  # 10m
    medium_res_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # 20m
    low_res_bands = ['B01', 'B09', 'B10']  # 60m
    
    # Dictionary to store paths to pansharpened bands
    pansharpened_bands_paths = {}
    
    # Find all band files (excluding TCI)
    band_files = [f for f in img_data_path.glob('*.jp2') if not f.name.endswith('TCI.jp2')]
    
    # Get a reference high-resolution band to use as template
    high_res_band_file = None
    for band_file in band_files:
        band_name = band_file.name.split('_')[-1].split('.')[0]
        if band_name in high_res_bands:
            high_res_band_file = band_file
            break
    
    if high_res_band_file is None:
        raise FileNotFoundError(f"No high-resolution band found in {img_data_path}")
    
    # Read high-resolution band metadata as template
    with rasterio.open(high_res_band_file) as src:
        high_res_profile = src.profile.copy()
        high_res_shape = (src.height, src.width)
        high_res_transform = src.transform
        high_res_crs = src.crs
    
    print(f"Using {os.path.basename(high_res_band_file)} as reference with shape {high_res_shape}")
    
    # Process all bands
    for band_file in band_files:
        band_name = band_file.name.split('_')[-1].split('.')[0]
        output_path = os.path.join(output_dir, f"{band_name}_pansharpened.tif")
        
        with rasterio.open(band_file) as src:
            # Determine if band needs pansharpening
            needs_pansharpening = (band_name in medium_res_bands or band_name in low_res_bands)
            
            if needs_pansharpening:
                print(f"Pansharpening {band_name} from shape {(src.height, src.width)} to {high_res_shape}")
                
                # Read band data
                band_data = src.read(1)
                
                # Apply pansharpening
                pansharpened_data = apply_pansharpening(band_data, high_res_shape)
            else:
                print(f"Copying {band_name} (already high resolution)")
                pansharpened_data = src.read(1)
            
            # Create profile for output file
            output_profile = high_res_profile.copy()
            output_profile.update({
                'driver': 'GTiff',
                'height': high_res_shape[0],
                'width': high_res_shape[1],
                'count': 1,
                'dtype': pansharpened_data.dtype,
                'crs': high_res_crs,
                'transform': high_res_transform
            })
            
            # Write pansharpened band
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(pansharpened_data, 1)
            
            pansharpened_bands_paths[band_name] = output_path
    
    print(f"Pansharpening completed for {len(pansharpened_bands_paths)} bands")
    return pansharpened_bands_paths


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


def align_with_ground_truth(pansharpened_bands_paths, ground_truth_info, output_dir):
    """
    Align pansharpened bands with ground truth data.
    
    Parameters:
    -----------
    pansharpened_bands_paths : dict
        Dictionary with band names as keys and file paths as values
    ground_truth_info : dict
        Dictionary with ground truth information
    output_dir : str
        Directory to save aligned bands
    
    Returns:
    --------
    aligned_bands_paths : dict
        Dictionary with band names as keys and file paths as values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ground truth information
    gt_crs = ground_truth_info['crs']
    gt_transform = ground_truth_info['transform']
    gt_shape = ground_truth_info['shape']
    
    print(f"Ground truth CRS: {gt_crs}")
    print(f"Ground truth shape: {gt_shape}")
    
    # Dictionary to store aligned band paths
    aligned_bands_paths = {}
    
    # Process each pansharpened band
    for band_name, band_path in pansharpened_bands_paths.items():
        output_path = os.path.join(output_dir, f"{band_name}_aligned.tif")
        
        with rasterio.open(band_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_shape = (src.height, src.width)
            
            print(f"Aligning {band_name} from CRS {src_crs} to {gt_crs}, shape {src_shape} to {gt_shape}")
            
            # Check if reprojection is needed
            if src_crs != gt_crs or src_transform != gt_transform or src_shape != gt_shape:
                # Create output profile based on ground truth
                dst_profile = src.profile.copy()
                dst_profile.update({
                    'driver': 'GTiff',
                    'height': gt_shape[0],
                    'width': gt_shape[1],
                    'count': 1,
                    'dtype': src.dtypes[0],
                    'crs': gt_crs,
                    'transform': gt_transform
                })
                
                # Read source data
                src_data = src.read(1)
                
                # Create destination array
                dst_data = np.zeros(gt_shape, dtype=src.dtypes[0])
                
                # Reproject
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=gt_transform,
                    dst_crs=gt_crs,
                    resampling=Resampling.bilinear
                )
                
                # Write aligned band
                with rasterio.open(output_path, 'w', **dst_profile) as dst:
                    dst.write(dst_data, 1)
            else:
                # No reprojection needed, just copy the file
                shutil.copy2(band_path, output_path)
            
            aligned_bands_paths[band_name] = output_path
    
    print(f"Alignment completed for {len(aligned_bands_paths)} bands")
    return aligned_bands_paths


def merge_aligned_bands(aligned_bands_list, ground_truth_info, output_dir):
    """
    Merge aligned bands from multiple tiles into a single mosaic.
    
    Parameters:
    -----------
    aligned_bands_list : list of dict
        List of dictionaries with band names as keys and file paths as values
    ground_truth_info : dict
        Dictionary with ground truth information
    output_dir : str
        Directory to save merged bands
    
    Returns:
    --------
    merged_bands_paths : dict
        Dictionary with band names as keys and file paths as values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ground truth bounds
    gt_bounds = ground_truth_info['bounds']
    gt_crs = ground_truth_info['crs']
    
    # Create shapely geometry for ground truth bounds
    gt_geom = box(*gt_bounds)
    
    # Dictionary to store merged band paths
    merged_bands_paths = {}
    
    # Get all unique band names
    all_band_names = set()
    for aligned_bands in aligned_bands_list:
        all_band_names.update(aligned_bands.keys())
    
    # Sort band names for consistency
    all_band_names = sorted(all_band_names)
    
    # Process each band
    for band_name in all_band_names:
        print(f"Merging {band_name} from {len(aligned_bands_list)} tiles...")
        
        # Collect all files for this band
        band_files = []
        for aligned_bands in aligned_bands_list:
            if band_name in aligned_bands:
                band_files.append(aligned_bands[band_name])
        
        if not band_files:
            print(f"No files found for band {band_name}, skipping")
            continue
        
        # Open all files
        src_files = [rasterio.open(file) for file in band_files]
        
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
            
            # Crop to ground truth bounds
            geom = mapping(gt_geom)
            
            try:
                # Create a memory file for the merged mosaic
                from rasterio.io import MemoryFile
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
                        output_path = os.path.join(output_dir, f"{band_name}_merged.tif")
                        with rasterio.open(output_path, 'w', **out_meta) as dst:
                            dst.write(cropped)
                        
                        merged_bands_paths[band_name] = output_path
            except Exception as e:
                print(f"Error cropping mosaic: {str(e)}")
                
                # Fallback: save the uncropped mosaic
                output_path = os.path.join(output_dir, f"{band_name}_merged.tif")
                with rasterio.open(output_path, 'w', **out_meta) as dst:
                    dst.write(mosaic)
                
                merged_bands_paths[band_name] = output_path
        finally:
            # Close all open files
            for src in src_files:
                src.close()
    
    print(f"Merging completed for {len(merged_bands_paths)} bands")
    return merged_bands_paths


def create_multiband_stack(merged_bands_paths, output_path):
    """
    Create a single multiband image by stacking all merged bands.
    
    Parameters:
    -----------
    merged_bands_paths : dict
        Dictionary with band names as keys and file paths as values
    output_path : str
        Path to save the stacked image
    
    Returns:
    --------
    output_path : str
        Path to the stacked image
    """
    # Sort band names alphabetically to ensure consistent order
    sorted_band_names = sorted(merged_bands_paths.keys())
    
    # Get metadata from the first band
    with rasterio.open(merged_bands_paths[sorted_band_names[0]]) as src:
        profile = src.profile.copy()
        profile.update({
            'count': len(sorted_band_names),
            'driver': 'GTiff'
        })
    
    # Create the stacked image
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Write each band
        for i, band_name in enumerate(sorted_band_names):
            with rasterio.open(merged_bands_paths[band_name]) as src:
                dst.write(src.read(1), i+1)
            
            # Set band name as description
            dst.set_band_description(i+1, band_name)
        
        # Add band names as metadata
        dst.update_tags(band_names=','.join(sorted_band_names))
    
    print(f"Created multiband stack with {len(sorted_band_names)} bands: {os.path.basename(output_path)}")
    print(f"Band order: {sorted_band_names}")
    
    return output_path


def visualize_rgb_large_image(multispectral_path, output_path=None, band_indices=None, max_size=4000):
    """
    Create RGB visualization from multiband image, with efficient handling for large images.
    
    Parameters:
    -----------
    multispectral_path : str
        Path to multiband image
    output_path : str, optional
        Path to save visualization
    band_indices : tuple, optional
        Indices of bands to use for RGB (0-based indexing)
        If None, tries to use B04, B03, B02 bands
    max_size : int, optional
        Maximum dimension (width or height) for visualization
    
    Returns:
    --------
    None
    """
    with rasterio.open(multispectral_path) as src:
        # Get image dimensions
        height, width = src.height, src.width
        
        # Calculate downsampling factor if image is too large
        if width > max_size or height > max_size:
            scale_factor = min(max_size / width, max_size / height)
            out_shape = (int(height * scale_factor), int(width * scale_factor))
            print(f"Downsampling image from {width}x{height} to {out_shape[1]}x{out_shape[0]} for visualization")
        else:
            scale_factor = 1
            out_shape = (height, width)
        
        # Get band names
        band_names = src.tags().get('band_names', '').split(',')
        
        # Try to find the right bands for RGB
        if band_indices is None:
            # Try to locate B04, B03, B02 for natural color
            rgb_indices = []
            for band_name in ['B04', 'B03', 'B02']:
                if band_name in band_names:
                    rgb_indices.append(band_names.index(band_name))
            
            # If not found, use first three bands
            if len(rgb_indices) != 3:
                rgb_indices = [0, 1, 2] if src.count >= 3 else [0, 0, 0]
            
            band_indices = tuple(rgb_indices)
        
        # Read and resample RGB bands
        rgb_resampled = np.zeros((3, *out_shape), dtype=np.float32)
        
        for i, band_idx in enumerate(band_indices):
            # Read band with resampling (add 1 because rasterio uses 1-based indexing)
            resampled = src.read(
                band_idx + 1,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            rgb_resampled[i] = resampled
        
        # Convert to HxWxC format for matplotlib
        rgb = np.transpose(rgb_resampled, (1, 2, 0))
        
        # Scale each band to 0-1 range individually using percentiles
        rgb_scaled = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
            if p98 > p2:
                rgb_scaled[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)
        
        # Create visualization
        plt.figure(figsize=(12, 12 * out_shape[0] / out_shape[1]))  # Maintain aspect ratio
        plt.imshow(rgb_scaled)
        plt.axis('off')
        
        # Set title with band names
        if len(band_names) >= 3:
            title_bands = [band_names[i] for i in band_indices]
            plt.title(f"RGB Visualization using {title_bands[0]}, {title_bands[1]}, {title_bands[2]}")
        else:
            plt.title("RGB Visualization")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"RGB visualization saved to {output_path}")
            plt.close()
        else:
            plt.show()


def create_visualization_thumbnails(multispectral_path, ground_truth_path, output_dir):
    """
    Create visualization thumbnails of the dataset.
    
    Parameters:
    -----------
    multispectral_path : str
        Path to multiband image
    ground_truth_path : str
        Path to ground truth image
    output_dir : str
        Directory to save visualizations
    
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Maximum thumbnail dimension
    max_size = 2000
    
    # Read metadata from both files
    with rasterio.open(multispectral_path) as img_src, rasterio.open(ground_truth_path) as mask_src:
        # Get image dimensions
        height, width = img_src.height, img_src.width
        
        # Calculate downsampling factor if image is too large
        if width > max_size or height > max_size:
            scale_factor = min(max_size / width, max_size / height)
            out_shape = (int(height * scale_factor), int(width * scale_factor))
            print(f"Downsampling images from {width}x{height} to {out_shape[1]}x{out_shape[0]} for visualization")
        else:
            scale_factor = 1
            out_shape = (height, width)
        
        # Get band names
        band_names = img_src.tags().get('band_names', '').split(',')
        
        # Try to find indices for RGB visualization
        rgb_indices = []
        for band_name in ['B04', 'B03', 'B02']:
            if band_name in band_names:
                rgb_indices.append(band_names.index(band_name))
        
        # Default to first three bands if not found
        if len(rgb_indices) != 3:
            rgb_indices = [0, 1, 2] if img_src.count >= 3 else [0, 0, 0]
        
        # Read and resample RGB bands
        rgb_resampled = np.zeros((3, *out_shape), dtype=np.float32)
        
        for i, band_idx in enumerate(rgb_indices):
            # Read band with resampling (add 1 because rasterio uses 1-based indexing)
            resampled = img_src.read(
                band_idx + 1,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            rgb_resampled[i] = resampled
        
        # Convert to HxWxC format for matplotlib
        rgb = np.transpose(rgb_resampled, (1, 2, 0))
        
        # Scale each band to 0-1 range individually using percentiles
        rgb_scaled = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
            if p98 > p2:
                rgb_scaled[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)
        
        # Read and resample ground truth
        mask_resampled = mask_src.read(
            1,
            out_shape=out_shape,
            resampling=Resampling.nearest  # Use nearest for categorical data
        )
        
        # Try to get class names
        class_names = None
        if 'classes' in mask_src.tags():
            class_names = mask_src.tags()['classes'].split(',')
        
        # Get unique values in mask
        unique_values = np.unique(mask_resampled)
    
    # Create figure with RGB image and ground truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7 * out_shape[0] / out_shape[1]))
    
    # Plot RGB image
    ax1.imshow(rgb_scaled)
    ax1.set_title("Sentinel-2 RGB")
    ax1.axis('off')
    
    # Plot ground truth
    im = ax2.imshow(mask_resampled, cmap='viridis')
    ax2.set_title("Ground Truth")
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add class names to colorbar if available
    if class_names and len(class_names) == len(unique_values):
        cbar.set_ticks(unique_values)
        cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_preview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset visualization saved to {os.path.join(output_dir, 'dataset_preview.png')}")


def create_training_dataset(multispectral_path, ground_truth_path, output_dir):
    """
    Create training dataset by copying multiband image and ground truth to output directory.
    
    Parameters:
    -----------
    multispectral_path : str
        Path to multiband image
    ground_truth_path : str
        Path to ground truth image
    output_dir : str
        Directory to save dataset
    
    Returns:
    --------
    dataset_info : dict
        Dictionary with dataset information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy files to dataset directory
    image_dst_path = os.path.join(output_dir, 'sentinel_image.tif')
    mask_dst_path = os.path.join(output_dir, 'ground_truth.tif')
    
    shutil.copy2(multispectral_path, image_dst_path)
    shutil.copy2(ground_truth_path, mask_dst_path)
    
    # Read metadata to create dataset info
    with rasterio.open(image_dst_path) as img_src, rasterio.open(mask_dst_path) as mask_src:
        # Get band names
        band_names = img_src.tags().get('band_names', '').split(',')
        
        # Try to get class names from mask metadata
        class_names = None
        if 'classes' in mask_src.tags():
            class_names = mask_src.tags()['classes'].split(',')
        
        # Get unique values in mask
        mask_data = mask_src.read(1)
        unique_values = np.unique(mask_data)
        
        # Create dataset info
        dataset_info = {
            'image_path': image_dst_path,
            'mask_path': mask_dst_path,
            'image_shape': (img_src.height, img_src.width, img_src.count),
            'mask_shape': (mask_src.height, mask_src.width),
            'crs': img_src.crs.to_string(),
            'band_names': band_names,
            'class_values': unique_values.tolist(),
            'class_names': class_names
        }
    
    # Create visualizations
    create_visualization_thumbnails(multispectral_path, ground_truth_path, output_dir)
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"  Image shape: {dataset_info['image_shape']}")
    print(f"  Mask shape: {dataset_info['mask_shape']}")
    print(f"  Number of bands: {len(band_names)}")
    print(f"  Band names: {band_names}")
    print(f"  Unique class values: {unique_values}")
    
    if class_names:
        print(f"  Class names: {class_names}")
    
    return dataset_info


def process_multiple_sentinel_tiles(safe_dirs, ground_truth_path, output_base_dir):
    """
    Process multiple Sentinel-2 SAFE directories and merge them into a single dataset.
    
    Parameters:
    -----------
    safe_dirs : list
        List of paths to Sentinel-2 SAFE directories
    ground_truth_path : str
        Path to ground truth GeoTIFF file
    output_base_dir : str
        Base directory for output
    
    Returns:
    --------
    dataset_info : dict
        Dictionary with dataset information
    """
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    merged_dir = os.path.join(output_base_dir, "merged")
    dataset_dir = os.path.join(output_base_dir, "dataset")
    
    # Get ground truth information
    print("\nReading ground truth information...")
    ground_truth_info = get_ground_truth_info(ground_truth_path)
    
    # For each tile: apply pansharpening and align with ground truth
    aligned_bands_list = []
    
    for i, safe_dir in enumerate(safe_dirs):
        # Extract tile ID from SAFE directory name
        tile_id = os.path.basename(safe_dir).split('_')[5]
        
        print(f"\nProcessing tile {i+1}/{len(safe_dirs)}: {tile_id}")
        
        # Create tile-specific output directories
        pansharpened_dir = os.path.join(output_base_dir, f"pansharpened_{tile_id}")
        aligned_dir = os.path.join(output_base_dir, f"aligned_{tile_id}")
        
        # Step 1: Apply pansharpening
        print(f"Step 1: Applying pansharpening to {os.path.basename(safe_dir)}...")
        pansharpened_bands_paths = pansharpen_sentinel_bands(safe_dir, pansharpened_dir)
        
        # Step 2: Align with ground truth
        print(f"Step 2: Aligning pansharpened bands with ground truth...")
        aligned_bands_paths = align_with_ground_truth(pansharpened_bands_paths, ground_truth_info, aligned_dir)
        
        # Add to list for merging
        aligned_bands_list.append(aligned_bands_paths)
    
    # Step 3: Merge aligned bands from all tiles
    print("\nStep 3: Merging aligned bands from all tiles...")
    merged_bands_paths = merge_aligned_bands(aligned_bands_list, ground_truth_info, merged_dir)
    
    # Step 4: Create multiband stack
    print("\nStep 4: Creating multiband stack...")
    multiband_path = os.path.join(output_base_dir, "combined_multiband.tif")
    multiband_path = create_multiband_stack(merged_bands_paths, multiband_path)
    
    # Step 5: Create RGB visualization
    print("\nStep 5: Creating RGB visualization...")
    vis_path = os.path.join(output_base_dir, "combined_rgb.png")
    visualize_rgb_large_image(multiband_path, vis_path)
    
    # Step 6: Create training dataset
    print("\nStep 6: Creating training dataset...")
    dataset_info = create_training_dataset(multiband_path, ground_truth_path, dataset_dir)
    
    return dataset_info


if __name__ == "__main__":
    # Find all SAFE directories
    sentinel_data_dir = "sentinel_data"
    safe_dirs = glob.glob(os.path.join(sentinel_data_dir, "*.SAFE"))
    
    if not safe_dirs:
        print("No SAFE directories found in", sentinel_data_dir)
        # Check if there are zip files that need to be extracted
        zip_files = glob.glob(os.path.join(sentinel_data_dir, "*.zip"))
        if zip_files:
            print(f"Found {len(zip_files)} zip files. Extracting...")
            for zip_file in zip_files:
                safe_name = os.path.basename(zip_file).replace(".zip", "")
                safe_dir = os.path.join(sentinel_data_dir, safe_name)
                
                # Extract if not already extracted
                if not os.path.exists(safe_dir):
                    print(f"Extracting {os.path.basename(zip_file)}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(sentinel_data_dir)
                
                safe_dirs.append(safe_dir)
    
    print(f"Found {len(safe_dirs)} SAFE directories:")
    for safe_dir in safe_dirs:
        print(f"  {os.path.basename(safe_dir)}")
    
    # Get ground truth path
    ground_truth_path = "ground_truth.tif"
    
    # Set output directory
    output_base_dir = "F:\\processed_data"
    
    # Process all tiles
    dataset_info = process_multiple_sentinel_tiles(safe_dirs, ground_truth_path, output_base_dir)