"""
Visualization utilities for satellite imagery and geospatial data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import Resampling


def visualize_rgb(multispectral_path, output_path=None, band_indices=None, max_size=4000):
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


def visualize_false_color(multispectral_path, output_path=None, combination='vegetation', max_size=4000):
    """
    Create false color visualization from multiband image.
    
    Parameters:
    -----------
    multispectral_path : str
        Path to multiband image
    output_path : str, optional
        Path to save visualization
    combination : str, optional
        Type of false color combination:
        - 'vegetation': Uses bands 8 (NIR), 4 (Red), 3 (Green) to highlight vegetation
        - 'urban': Uses bands 12 (SWIR), 11 (SWIR), 4 (Red) to highlight urban areas
        - 'geology': Uses bands 12 (SWIR), 11 (SWIR), 2 (Blue) for geological features
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
        
        # Define band combinations
        combinations = {
            'vegetation': ['B08', 'B04', 'B03'],  # NIR, Red, Green
            'urban': ['B12', 'B11', 'B04'],       # SWIR2, SWIR1, Red
            'geology': ['B12', 'B11', 'B02']      # SWIR2, SWIR1, Blue
        }
        
        if combination not in combinations:
            print(f"Unknown combination: {combination}, using 'vegetation' instead")
            combination = 'vegetation'
        
        # Get band indices
        band_indices = []
        for band_name in combinations[combination]:
            if band_name in band_names:
                band_indices.append(band_names.index(band_name))
            else:
                print(f"Band {band_name} not found in the image")
                return
        
        # Read and resample bands
        bands_resampled = np.zeros((3, *out_shape), dtype=np.float32)
        
        for i, band_idx in enumerate(band_indices):
            # Read band with resampling (add 1 because rasterio uses 1-based indexing)
            resampled = src.read(
                band_idx + 1,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            bands_resampled[i] = resampled
        
        # Convert to HxWxC format for matplotlib
        false_color = np.transpose(bands_resampled, (1, 2, 0))
        
        # Scale each band to 0-1 range individually using percentiles
        fc_scaled = np.zeros_like(false_color, dtype=np.float32)
        for i in range(3):
            p2, p98 = np.percentile(false_color[:, :, i], (2, 98))
            if p98 > p2:
                fc_scaled[:, :, i] = np.clip((false_color[:, :, i] - p2) / (p98 - p2), 0, 1)
        
        # Create visualization
        plt.figure(figsize=(12, 12 * out_shape[0] / out_shape[1]))  # Maintain aspect ratio
        plt.imshow(fc_scaled)
        plt.axis('off')
        
        # Set title
        title_bands = [band_names[i] for i in band_indices]
        plt.title(f"False Color ({combination.capitalize()}) using {title_bands[0]}, {title_bands[1]}, {title_bands[2]}")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"False color visualization saved to {output_path}")
            plt.close()
        else:
            plt.show()


def visualize_dataset(multispectral_path, ground_truth_path, output_dir, max_size=2000):
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
    max_size : int, optional
        Maximum dimension (width or height) for visualization
    
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
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


def visualize_validity_mask(mask_path, output_path=None, max_size=4000):
    """
    Visualize a validity mask.
    
    Parameters:
    -----------
    mask_path : str
        Path to validity mask
    output_path : str, optional
        Path to save visualization
    max_size : int, optional
        Maximum dimension (width or height) for visualization
    
    Returns:
    --------
    None
    """
    with rasterio.open(mask_path) as src:
        # Get image dimensions
        height, width = src.height, src.width
        
        # Calculate downsampling factor if image is too large
        if width > max_size or height > max_size:
            scale_factor = min(max_size / width, max_size / height)
            out_shape = (int(height * scale_factor), int(width * scale_factor))
            print(f"Downsampling mask from {width}x{height} to {out_shape[1]}x{out_shape[0]} for visualization")
        else:
            scale_factor = 1
            out_shape = (height, width)
        
        # Read and resample mask
        mask_resampled = src.read(
            1,
            out_shape=out_shape,
            resampling=Resampling.nearest  # Use nearest for binary data
        )
    
    # Create visualization
    plt.figure(figsize=(12, 12 * out_shape[0] / out_shape[1]))  # Maintain aspect ratio
    plt.imshow(mask_resampled, cmap='gray')
    plt.title("Validity Mask (white=valid, black=invalid)")
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Validity mask visualization saved to {output_path}")
        plt.close()
    else:
        plt.show()


def visualize_band(band_path, band_name=None, output_path=None, max_size=4000, cmap='viridis'):
    """
    Visualize a single band from a GeoTIFF.
    
    Parameters:
    -----------
    band_path : str
        Path to the band file
    band_name : str, optional
        Name of the band for the title
    output_path : str, optional
        Path to save visualization
    max_size : int, optional
        Maximum dimension (width or height) for visualization
    cmap : str, optional
        Colormap to use
    
    Returns:
    --------
    None
    """
    with rasterio.open(band_path) as src:
        # Get image dimensions
        height, width = src.height, src.width
        
        # Calculate downsampling factor if image is too large
        if width > max_size or height > max_size:
            scale_factor = min(max_size / width, max_size / height)
            out_shape = (int(height * scale_factor), int(width * scale_factor))
            print(f"Downsampling band from {width}x{height} to {out_shape[1]}x{out_shape[0]} for visualization")
        else:
            scale_factor = 1
            out_shape = (height, width)
        
        # Read and resample band
        band_resampled = src.read(
            1,
            out_shape=out_shape,
            resampling=Resampling.bilinear
        )
    
    # Create visualization
    plt.figure(figsize=(12, 12 * out_shape[0] / out_shape[1]))  # Maintain aspect ratio
    
    # Scale to 0-1 range using percentiles to avoid outliers
    p2, p98 = np.percentile(band_resampled, (2, 98))
    if p98 > p2:
        band_scaled = np.clip((band_resampled - p2) / (p98 - p2), 0, 1)
    else:
        band_scaled = band_resampled
    
    plt.imshow(band_scaled, cmap=cmap)
    if band_name:
        plt.title(f"Band {band_name}")
    else:
        plt.title("Band Visualization")
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Band visualization saved to {output_path}")
        plt.close()
    else:
        plt.show()