import os
import glob
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
import pandas as pd  # Add pandas import for concat
from shapely.geometry import box
from matplotlib.patches import Patch
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

from cloud_coverage import check_cloud_coverage_improved

def extract_sentinel_zip(zip_path, extract_dir=None):
    """
    Extract a Sentinel-2 zip file.
    
    Parameters:
    -----------
    zip_path : str
        Path to the Sentinel-2 zip file
    extract_dir : str, optional
        Directory to extract to. If None, extracts to the same directory as the zip file.
    
    Returns:
    --------
    safe_dir : str
        Path to the extracted SAFE directory
    """
    if extract_dir is None:
        extract_dir = os.path.dirname(zip_path)
    
    # Get the base name of the zip file without extension
    safe_name = os.path.basename(zip_path).replace('.zip', '')
    safe_dir = os.path.join(extract_dir, safe_name)
    
    # Check if already extracted
    if os.path.exists(safe_dir):
        print(f"Directory {safe_dir} already exists, skipping extraction")
        return safe_dir
    
    # Extract the zip file
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return safe_dir


def get_sentinel_bounds(safe_dir):
    """
    Get the geographic bounds of a Sentinel-2 tile.
    
    Parameters:
    -----------
    safe_dir : str
        Path to the Sentinel-2 SAFE directory
    
    Returns:
    --------
    bounds : tuple
        (left, bottom, right, top) bounds in the CRS of the tile
    crs : rasterio.crs.CRS
        Coordinate Reference System of the tile
    """
    # Find any band file to get the bounds
    img_data_paths = list(Path(safe_dir).glob('GRANULE/*/IMG_DATA'))
    if not img_data_paths:
        raise FileNotFoundError(f"No IMG_DATA directory found in {safe_dir}")
    
    img_data_path = img_data_paths[0]
    
    # Try to find a 10m band (B02, B03, B04, B08)
    band_patterns = ['*B02.jp2', '*B03.jp2', '*B04.jp2', '*B08.jp2']
    
    band_file = None
    for pattern in band_patterns:
        band_files = list(img_data_path.glob(pattern))
        if band_files:
            band_file = band_files[0]
            break
    
    if not band_file:
        # If no 10m band found, try any JP2 file
        band_files = list(img_data_path.glob('*.jp2'))
        if band_files:
            band_file = band_files[0]
        else:
            raise FileNotFoundError(f"No JP2 files found in {img_data_path}")
    
    with rasterio.open(band_file) as src:
        bounds = src.bounds
        crs = src.crs
    
    return bounds, crs


def get_ground_truth_bounds(ground_truth_path):
    """
    Get the geographic bounds of the ground truth data.
    
    Parameters:
    -----------
    ground_truth_path : str
        Path to the ground truth GeoTIFF file
    
    Returns:
    --------
    bounds : tuple
        (left, bottom, right, top) bounds in the CRS of the ground truth
    crs : rasterio.crs.CRS
        Coordinate Reference System of the ground truth
    """
    with rasterio.open(ground_truth_path) as src:
        bounds = src.bounds
        crs = src.crs
    
    return bounds, crs


def check_coverage(safe_dirs, ground_truth_path):
    """
    Check if the provided Sentinel-2 tiles cover the ground truth area completely.
    
    Parameters:
    -----------
    safe_dirs : list
        List of paths to Sentinel-2 SAFE directories
    ground_truth_path : str
        Path to the ground truth GeoTIFF file
    
    Returns:
    --------
    covered : bool
        True if the ground truth area is completely covered, False otherwise
    missing_area : GeoDataFrame or None
        GeoDataFrame with the area of the ground truth not covered by the tiles,
        or None if completely covered
    """
    # Get ground truth bounds
    gt_bounds, gt_crs = get_ground_truth_bounds(ground_truth_path)
    gt_geom = box(*gt_bounds)
    gt_gdf = gpd.GeoDataFrame(geometry=[gt_geom], crs=gt_crs)
    
    # Get all tile geometries
    tile_geometries = []
    
    for safe_dir in safe_dirs:
        try:
            tile_bounds, tile_crs = get_sentinel_bounds(safe_dir)
            tile_geom = box(*tile_bounds)
            tile_gdf = gpd.GeoDataFrame(geometry=[tile_geom], crs=tile_crs)
            
            # Reproject to ground truth CRS if needed
            if tile_crs != gt_crs:
                tile_gdf = tile_gdf.to_crs(gt_crs)
            
            tile_geometries.append(tile_gdf.geometry[0])
        except Exception as e:
            print(f"Error processing {safe_dir}: {str(e)}")
    
    if not tile_geometries:
        return False, gt_gdf
    
    # Combine all tile geometries into a single geometry
    all_tiles_geom = unary_union(tile_geometries)
    
    # Check if ground truth is completely covered
    is_covered = gt_geom.within(all_tiles_geom)
    
    # Calculate missing area if not fully covered
    if not is_covered:
        missing_geom = gt_geom.difference(all_tiles_geom)
        missing_area = gpd.GeoDataFrame(geometry=[missing_geom], crs=gt_crs)
        return False, missing_area
    else:
        return True, None


def visualize_coverage(safe_dirs, ground_truth_path, output_path=None):
    """
    Visualize the coverage of Sentinel-2 tiles and ground truth area.
    
    Parameters:
    -----------
    safe_dirs : list
        List of paths to Sentinel-2 SAFE directories
    ground_truth_path : str
        Path to the ground truth GeoTIFF file
    output_path : str, optional
        Path to save the visualization
    """
    # Get ground truth bounds
    gt_bounds, gt_crs = get_ground_truth_bounds(ground_truth_path)
    gt_geom = box(*gt_bounds)
    gt_gdf = gpd.GeoDataFrame(geometry=[gt_geom], crs=gt_crs)
    
    # Get all tile bounds with their names
    tile_gdfs = []
    
    for safe_dir in safe_dirs:
        try:
            tile_bounds, tile_crs = get_sentinel_bounds(safe_dir)
            
            # Extract tile ID (e.g., T34SFJ) from directory name
            tile_id = os.path.basename(safe_dir).split('_')[5]
            
            # Create GeoDataFrame with the tile geometry and ID
            tile_gdf = gpd.GeoDataFrame(
                {'tile_id': [tile_id]},
                geometry=[box(*tile_bounds)], 
                crs=tile_crs
            )
            
            # Reproject to ground truth CRS if needed
            if tile_crs != gt_crs:
                tile_gdf = tile_gdf.to_crs(gt_crs)
            
            tile_gdfs.append(tile_gdf)
        except Exception as e:
            print(f"Error processing {safe_dir} for visualization: {str(e)}")
    
    if not tile_gdfs:
        print("No valid Sentinel-2 tiles found for visualization")
        return
    
    # Combine all tile GeoDataFrames
    all_tiles_gdf = pd.concat(tile_gdfs, ignore_index=True)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot tiles with different colors and labels
    all_tiles_gdf.plot(
        ax=ax,
        column='tile_id',  # Use tile_id for coloring
        categorical=True,
        legend=True,
        alpha=0.5,
        edgecolor='black'
    )
    
    # Plot ground truth in red
    gt_gdf.plot(
        ax=ax,
        color='red',
        alpha=0.3,
        edgecolor='red',
        label='Ground Truth'
    )
    
    # Add title and labels
    ax.set_title('Sentinel-2 Tiles and Ground Truth Coverage')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add ground truth to legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor='red', alpha=0.3, edgecolor='red', label='Ground Truth'))
    labels.append('Ground Truth')
    ax.legend(handles, labels, title='Coverage', loc='lower right')
    
    # Add tile IDs as text labels
    for idx, row in all_tiles_gdf.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, row['tile_id'], 
                ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Coverage visualization saved to {output_path}")
    else:
        plt.show()


def get_sentinel_bands_info(safe_dir):
    """
    Get information about the available bands in a Sentinel-2 tile.
    
    Parameters:
    -----------
    safe_dir : str
        Path to the Sentinel-2 SAFE directory
    
    Returns:
    --------
    bands_info : dict
        Dictionary with band information
    """
    img_data_paths = list(Path(safe_dir).glob('GRANULE/*/IMG_DATA'))
    if not img_data_paths:
        raise FileNotFoundError(f"No IMG_DATA directory found in {safe_dir}")
    
    img_data_path = img_data_paths[0]
    
    # Get all JP2 files excluding TCI (true color image)
    band_files = [f for f in img_data_path.glob('*.jp2') if not f.name.endswith('TCI.jp2')]
    
    # Initialize bands info dictionary
    bands_info = {
        'band_files': {},
        'resolutions': {
            '10m': ['B02', 'B03', 'B04', 'B08'],
            '20m': ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
            '60m': ['B01', 'B09', 'B10']
        }
    }
    
    # Group files by band
    for band_file in band_files:
        # Extract band name (e.g., B01, B02, etc.)
        band_name = band_file.name.split('_')[-1].split('.')[0]
        bands_info['band_files'][band_name] = str(band_file)
        
        # Read band metadata
        with rasterio.open(band_file) as src:
            if band_name not in bands_info:
                bands_info[band_name] = {}
            
            bands_info[band_name]['width'] = src.width
            bands_info[band_name]['height'] = src.height
            bands_info[band_name]['crs'] = src.crs
            bands_info[band_name]['transform'] = src.transform
    
    return bands_info


def main():
    """Main function to check Sentinel-2 data and ground truth coverage."""
    
    # Define directories
    sentinel_data_dir = "sentinel_data"
    ground_truth_path = "ground_truth.tif"
    output_dir = "processed_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Sentinel-2 zip files
    zip_files = glob.glob(os.path.join(sentinel_data_dir, "*.zip"))
    
    # Find all existing SAFE directories
    existing_safe_dirs = glob.glob(os.path.join(sentinel_data_dir, "*.SAFE"))
    
    # Extract zip files if needed
    safe_dirs = []
    
    for zip_file in zip_files:
        safe_dir = extract_sentinel_zip(zip_file, sentinel_data_dir)
        if safe_dir not in safe_dirs:
            safe_dirs.append(safe_dir)
    
    # Add existing SAFE directories that weren't just extracted
    for safe_dir in existing_safe_dirs:
        if safe_dir not in safe_dirs:
            safe_dirs.append(safe_dir)
    
    print(f"\nFound {len(safe_dirs)} Sentinel-2 datasets.")
    
    # Check cloud coverage for each SAFE directory
    valid_safe_dirs = []
    cloud_coverage_results = []
    
    print("\nChecking cloud coverage for each dataset:")
    print("-" * 60)
    print(f"{'Dataset':^40} | {'Cloud Coverage':^15}")
    print("-" * 60)
    
    for safe_dir in safe_dirs:
        dataset_name = os.path.basename(safe_dir)
        
        # Use improved cloud coverage detection
        cloud_coverage = check_cloud_coverage_improved(safe_dir)
        
        if cloud_coverage is not None:
            cloud_coverage_results.append({
                'dataset': dataset_name,
                'cloud_coverage': cloud_coverage,
                'valid': cloud_coverage <= 10.0
            })
            
            status = "✓ Valid" if cloud_coverage <= 10.0 else "✗ Invalid"
            print(f"{dataset_name:40} | {cloud_coverage:6.2f}% {status}")
            
            if cloud_coverage <= 10.0:
                valid_safe_dirs.append(safe_dir)
        else:
            print(f"{dataset_name:40} | {'Unknown':>6} ✓ Included by default")
            valid_safe_dirs.append(safe_dir)
            cloud_coverage_results.append({
                'dataset': dataset_name,
                'cloud_coverage': "Unknown",
                'valid': True
            })
    
    print("-" * 60)
    print(f"Valid datasets: {len(valid_safe_dirs)} out of {len(safe_dirs)}")
    
    # Check coverage of ground truth area
    print("\nChecking coverage of ground truth area...")
    is_covered, missing_area = check_coverage(valid_safe_dirs, ground_truth_path)
    
    if is_covered:
        print("✓ Ground truth area is completely covered by the Sentinel-2 tiles")
    else:
        print("⚠ Warning: Ground truth area is not completely covered by the Sentinel-2 tiles!")
        if missing_area is not None:
            print(f"  Missing area: {missing_area.area.iloc[0]:.2f} square units")
    
    # Create visualization of coverage
    visualize_coverage(valid_safe_dirs, ground_truth_path, os.path.join(output_dir, "coverage.png"))
    
    # Print summary of valid Sentinel-2 datasets
    print("\nValid Sentinel-2 SAFE directories:")
    for safe_dir in valid_safe_dirs:
        print(f"  {os.path.basename(safe_dir)}")
    
    # Get band information for the first valid SAFE directory
    if valid_safe_dirs:
        try:
            first_dataset = valid_safe_dirs[0]
            bands_info = get_sentinel_bands_info(first_dataset)
            
            print(f"\nBand information for {os.path.basename(first_dataset)}")
            print(f"  Available bands: {list(bands_info['band_files'].keys())}")
            
            # Print resolution information
            print("  Bands by resolution:")
            for resolution, bands in bands_info['resolutions'].items():
                print(f"    {resolution}: {', '.join(bands)}")
        except Exception as e:
            print(f"Error getting band information: {str(e)}")
    
    # Save cloud coverage results to a CSV file
    df = pd.DataFrame(cloud_coverage_results)
    df.to_csv(os.path.join(output_dir, "cloud_coverage_summary.csv"), index=False)
    print(f"\nCloud coverage summary saved to {os.path.join(output_dir, 'cloud_coverage_summary.csv')}")
    
    return valid_safe_dirs


if __name__ == "__main__":
    main()