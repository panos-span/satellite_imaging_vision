"""
Main class for processing Sentinel-2 data.
"""
import os
import glob
import shutil
import numpy as np
import rasterio
from pathlib import Path
import zipfile
from tqdm import tqdm

from geospatial_utils import get_ground_truth_info, align_raster_to_reference, merge_rasters, create_multiband_stack
from pansharpening import get_pansharpening_method
from validity_masks import ValidityMaskCreator, merge_validity_masks
from visualization import visualize_rgb, visualize_dataset, visualize_validity_mask


class SentinelProcessor:
    """
    Class for processing Sentinel-2 data for land cover classification.
    """
    
    def __init__(self, output_base_dir='processed_data'):
        """
        Initialize the Sentinel processor.
        
        Parameters:
        -----------
        output_base_dir : str, optional
            Base directory for output
        """
        self.output_base_dir = output_base_dir
        self.ground_truth_info = None
        
        # Create output directories
        os.makedirs(output_base_dir, exist_ok=True)
        self.merged_dir = os.path.join(output_base_dir, "merged")
        self.dataset_dir = os.path.join(output_base_dir, "dataset")
        
        os.makedirs(self.merged_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    def extract_safe_dirs(self, data_dir):
        """
        Extract Sentinel-2 zip files and return paths to SAFE directories.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing Sentinel-2 data (zip files or SAFE directories)
        
        Returns:
        --------
        safe_dirs : list
            List of paths to SAFE directories
        """
        # Find existing SAFE directories
        safe_dirs = glob.glob(os.path.join(data_dir, "*.SAFE"))
        
        # Find zip files
        zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
        
        if not safe_dirs and not zip_files:
            raise FileNotFoundError(f"No Sentinel-2 data found in {data_dir}")
        
        # Extract zip files if needed
        for zip_file in zip_files:
            safe_name = os.path.basename(zip_file).replace('.zip', '')
            safe_dir = os.path.join(data_dir, safe_name)
            
            # Extract if not already extracted
            if not os.path.exists(safe_dir):
                print(f"Extracting {os.path.basename(zip_file)}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                
                if safe_dir not in safe_dirs:
                    safe_dirs.append(safe_dir)
        
        print(f"Found {len(safe_dirs)} Sentinel-2 SAFE directories")
        return safe_dirs
    
    def load_ground_truth(self, ground_truth_path):
        """
        Load ground truth data and extract information.
        
        Parameters:
        -----------
        ground_truth_path : str
            Path to ground truth GeoTIFF file
        """
        print("Loading ground truth data...")
        self.ground_truth_path = ground_truth_path
        self.ground_truth_info = get_ground_truth_info(ground_truth_path)
        
        # Print ground truth information
        print(f"Ground truth CRS: {self.ground_truth_info['crs']}")
        print(f"Ground truth shape: {self.ground_truth_info['shape']}")
        print(f"Ground truth unique values: {self.ground_truth_info['unique_values']}")
        
        if 'classes' in self.ground_truth_info:
            print(f"Ground truth classes: {self.ground_truth_info['classes']}")
    
    def process_sentinel_tile(self, safe_dir, pansharpening_method='brovey', create_validity_mask=True):
        """
        Process a single Sentinel-2 SAFE directory.
        
        Parameters:
        -----------
        safe_dir : str
            Path to Sentinel-2 SAFE directory
        pansharpening_method : str, optional
            Pansharpening method to use: 'simple', 'brovey', or 'hpf'
        create_validity_mask : bool, optional
            Whether to create a validity mask
        
        Returns:
        --------
        result : dict
            Dictionary with processing results
        """
        if not self.ground_truth_info:
            raise ValueError("Ground truth not loaded. Call load_ground_truth first.")
        
        # Extract tile ID from SAFE directory name
        tile_id = os.path.basename(safe_dir).split('_')[5]
        print(f"Processing tile {tile_id} with {pansharpening_method} pansharpening...")
        
        # Create tile-specific output directories
        pansharpened_dir = os.path.join(self.output_base_dir, f"pansharpened_{tile_id}")
        aligned_dir = os.path.join(self.output_base_dir, f"aligned_{tile_id}")
        mask_dir = os.path.join(self.output_base_dir, f"validity_mask_{tile_id}")
        
        os.makedirs(pansharpened_dir, exist_ok=True)
        os.makedirs(aligned_dir, exist_ok=True)
        if create_validity_mask:
            os.makedirs(mask_dir, exist_ok=True)
        
        # Step 1: Find all band files
        img_data_path = list(Path(safe_dir).glob('GRANULE/*/IMG_DATA'))[0]
        band_files = [f for f in img_data_path.glob('*.jp2') if not f.name.endswith('TCI.jp2')]
        
        # Define band resolution groups
        high_res_bands = ['B02', 'B03', 'B04', 'B08']  # 10m
        medium_res_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # 20m
        low_res_bands = ['B01', 'B09', 'B10']  # 60m
        
        # Step 2: Get high-resolution band metadata and data
        high_res_band_data = {}
        high_res_profile = None
        high_res_shape = None
        high_res_transform = None
        high_res_crs = None
        
        # Find and read high-resolution bands
        for band_name in high_res_bands:
            band_files_filtered = [f for f in band_files if f.name.split('_')[-1].split('.')[0] == band_name]
            if band_files_filtered:
                high_res_band_file = band_files_filtered[0]
                with rasterio.open(high_res_band_file) as src:
                    # Store band data
                    high_res_band_data[band_name] = src.read(1)
                    
                    # Store metadata from the first high-resolution band
                    if high_res_profile is None:
                        high_res_profile = src.profile.copy()
                        high_res_shape = (src.height, src.width)
                        high_res_transform = src.transform
                        high_res_crs = src.crs
        
        if not high_res_band_data:
            raise FileNotFoundError(f"No high-resolution bands found in {img_data_path}")
        
        print(f"Found {len(high_res_band_data)} high-resolution bands with shape {high_res_shape}")
        
        # Step 3: Apply pansharpening to all bands
        print("Applying pansharpening...")
        pansharpened_bands_paths = {}
        pansharpening_func = get_pansharpening_method(pansharpening_method)
        
        for band_file in tqdm(band_files, desc="Pansharpening bands"):
            band_name = band_file.name.split('_')[-1].split('.')[0]
            output_path = os.path.join(pansharpened_dir, f"{band_name}_pansharpened.tif")
            
            with rasterio.open(band_file) as src:
                # Determine if band needs pansharpening
                needs_pansharpening = (band_name in medium_res_bands or band_name in low_res_bands)
                
                if needs_pansharpening:
                    # Read band data
                    band_data = src.read(1)
                    
                    # Apply appropriate pansharpening method
                    if pansharpening_method == 'brovey' and len(high_res_band_data) >= 3:
                        pansharpened_data = pansharpening_func(
                            band_data, high_res_band_data, high_res_shape
                        )
                    elif pansharpening_method == 'hpf' and 'B08' in high_res_band_data:
                        # Use NIR band (B08) as the high-resolution band for HPF
                        pansharpened_data = pansharpening_func(
                            band_data, high_res_band_data['B08'], high_res_shape
                        )
                    else:
                        # Fallback to simple resize
                        from pansharpening import simple_pansharpening
                        pansharpened_data = simple_pansharpening(band_data, high_res_shape)
                else:
                    # No pansharpening needed, just read the band
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
        
        # Step 4: Align with ground truth
        print("Aligning with ground truth...")
        aligned_bands_paths = {}
        
        for band_name, band_path in tqdm(pansharpened_bands_paths.items(), desc="Aligning bands"):
            output_path = os.path.join(aligned_dir, f"{band_name}_aligned.tif")
            aligned_path = align_raster_to_reference(
                band_path, self.ground_truth_info, output_path
            )
            aligned_bands_paths[band_name] = aligned_path
        
        # Step 5: Create validity mask if requested
        validity_mask_path = None
        if create_validity_mask:
            print("Creating validity mask...")
            mask_creator = ValidityMaskCreator(safe_dir, aligned_bands_paths)
            validity_mask_path = os.path.join(mask_dir, 'validity_mask.tif')
            mask_creator.create_validity_mask(validity_mask_path)
        
        return {
            'tile_id': tile_id,
            'aligned_bands_paths': aligned_bands_paths,
            'validity_mask_path': validity_mask_path
        }
    
    def merge_tiles(self, tile_results):
        """
        Merge multiple processed Sentinel-2 tiles.
        
        Parameters:
        -----------
        tile_results : list
            List of dictionaries with processing results
        
        Returns:
        --------
        result : dict
            Dictionary with merged results
        """
        if not self.ground_truth_info:
            raise ValueError("Ground truth not loaded. Call load_ground_truth first.")
        
        # Step 1: Prepare lists for merging
        aligned_bands_list = [result['aligned_bands_paths'] for result in tile_results]
        validity_mask_paths = [result['validity_mask_path'] for result in tile_results 
                              if result['validity_mask_path'] is not None]
        
        # Step 2: Get all unique band names
        all_band_names = set()
        for aligned_bands in aligned_bands_list:
            all_band_names.update(aligned_bands.keys())
        
        # Sort band names for consistency
        all_band_names = sorted(all_band_names)
        
        # Step 3: Merge each band
        print("Merging aligned bands...")
        merged_bands_paths = {}
        
        for band_name in tqdm(all_band_names, desc="Merging bands"):
            # Collect all files for this band
            band_files = []
            for aligned_bands in aligned_bands_list:
                if band_name in aligned_bands:
                    band_files.append(aligned_bands[band_name])
            
            if not band_files:
                print(f"No files found for band {band_name}, skipping")
                continue
            
            # Merge the files
            output_path = os.path.join(self.merged_dir, f"{band_name}_merged.tif")
            merged_path = merge_rasters(
                band_files, 
                output_path, 
                self.ground_truth_info['bounds'],
                self.ground_truth_info['crs']
            )
            
            merged_bands_paths[band_name] = merged_path
        
        # Step 4: Merge validity masks if available
        merged_validity_mask_path = None
        if validity_mask_paths:
            print("Merging validity masks...")
            merged_validity_mask_path = os.path.join(self.output_base_dir, "merged_validity_mask.tif")
            merge_validity_masks(
                validity_mask_paths,
                self.ground_truth_info,
                merged_validity_mask_path
            )
        
        # Step 5: Create multiband stack
        print("Creating multiband stack...")
        multiband_path = os.path.join(self.output_base_dir, "combined_multiband.tif")
        multiband_path, band_order = create_multiband_stack(merged_bands_paths, multiband_path)
        
        return {
            'merged_bands_paths': merged_bands_paths,
            'merged_validity_mask_path': merged_validity_mask_path,
            'multiband_path': multiband_path,
            'band_order': band_order
        }
    
    def create_dataset(self, merge_result):
        """
        Create the final dataset for training.
        
        Parameters:
        -----------
        merge_result : dict
            Dictionary with merged results
        
        Returns:
        --------
        dataset_info : dict
            Dictionary with dataset information
        """
        # Step 1: Copy multiband image and ground truth to dataset directory
        dataset_image_path = os.path.join(self.dataset_dir, 'sentinel_image.tif')
        dataset_mask_path = os.path.join(self.dataset_dir, 'ground_truth.tif')
        
        shutil.copy2(merge_result['multiband_path'], dataset_image_path)
        shutil.copy2(self.ground_truth_path, dataset_mask_path)
        
        # Step 2: Copy validity mask if available
        dataset_validity_path = None
        if merge_result['merged_validity_mask_path'] is not None:
            dataset_validity_path = os.path.join(self.dataset_dir, 'validity_mask.tif')
            shutil.copy2(merge_result['merged_validity_mask_path'], dataset_validity_path)
        
        # Step 3: Create visualizations
        print("Creating dataset visualizations...")
        visualize_dataset(dataset_image_path, dataset_mask_path, self.dataset_dir)
        
        if dataset_validity_path:
            validity_vis_path = os.path.join(self.dataset_dir, 'validity_mask.png')
            visualize_validity_mask(dataset_validity_path, validity_vis_path)
        
        # Step 4: Create RGB visualization
        rgb_vis_path = os.path.join(self.output_base_dir, 'rgb_visualization.png')
        visualize_rgb(dataset_image_path, rgb_vis_path)
        
        # Step 5: Read dataset metadata
        with rasterio.open(dataset_image_path) as img_src, rasterio.open(dataset_mask_path) as mask_src:
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
                'image_path': dataset_image_path,
                'mask_path': dataset_mask_path,
                'validity_mask_path': dataset_validity_path,
                'image_shape': (img_src.height, img_src.width, img_src.count),
                'mask_shape': (mask_src.height, mask_src.width),
                'crs': img_src.crs.to_string(),
                'band_names': band_names,
                'class_values': unique_values.tolist(),
                'class_names': class_names
            }
        
        # Print dataset information
        print(f"\nDataset Information:")
        print(f"  Image shape: {dataset_info['image_shape']}")
        print(f"  Mask shape: {dataset_info['mask_shape']}")
        print(f"  Number of bands: {len(band_names)}")
        print(f"  Band names: {band_names}")
        print(f"  Unique class values: {unique_values}")
        
        if class_names:
            print(f"  Class names: {class_names}")
        
        print(f"\nDataset saved to {self.dataset_dir}")
        
        return dataset_info
    
    def process_all(self, safe_dirs, ground_truth_path, pansharpening_method='brovey', create_validity_masks=True):
        """
        Process all Sentinel-2 SAFE directories and create a dataset.
        
        Parameters:
        -----------
        safe_dirs : list
            List of paths to Sentinel-2 SAFE directories
        ground_truth_path : str
            Path to ground truth GeoTIFF file
        pansharpening_method : str, optional
            Pansharpening method to use: 'simple', 'brovey', or 'hpf'
        create_validity_masks : bool, optional
            Whether to create validity masks
        
        Returns:
        --------
        dataset_info : dict
            Dictionary with dataset information
        """
        # Step 1: Load ground truth
        self.load_ground_truth(ground_truth_path)
        
        # Step 2: Process each tile
        tile_results = []
        for i, safe_dir in enumerate(safe_dirs):
            print(f"\nProcessing tile {i+1}/{len(safe_dirs)}: {os.path.basename(safe_dir)}")
            result = self.process_sentinel_tile(
                safe_dir, 
                pansharpening_method=pansharpening_method,
                create_validity_mask=create_validity_masks
            )
            tile_results.append(result)
        
        # Step 3: Merge tiles
        print("\nMerging tiles...")
        merge_result = self.merge_tiles(tile_results)
        
        # Step 4: Create dataset
        print("\nCreating dataset...")
        dataset_info = self.create_dataset(merge_result)
        
        return dataset_info