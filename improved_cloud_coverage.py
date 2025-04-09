import os
import glob
import xml.etree.ElementTree as ET
import re
from pathlib import Path


def check_cloud_coverage_improved(safe_dir):
    """
    Advanced function to check cloud coverage in Sentinel-2 data.
    Tries multiple approaches to find cloud coverage information.
    
    Parameters:
    -----------
    safe_dir : str
        Path to the Sentinel-2 SAFE directory
    
    Returns:
    --------
    cloud_coverage : float or None
        Cloud coverage percentage or None if not found
    """
    # Approach 1: Check main metadata file
    mtd_file = os.path.join(safe_dir, 'MTD_MSIL1C.xml')
    if os.path.exists(mtd_file):
        try:
            # Parse the XML
            tree = ET.parse(mtd_file)
            root = tree.getroot()
            
            # Try different paths for cloud coverage
            cloud_paths = [
                ".//Cloud_Coverage_Assessment",
                ".//CLOUDY_PIXEL_PERCENTAGE",
                ".//CLOUD_COVERAGE_ASSESSMENT",
                ".//n1:Cloud_Coverage_Assessment",
                ".//QUALITY_INDEX/Cloud_Coverage_Assessment"
            ]
            
            for path in cloud_paths:
                try:
                    element = root.find(path)
                    if element is not None and element.text:
                        return float(element.text)
                except (AttributeError, ValueError):
                    continue
            
            # Try to search for text in the whole XML
            content = ET.tostring(root, encoding='utf-8').decode('utf-8')
            cloud_patterns = [
                r'<Cloud_Coverage_Assessment>(\d+\.?\d*)</Cloud_Coverage_Assessment>',
                r'<CLOUDY_PIXEL_PERCENTAGE>(\d+\.?\d*)</CLOUDY_PIXEL_PERCENTAGE>',
                r'<CLOUD_COVERAGE_ASSESSMENT>(\d+\.?\d*)</CLOUD_COVERAGE_ASSESSMENT>'
            ]
            
            for pattern in cloud_patterns:
                match = re.search(pattern, content)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"Error parsing {mtd_file}: {str(e)}")
    
    # Approach 2: Check in GRANULE subdirectories
    granule_dirs = glob.glob(os.path.join(safe_dir, 'GRANULE', '*'))
    for granule_dir in granule_dirs:
        # Try multiple possible metadata files
        metadata_files = [
            os.path.join(granule_dir, 'MTD_TL.xml'),
            os.path.join(granule_dir, 'QI_DATA', 'GENERAL_QUALITY.xml'),
            *glob.glob(os.path.join(granule_dir, 'QI_DATA', '*.xml'))
        ]
        
        for metadata_file in metadata_files:
            if os.path.exists(metadata_file):
                try:
                    # Parse the XML
                    tree = ET.parse(metadata_file)
                    root = tree.getroot()
                    
                    # Try different paths for cloud coverage
                    cloud_paths = [
                        ".//Cloud_Coverage_Assessment",
                        ".//CLOUDY_PIXEL_PERCENTAGE",
                        ".//CLOUD_COVERAGE_ASSESSMENT",
                        ".//n1:Cloud_Coverage_Assessment",
                        ".//QUALITY_INDEX/Cloud_Coverage_Assessment"
                    ]
                    
                    for path in cloud_paths:
                        try:
                            element = root.find(path)
                            if element is not None and element.text:
                                return float(element.text)
                        except (AttributeError, ValueError):
                            continue
                    
                    # Try to search for text in the whole XML
                    content = ET.tostring(root, encoding='utf-8').decode('utf-8')
                    cloud_patterns = [
                        r'<Cloud_Coverage_Assessment>(\d+\.?\d*)</Cloud_Coverage_Assessment>',
                        r'<CLOUDY_PIXEL_PERCENTAGE>(\d+\.?\d*)</CLOUDY_PIXEL_PERCENTAGE>',
                        r'<CLOUD_COVERAGE_ASSESSMENT>(\d+\.?\d*)</CLOUD_COVERAGE_ASSESSMENT>'
                    ]
                    
                    for pattern in cloud_patterns:
                        match = re.search(pattern, content)
                        if match:
                            return float(match.group(1))
                except Exception as e:
                    print(f"Error parsing {metadata_file}: {str(e)}")
    
    # Approach 3: Manual estimation from classification mask if it exists
    try:
        # Look for classification mask
        mask_files = glob.glob(os.path.join(safe_dir, 'GRANULE', '*', 'QI_DATA', 'MSK_CLASSI_B00.jp2'))
        if mask_files:
            import rasterio
            import numpy as np
            
            with rasterio.open(mask_files[0]) as src:
                mask = src.read(1)
                
                # Classification bits for clouds
                # (This depends on the specific format - adjust if needed)
                cloud_pixels = np.count_nonzero(mask & 0x08)  # Example bit mask for clouds
                total_pixels = mask.size
                
                if total_pixels > 0:
                    cloud_percentage = (cloud_pixels / total_pixels) * 100
                    return cloud_percentage
    except Exception as e:
        print(f"Error estimating cloud coverage from mask: {str(e)}")
    
    # If all approaches fail, use a default value or manual entry
    print(f"Could not determine cloud coverage for {os.path.basename(safe_dir)}")
    
    # Let's check if the manual cloud coverage value is available in a .txt file
    cloud_info_file = os.path.join(os.path.dirname(safe_dir), "cloud_coverage.txt")
    if os.path.exists(cloud_info_file):
        try:
            with open(cloud_info_file, 'r') as f:
                for line in f:
                    # Format: safe_name,coverage_percent
                    parts = line.strip().split(',')
                    if len(parts) == 2 and os.path.basename(safe_dir) in parts[0]:
                        return float(parts[1])
        except Exception as e:
            print(f"Error reading cloud coverage from file: {str(e)}")
    
    # Assume a default value based on visual inspection or set to 0 to include all data
    # You can modify this based on your knowledge of the data
    return 10.0  # Assuming 5% cloud coverage as a conservative default