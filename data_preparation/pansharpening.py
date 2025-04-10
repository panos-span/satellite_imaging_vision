"""
Pansharpening algorithms for improving spatial resolution of satellite imagery.
"""
import numpy as np
from skimage.transform import resize
import scipy.ndimage as ndimage


def simple_pansharpening(low_res_band, high_res_shape):
    """
    Apply simple pansharpening by resizing the low-resolution band to match the high-resolution shape.
    
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


def brovey_pansharpening(low_res_band, high_res_bands, high_res_shape):
    """
    Apply Brovey transform pansharpening to a low-resolution band.
    
    The Brovey transform uses a ratio approach where each low-resolution
    band is multiplied by the ratio of the high-resolution band to the
    sum of all low-resolution bands used for the transform.
    
    Parameters:
    -----------
    low_res_band : numpy.ndarray
        Low-resolution band data to be pansharpened
    high_res_bands : dict
        Dictionary with band names as keys and high-resolution band data as values
    high_res_shape : tuple
        Shape of high-resolution band (height, width)
    
    Returns:
    --------
    pansharpened_band : numpy.ndarray
        Pansharpened band data
    """
    # First, resize the low-resolution band to match the high-resolution shape
    low_res_resized = resize(
        low_res_band, 
        high_res_shape, 
        order=3,  # cubic interpolation
        preserve_range=True
    )
    
    # Create a synthetic intensity image from the high-resolution bands
    # Typically for Sentinel-2, we can use B02, B03, B04 bands
    if 'B02' in high_res_bands and 'B03' in high_res_bands and 'B04' in high_res_bands:
        # Use RGB bands to create intensity
        intensity = (high_res_bands['B04'] + high_res_bands['B03'] + high_res_bands['B02']) / 3.0
    elif 'B08' in high_res_bands:
        # If RGB bands not available, use NIR (B08) as intensity
        intensity = high_res_bands['B08']
    else:
        # If no suitable high-res bands, just return the resized low-res band
        return low_res_resized
    
    # Avoid division by zero
    epsilon = np.finfo(low_res_resized.dtype).eps
    intensity = np.maximum(intensity, epsilon)
    
    # Apply Brovey transform
    pansharpened_band = low_res_resized * (intensity / intensity.mean())
    
    # Clip to original data range
    min_val = low_res_band.min()
    max_val = low_res_band.max()
    pansharpened_band = np.clip(pansharpened_band, min_val, max_val)
    
    return pansharpened_band.astype(low_res_band.dtype)


def hpf_pansharpening(low_res_band, high_res_band, high_res_shape):
    """
    Apply High-Pass Filter (HPF) pansharpening.
    
    The HPF method extracts high-frequency spatial details from the
    high-resolution band using a high-pass filter and adds these
    details to the upsampled low-resolution band.
    
    Parameters:
    -----------
    low_res_band : numpy.ndarray
        Low-resolution band data to be pansharpened
    high_res_band : numpy.ndarray
        High-resolution band data (typically PAN band)
    high_res_shape : tuple
        Shape of high-resolution band (height, width)
    
    Returns:
    --------
    pansharpened_band : numpy.ndarray
        Pansharpened band data
    """
    # First, resize the low-resolution band to match the high-resolution shape
    low_res_resized = resize(
        low_res_band, 
        high_res_shape, 
        order=3,  # cubic interpolation
        preserve_range=True
    )
    
    # Create a high-pass filter kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]]) / 9.0
    
    # Apply high-pass filter to the high-resolution band
    high_freq = ndimage.convolve(high_res_band, kernel)
    
    # Normalize the high-frequency details to avoid changing the overall intensity
    high_freq = high_freq - high_freq.mean()
    
    # Add high-frequency details to the upsampled low-resolution band
    # The weight factor can be adjusted (0.7 is a reasonable default)
    weight = 0.7
    pansharpened_band = low_res_resized + weight * high_freq
    
    # Clip to original data range
    min_val = low_res_band.min()
    max_val = low_res_band.max()
    pansharpened_band = np.clip(pansharpened_band, min_val, max_val)
    
    return pansharpened_band.astype(low_res_band.dtype)


# Factory function to get the appropriate pansharpening method
def get_pansharpening_method(method_name):
    """
    Get a pansharpening function by name.
    
    Parameters:
    -----------
    method_name : str
        Name of the pansharpening method: 'simple', 'brovey', or 'hpf'
    
    Returns:
    --------
    method : function
        Pansharpening function
    """
    methods = {
        'simple': simple_pansharpening,
        'brovey': brovey_pansharpening,
        'hpf': hpf_pansharpening,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown pansharpening method: {method_name}")
    
    return methods[method_name]