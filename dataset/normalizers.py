"""
Normalization utilities for satellite imagery.

This module provides specialized normalization approaches for multispectral 
satellite imagery, including Sentinel-2 specific normalizations.
"""
import numpy as np
import torch
import rasterio
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


class MinMaxScaler:
    """
    Normalize each band to [0, 1] range based on min and max values.
    
    This class wraps sklearn.preprocessing.MinMaxScaler to handle
    multi-band satellite imagery more conveniently.
    
    Parameters:
    -----------
    feature_range : tuple, optional
        Desired range of transformed data (default: (0, 1))
    min_values : list, optional
        List of minimum values for each band. If None, calculated from data
    max_values : list, optional
        List of maximum values for each band. If None, calculated from data
    """
    def __init__(self, feature_range=(0, 1), min_values=None, max_values=None):
        self.feature_range = feature_range
        self.min_values = min_values
        self.max_values = max_values
        self.fitted = min_values is not None and max_values is not None
        
        # Create a list of sklearn MinMaxScalers (one per band)
        self.scalers = []
    
    def fit(self, images):
        """
        Calculate min and max values from a set of images.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        """
        if isinstance(images, list):
            # Stack along batch dimension
            images = np.stack(images, axis=0)
        
        # Extract number of channels
        num_channels = images.shape[1]
        
        # Initialize scalers for each channel
        self.scalers = [SklearnMinMaxScaler(feature_range=self.feature_range) for _ in range(num_channels)]
        
        # Reshape and fit each channel separately
        for i in range(num_channels):
            # Extract channel and reshape to have samples in rows
            channel_data = images[:, i, :, :].reshape(images.shape[0], -1)
            self.scalers[i].fit(channel_data)
        
        # Store min and max values for reference
        self.min_values = [scaler.data_min_.min() for scaler in self.scalers]
        self.max_values = [scaler.data_max_.max() for scaler in self.scalers]
        
        self.fitted = True
        return self
    
    def transform(self, image):
        """
        Normalize image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (CHW format)
        
        Returns:
        --------
        normalized : numpy.ndarray
            Normalized image
        """
        if not self.fitted:
            raise ValueError("Scaler needs to be fitted before transform")
        
        channels = image.shape[0]
        if len(self.scalers) != channels:
            raise ValueError(f"Image has {channels} channels, but scaler has {len(self.scalers)} scalers")
        
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(channels):
            # Reshape channel to 2D array with samples in rows
            channel_data = image[i].reshape(1, -1)
            # Transform the data
            normalized_data = self.scalers[i].transform(channel_data)
            # Reshape back to original shape
            normalized[i] = normalized_data.reshape(image[i].shape)
        
        return normalized
    
    def fit_transform(self, images):
        """
        Fit scaler and transform images.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        
        Returns:
        --------
        normalized : numpy.ndarray
            Normalized images
        """
        self.fit(images)
        
        if isinstance(images, list):
            return [self.transform(img) for img in images]
        else:
            return np.stack([self.transform(images[i]) for i in range(images.shape[0])])


class StandardScaler:
    """
    Normalize each band to zero mean and unit variance.
    
    This class wraps sklearn.preprocessing.StandardScaler to handle
    multi-band satellite imagery more conveniently.
    
    Parameters:
    -----------
    mean_values : list, optional
        List of mean values for each band. If None, calculated from data
    std_values : list, optional
        List of standard deviation values for each band. If None, calculated from data
    """
    def __init__(self, mean_values=None, std_values=None):
        self.mean_values = mean_values
        self.std_values = std_values
        self.fitted = mean_values is not None and std_values is not None
        
        # Create a list of sklearn StandardScalers (one per band)
        self.scalers = []
    
    def fit(self, images):
        """
        Calculate mean and standard deviation from a set of images.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        """
        if isinstance(images, list):
            # Stack along batch dimension
            images = np.stack(images, axis=0)
        
        # Extract number of channels
        num_channels = images.shape[1]
        
        # Initialize scalers for each channel
        self.scalers = [SklearnStandardScaler() for _ in range(num_channels)]
        
        # Reshape and fit each channel separately
        for i in range(num_channels):
            # Extract channel and reshape to have samples in rows
            channel_data = images[:, i, :, :].reshape(images.shape[0], -1)
            self.scalers[i].fit(channel_data)
        
        # Store mean and std values for reference
        self.mean_values = [scaler.mean_.mean() for scaler in self.scalers]
        self.std_values = [scaler.scale_.mean() for scaler in self.scalers]
        
        self.fitted = True
        return self
    
    def transform(self, image):
        """
        Normalize image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input image (CHW format)
        
        Returns:
        --------
        normalized : numpy.ndarray
            Normalized image
        """
        if not self.fitted:
            raise ValueError("Scaler needs to be fitted before transform")
        
        channels = image.shape[0]
        if len(self.scalers) != channels:
            raise ValueError(f"Image has {channels} channels, but scaler has {len(self.scalers)} scalers")
        
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(channels):
            # Reshape channel to 2D array with samples in rows
            channel_data = image[i].reshape(1, -1)
            # Transform the data
            normalized_data = self.scalers[i].transform(channel_data)
            # Reshape back to original shape
            normalized[i] = normalized_data.reshape(image[i].shape)
        
        return normalized
    
    def fit_transform(self, images):
        """
        Fit scaler and transform images.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        
        Returns:
        --------
        normalized : numpy.ndarray
            Normalized images
        """
        self.fit(images)
        
        if isinstance(images, list):
            return [self.transform(img) for img in images]
        else:
            return np.stack([self.transform(images[i]) for i in range(images.shape[0])])


class Sentinel2Normalizer:
    """
    Specialized normalizer for Sentinel-2 imagery.
    
    This normalizer handles the specific band structure of Sentinel-2,
    including different methods for RGB vs other bands.
    
    Parameters:
    -----------
    method : str
        Normalization method ('minmax', 'standard', or 'pretrained')
    rgb_indices : list
        Indices of RGB bands
    """
    def __init__(self, method='pretrained', rgb_indices=[0, 1, 2]):
        self.method = method
        self.rgb_indices = rgb_indices
        
        # Define band names for reference
        self.band_names = [
            'B02', 'B03', 'B04',  # RGB bands
            'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12',  # Other bands
            'B01', 'B09', 'B10'  # 60m bands (if included)
        ]
        
        # For pretrained method, define ImageNet mean and std
        if method == 'pretrained':
            self.rgb_mean = [0.485, 0.456, 0.406]
            self.rgb_std = [0.229, 0.224, 0.225]
            self.other_mean = 0.5
            self.other_std = 0.5
        
        # Initialize scalers for other methods
        self.scalers = None
    
    def fit(self, images, image_path=None):
        """
        Fit normalizer on training data or load from image metadata.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        image_path : str, optional
            Path to a Sentinel-2 GeoTIFF to get stats from metadata
        """
        if self.method == 'pretrained':
            # No fitting needed for pretrained method
            pass
        
        elif image_path is not None:
            # Try to get statistics from image metadata
            try:
                with rasterio.open(image_path) as src:
                    metadata = src.tags()
                    if 'STATISTICS_MEAN' in metadata and 'STATISTICS_STDDEV' in metadata:
                        means = metadata['STATISTICS_MEAN'].split(',')
                        stds = metadata['STATISTICS_STDDEV'].split(',')
                        
                        if len(means) == src.count and len(stds) == src.count:
                            means = [float(m) for m in means]
                            stds = [float(s) for s in stds]
                            
                            if self.method == 'minmax':
                                # Approximate min-max from mean and std
                                mins = [m - 3*s for m, s in zip(means, stds)]
                                maxs = [m + 3*s for m, s in zip(means, stds)]
                                
                                # Create scikit-learn scalers for each band
                                self.scalers = []
                                for mn, mx in zip(mins, maxs):
                                    scaler = SklearnMinMaxScaler()
                                    scaler.fit(np.array([[mn], [mx]]))
                                    self.scalers.append(scaler)
                            else:  # standard
                                # Create scikit-learn scalers for each band
                                self.scalers = []
                                for m, s in zip(means, stds):
                                    scaler = SklearnStandardScaler()
                                    # Provide fake data that will result in these stats
                                    # This allows us to initialize the scaler without actual data
                                    fake_data = np.array([m - s, m, m + s]).reshape(-1, 1)
                                    scaler.fit(fake_data)
                                    self.scalers.append(scaler)
                            
                            return self
            except Exception as e:
                print(f"Could not read statistics from image metadata: {e}")
                print("Calculating statistics from data instead.")
        
        # If we get here, calculate from data
        if self.method == 'minmax':
            if images is None:
                raise ValueError("Cannot fit MinMaxScaler without images or valid metadata")
            
            if isinstance(images, list):
                # Stack along batch dimension
                images = np.stack(images, axis=0)
            
            # Extract number of channels
            num_channels = images.shape[1]
            
            # Initialize scalers for each channel
            self.scalers = []
            for i in range(num_channels):
                # Extract channel and reshape to have samples in rows
                channel_data = images[:, i, :, :].reshape(images.shape[0], -1)
                scaler = SklearnMinMaxScaler()
                scaler.fit(channel_data)
                self.scalers.append(scaler)
                
        elif self.method == 'standard':
            if images is None:
                raise ValueError("Cannot fit StandardScaler without images or valid metadata")
            
            if isinstance(images, list):
                # Stack along batch dimension
                images = np.stack(images, axis=0)
            
            # Extract number of channels
            num_channels = images.shape[1]
            
            # Initialize scalers for each channel
            self.scalers = []
            for i in range(num_channels):
                # Extract channel and reshape to have samples in rows
                channel_data = images[:, i, :, :].reshape(images.shape[0], -1)
                scaler = SklearnStandardScaler()
                scaler.fit(channel_data)
                self.scalers.append(scaler)
        
        return self
    
    def transform(self, image):
        """
        Normalize image based on the selected method.
        
        Parameters:
        -----------
        image : numpy.ndarray or torch.Tensor
            Input image (CHW format)
        
        Returns:
        --------
        normalized : numpy.ndarray or torch.Tensor
            Normalized image in the same format as input
        """
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            image_np = image.numpy()
        else:
            image_np = image
        
        if self.method == 'pretrained':
            # Create output array
            normalized = np.zeros_like(image_np, dtype=np.float32)
            
            # Normalize RGB bands with ImageNet stats
            for i, idx in enumerate(self.rgb_indices):
                if idx < image_np.shape[0]:
                    normalized[idx] = (image_np[idx] - self.rgb_mean[i]) / self.rgb_std[i]
            
            # Normalize other bands to [0-1] and then apply 0.5 mean and std
            for i in range(image_np.shape[0]):
                if i not in self.rgb_indices:
                    # First rescale to [0-1]
                    band = image_np[i]
                    if band.max() > band.min():
                        band = (band - band.min()) / (band.max() - band.min())
                    
                    # Then normalize with mean 0.5 and std 0.5
                    normalized[i] = (band - self.other_mean) / self.other_std
        
        elif self.scalers is not None:
            # Use fitted scalers
            normalized = np.zeros_like(image_np, dtype=np.float32)
            
            for i in range(min(image_np.shape[0], len(self.scalers))):
                # Reshape channel to 2D array with samples in rows
                channel_data = image_np[i].reshape(1, -1)
                # Transform the data
                normalized_data = self.scalers[i].transform(channel_data)
                # Reshape back to original shape
                normalized[i] = normalized_data.reshape(image_np[i].shape)
        else:
            raise ValueError("Normalizer needs to be fitted before transform")
        
        # Convert back to tensor if input was a tensor
        if is_tensor:
            return torch.from_numpy(normalized)
        else:
            return normalized
    
    def fit_transform(self, images, image_path=None):
        """
        Fit normalizer and transform images.
        
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images (CHW format) or a single 4D array (BCHW format)
        image_path : str, optional
            Path to a Sentinel-2 GeoTIFF to get stats from metadata
        
        Returns:
        --------
        normalized : numpy.ndarray or list
            Normalized images in the same format as input
        """
        self.fit(images, image_path)
        
        if isinstance(images, list):
            return [self.transform(img) for img in images]
        else:
            return np.stack([self.transform(images[i]) for i in range(images.shape[0])])


def get_sentinel2_statistics(image_path, bands=None, sample_size=1000):
    """
    Calculate statistics for Sentinel-2 bands from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to Sentinel-2 GeoTIFF
    bands : list, optional
        List of band indices to calculate statistics for. If None, all bands.
    sample_size : int, optional
        Number of random samples to use for calculation
    
    Returns:
    --------
    stats : dict
        Dictionary with mean, std, min, and max values for each band
    """
    import random
    
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        num_bands = src.count
        
        if bands is None:
            bands = list(range(1, num_bands + 1))  # 1-based indexing
        
        # Generate random pixel locations
        random.seed(42)  # For reproducibility
        sample_pixels = [(random.randint(0, height - 1), random.randint(0, width - 1)) 
                         for _ in range(sample_size)]
        
        stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': []
        }
        
        # Read values for each band
        for band_idx in bands:
            band_data = src.read(band_idx)
            
            # Collect sample values
            sample_values = [band_data[y, x] for y, x in sample_pixels 
                             if 0 <= y < height and 0 <= x < width]
            
            # Calculate statistics
            sample_values = np.array(sample_values)
            sample_values = sample_values[~np.isnan(sample_values)]  # Remove NaN values
            
            mean_val = sample_values.mean()
            std_val = sample_values.std()
            min_val = sample_values.min()
            max_val = sample_values.max()
            
            stats['mean'].append(mean_val)
            stats['std'].append(std_val)
            stats['min'].append(min_val)
            stats['max'].append(max_val)
            
            print(f"Band {band_idx}: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}")
    
    return stats