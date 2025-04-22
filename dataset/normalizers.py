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
import random

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

    try:
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            num_bands = src.count

            if bands is None:
                bands = list(range(1, num_bands + 1))  # 1-based indexing

            # Generate random pixel locations
            random.seed(42)  # For reproducibility
            sample_pixels = [
                (random.randint(0, height - 1), random.randint(0, width - 1))
                for _ in range(sample_size)
            ]

            stats = {"mean": [], "std": [], "min": [], "max": []}

            # Read values for each band
            for band_idx in bands:
                band_data = src.read(band_idx)

                # Collect sample values
                sample_values = [
                    band_data[y, x]
                    for y, x in sample_pixels
                    if 0 <= y < height and 0 <= x < width
                ]

                # Calculate statistics
                sample_values = np.array(sample_values)
                sample_values = sample_values[~np.isnan(sample_values)]  # Remove NaN values

                if len(sample_values) > 0:
                    mean_val = sample_values.mean()
                    std_val = sample_values.std()
                    min_val = sample_values.min()
                    max_val = sample_values.max()
                else:
                    mean_val = 0
                    std_val = 1
                    min_val = 0
                    max_val = 1

                stats["mean"].append(mean_val)
                stats["std"].append(std_val)
                stats["min"].append(min_val)
                stats["max"].append(max_val)

                print(
                    f"Band {band_idx}: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}"
                )

        return stats
    
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {"mean": [], "std": [], "min": [], "max": []}

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
        self.scalers = [
            SklearnMinMaxScaler(feature_range=self.feature_range)
            for _ in range(num_channels)
        ]

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
            raise ValueError(
                f"Image has {channels} channels, but scaler has {len(self.scalers)} scalers"
            )

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
            raise ValueError(
                f"Image has {channels} channels, but scaler has {len(self.scalers)} scalers"
            )

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
    Enhanced normalizer for Sentinel-2 imagery with TorchGeo compatibility.

    This normalizer handles the specific band structure of Sentinel-2,
    including different methods for RGB vs other bands.

    Parameters:
    -----------
    method : str
        Normalization method ('minmax', 'standard', or 'pretrained')
    rgb_indices : list
        Indices of RGB bands
    """

    def __init__(self, method="pretrained", rgb_indices=[0, 1, 2]):
        self.method = method
        self.rgb_indices = rgb_indices
        self.torchgeo_specific = False  # Additional flag for TorchGeo compatibility

        # Define band names for reference
        self.band_names = [
            "B02",
            "B03",
            "B04",  # RGB bands
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",  # Other bands
            "B01",
            "B09",
            "B10",  # 60m bands (if included)
        ]

        # For pretrained method, define normalization parameters
        if method == "pretrained":
            # ImageNet RGB stats for TorchGeo compatibility
            self.rgb_mean = [0.485, 0.456, 0.406]
            self.rgb_std = [0.229, 0.224, 0.225]
            self.other_mean = 0.5
            self.other_std = 0.5

        # For TorchGeo-specific models, adjust these values
        self.is_fitted = False
        self.scalers = None
        self.raw_scale_factor = (
            10000.0  # Scale factor for raw Sentinel-2 reflectance values
        )

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
        if self.method == "pretrained":
            # No fitting needed for pretrained method
            self.is_fitted = True
            return self

        elif image_path is not None:
            # Try to get statistics from image metadata
            try:
                with rasterio.open(image_path) as src:
                    metadata = src.tags()
                    if (
                        "STATISTICS_MEAN" in metadata
                        and "STATISTICS_STDDEV" in metadata
                    ):
                        means = metadata["STATISTICS_MEAN"].split(",")
                        stds = metadata["STATISTICS_STDDEV"].split(",")

                        if len(means) == src.count and len(stds) == src.count:
                            means = [float(m) for m in means]
                            stds = [float(s) for s in stds]

                            if self.method == "minmax":
                                # Approximate min-max from mean and std
                                mins = [m - 3 * s for m, s in zip(means, stds)]
                                maxs = [m + 3 * s for m, s in zip(means, stds)]

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
                                    fake_data = np.array([m - s, m, m + s]).reshape(
                                        -1, 1
                                    )
                                    scaler.fit(fake_data)
                                    self.scalers.append(scaler)

                            self.is_fitted = True
                            return self
            except Exception as e:
                print(f"Could not read statistics from image metadata: {e}")
                print("Calculating statistics from data instead.")

        # If we get here, calculate from data
        if self.method in ["minmax", "standard"] and images is not None:
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

                if self.method == "minmax":
                    scaler = SklearnMinMaxScaler()
                else:  # standard
                    scaler = SklearnStandardScaler()

                scaler.fit(channel_data)
                self.scalers.append(scaler)

            self.is_fitted = True
        elif self.method == "pretrained":
            self.is_fitted = True
        else:
            print("Warning: Normalizer not fitted. Will use default normalization.")
            self.is_fitted = True  # Set to True to avoid errors, but will use defaults

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
            image_np = image.cpu().numpy()
        else:
            image_np = image.copy()  # Create a copy to avoid modifying the original

        # Check if values might be raw Sentinel-2 reflectance (typically 0-10000)
        if image_np.max() > 100:
            # Apply scaling factor for TorchGeo models
            image_np = image_np / self.raw_scale_factor
            #if self.torchgeo_specific:
            #    print(f"Applied scaling factor: {self.raw_scale_factor}")

        # Create normalized output
        normalized = np.zeros_like(image_np, dtype=np.float32)

        if self.method == "pretrained":
            # Normalize RGB bands with ImageNet stats (or custom stats for TorchGeo)
            for i, idx in enumerate(self.rgb_indices):
                if idx < image_np.shape[0]:
                    normalized[idx] = (image_np[idx] - self.rgb_mean[i]) / self.rgb_std[
                        i
                    ]

            # Normalize other bands
            for i in range(image_np.shape[0]):
                if i not in self.rgb_indices:
                    # First rescale to [0-1] if needed
                    band = image_np[i]
                    if (
                        not self.torchgeo_specific
                    ):  # Skip this for TorchGeo specific models
                        if band.max() > band.min():
                            band = (band - band.min()) / (band.max() - band.min())

                    # Then normalize with mean and std
                    normalized[i] = (band - self.other_mean) / self.other_std

        elif self.scalers is not None:
            # Use fitted scalers
            for i in range(min(image_np.shape[0], len(self.scalers))):
                # Reshape channel to 2D array with samples in rows
                channel_data = image_np[i].reshape(1, -1)
                # Transform the data
                normalized_data = self.scalers[i].transform(channel_data)
                # Reshape back to original shape
                normalized[i] = normalized_data.reshape(image_np[i].shape)
        else:
            # Fallback to simple [0,1] normalization
            for i in range(image_np.shape[0]):
                band = image_np[i]
                if band.max() > band.min():
                    normalized[i] = (band - band.min()) / (band.max() - band.min())
                else:
                    normalized[i] = band  # Keep as is if constant

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


# Function to load a saved normalizer
def load_sentinel2_normalizer(normalizer_path):
    """
    Load a previously saved Sentinel2Normalizer with improved error handling.
    """
    import pickle

    try:
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)

        # Ensure TorchGeo compatibility attribute exists
        if not hasattr(normalizer, "torchgeo_specific"):
            normalizer.torchgeo_specific = False

        # Ensure is_fitted attribute exists
        if not hasattr(normalizer, "is_fitted"):
            normalizer.is_fitted = True
            
        # Ensure method is set
        if not hasattr(normalizer, "method"):
            normalizer.method = "pretrained"
            
        # Ensure RGB stats are properly set
        if not hasattr(normalizer, "rgb_mean") or not hasattr(normalizer, "rgb_std"):
            normalizer.rgb_mean = [0.485, 0.456, 0.406]
            normalizer.rgb_std = [0.229, 0.224, 0.225]
            
        # Ensure other band stats are properly set
        if not hasattr(normalizer, "other_mean") or not hasattr(normalizer, "other_std"):
            normalizer.other_mean = 0.5
            normalizer.other_std = 0.5
            
        # Ensure raw_scale_factor is set
        if not hasattr(normalizer, "raw_scale_factor"):
            normalizer.raw_scale_factor = 10000.0

        print(f"Successfully loaded normalizer with method: {normalizer.method}")
        return normalizer
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        print("Creating a default normalizer for Sentinel-2 data...")
        return Sentinel2Normalizer(method="pretrained")

# Function to normalize a batch of images
def normalize_batch(batch, normalizer=None, device=None):
    """
    Normalize a batch of images using the provided normalizer with improved error handling.

    Parameters:
    -----------
    batch : torch.Tensor
        Batch of images (B, C, H, W)
    normalizer : Sentinel2Normalizer or None
        Normalizer to use. If None, applies basic normalization
    device : torch.device or None
        Device to put the normalized batch on

    Returns:
    --------
    torch.Tensor
        Normalized batch
    """
    # Error checking - make sure we don't have NaNs or Infs in the input
    if torch.isnan(batch).any() or torch.isinf(batch).any():
        print("WARNING: Input batch contains NaN or Inf values. Applying clipping.")
        batch = torch.nan_to_num(batch, nan=0.0, posinf=10000.0, neginf=0.0)
    
    # Handle None normalizer case
    if normalizer is None:
        # Simple min-max normalization to [0,1]
        batch_min = batch.min()
        batch_max = batch.max()

        # Check if data might be raw Sentinel-2 reflectance
        if batch_max > 100:
            # Divide by typical Sentinel-2 scale
            normalized = batch / 10000.0
        elif batch_max > batch_min:
            # Standard min-max normalization with epsilon for stability
            epsilon = 1e-8
            normalized = (batch - batch_min) / max((batch_max - batch_min), epsilon)
        else:
            # Handle constant values
            normalized = torch.zeros_like(batch)
    else:
        try:
            # Use the provided normalizer
            if isinstance(batch, torch.Tensor):
                # Process each image in the batch
                normalized_list = []
                for img in batch:
                    # Apply normalizer with error checks
                    try:
                        norm_img = normalizer.transform(img)
                        normalized_list.append(norm_img)
                    except Exception as e:
                        print(f"Error in normalizer: {e}. Using fallback normalization.")
                        # Fallback to simple normalization
                        if img.max() > 100:
                            norm_img = img / 10000.0
                        else:
                            norm_img = img
                        normalized_list.append(norm_img)
                
                normalized = torch.stack([torch.as_tensor(img, dtype=torch.float32) 
                                         for img in normalized_list])
            else:
                # Handle numpy arrays
                normalized = torch.from_numpy(
                    np.stack([normalizer.transform(img) for img in batch])
                )
        except Exception as e:
            print(f"Critical error in normalization: {e}. Using safe defaults.")
            # Most robust fallback - just scale values to a safe range
            normalized = batch / 10000.0 if batch.max() > 100 else batch

    # Final safety check - catch any remaining NaNs
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        print("WARNING: Normalized result contains NaN/Inf. Using basic scaling.")
        normalized = batch / 10000.0 if batch.max() > 100 else batch / (batch.max() + 1e-8)

    # Move to device if specified
    if device is not None:
        normalized = normalized.to(device)

    return normalized