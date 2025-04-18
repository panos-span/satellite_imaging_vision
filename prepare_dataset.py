"""
Script to prepare remapped datasets for training segmentation models with Sentinel-2 imagery.

This optimized version includes:
1. Robust remapping of non-consecutive class values (e.g., [0, 10, 20, 30]) to
   consecutive indices ([0, 1, 2, 3]) for efficient training
2. Enhanced normalization for TorchGeo compatibility
3. Proper handling of band statistics for Sentinel-2 data
"""

import os
import argparse
import numpy as np
import json
import rasterio
import pickle

# Import custom modules
try:
    from dataset_splitter import create_patch_based_splits
    from patch_dataset import create_patch_data_loaders
    from augmentation import get_train_transform, get_val_transform
    from normalizers import Sentinel2Normalizer
except ImportError:
    # Try alternative import paths
    try:
        from dataset.dataset_splitter import create_patch_based_splits
        from dataset.patch_dataset import create_patch_data_loaders
        from dataset.augmentation import get_train_transform, get_val_transform
        from dataset.normalizers import Sentinel2Normalizer
    except ImportError:
        print(
            "Warning: Could not import dataset modules. Some functionality might be limited."
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare remapped datasets for training"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to multiband Sentinel-2 image",
    )
    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to ground truth mask"
    )
    parser.add_argument(
        "--validity_mask_path",
        type=str,
        default=None,
        help="Path to validity mask (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="remapped_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Size of patches")
    parser.add_argument(
        "--overlap", type=int, default=32, help="Overlap between patches"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for data loaders"
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="pretrained",  # Default to pretrained for TorchGeo compatibility
        choices=["minmax", "standard", "pretrained"],
        help="Normalization method",
    )
    parser.add_argument(
        "--class_mapping_file",
        type=str,
        default=None,
        help="JSON file with class mapping (if using custom remapping)",
    )
    parser.add_argument(
        "--torchgeo_compatible",
        action="store_true",
        help="Make normalization compatible with TorchGeo models",
    )
    parser.add_argument(
        "--use_copy_paste", action="store_true", help="Use CopyPaste augmentation"
    )
    parser.add_argument(
        "--force_include_zero",
        action="store_true",
        default=True,
        help="Force inclusion of class 0 (background) in the class mapping",
    )
    parser.add_argument(
        "--disable_class_remapping",
        action="store_true",
        help="Disable automatic class remapping (use original class values)",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of samples to analyze for class detection",
    )

    return parser.parse_args()


def get_sentinel2_statistics(image_path, sample_size=2000):
    """
    Calculate statistics from a Sentinel-2 image for normalization.

    Parameters:
    -----------
    image_path : str
        Path to Sentinel-2 GeoTIFF
    sample_size : int
        Number of random pixels to sample

    Returns:
    --------
    dict
        Dictionary with statistics for each band
    """
    with rasterio.open(image_path) as src:
        # Get number of bands and dimensions
        num_bands = src.count
        height, width = src.height, src.width

        # Sample random pixels
        stats = []
        for band in range(1, num_bands + 1):
            # Generate random indices
            random_indices = np.random.choice(
                height * width, min(sample_size, height * width), replace=False
            )
            row_indices = random_indices // width
            col_indices = random_indices % width

            # Read data at sampled locations
            band_data = src.read(band)
            sampled_values = band_data[row_indices, col_indices]

            # Filter out nodata values (assuming 0 is nodata)
            valid_values = sampled_values[sampled_values > 0]
            if len(valid_values) == 0:
                valid_values = sampled_values  # Use all values if all are 0

            # Calculate statistics
            band_stats = {
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
            }
            stats.append(band_stats)

    return {f"band_{i+1}": stats[i] for i in range(num_bands)}


def find_unique_class_values(mask_path, sample_limit=None):
    """
    Find all unique class values in a mask.

    Parameters:
    -----------
    mask_path : str
        Path to the mask file
    sample_limit : int or None
        Maximum number of pixels to analyze, None for all

    Returns:
    --------
    list
        List of unique class values
    dict
        Class distribution information
    """
    with rasterio.open(mask_path) as src:
        # Get dimensions
        height, width = src.height, src.width

        if sample_limit is not None and sample_limit < height * width:
            # Sample random pixels
            flat_indices = np.random.choice(height * width, sample_limit, replace=False)
            row_indices = flat_indices // width
            col_indices = flat_indices % width

            # Read mask data at sampled locations
            mask_data = src.read(1)
            sampled_values = mask_data[row_indices, col_indices]

            # Find unique values
            unique_values, counts = np.unique(sampled_values, return_counts=True)
        else:
            # Read entire mask
            mask_data = src.read(1)

            # Find unique values
            unique_values, counts = np.unique(mask_data, return_counts=True)

    # Calculate percentages
    total_pixels = len(mask_data.flatten()) if sample_limit is None else sample_limit
    percentages = (counts / total_pixels) * 100

    # Create class distribution dictionary
    distribution = {}
    for value, count, percentage in zip(unique_values, counts, percentages):
        distribution[int(value)] = {
            "count": int(count),
            "percentage": f"{percentage:.2f}%",
        }

    # Check if class values are consecutive
    sorted_values = sorted(unique_values)
    is_consecutive = len(sorted_values) == (
        sorted_values[-1] - sorted_values[0] + 1
    ) and np.all(np.diff(sorted_values) == 1)

    # Check if class values are sparse
    is_sparse = not is_consecutive and len(sorted_values) < sorted_values[-1] + 1

    # Check if remapping is needed
    need_remapping = is_sparse or not (0 in sorted_values and sorted_values[0] == 0)

    return {
        "unique_values": sorted_values.tolist(),
        "distribution": distribution,
        "num_classes": len(sorted_values),
        "is_consecutive": is_consecutive,
        "is_sparse": is_sparse,
        "need_remapping": need_remapping,
    }


def create_class_mapping(unique_values, force_include_zero=True):
    """
    Create a mapping from original class values to consecutive indices.

    Parameters:
    -----------
    unique_values : list
        List of unique class values
    force_include_zero : bool
        Whether to force class 0 to be included

    Returns:
    --------
    dict
        Dictionary mapping original values to consecutive indices
    """
    # Sort unique values
    sorted_values = sorted(unique_values)

    # Check if zero should be included
    if force_include_zero and 0 not in sorted_values:
        sorted_values = [0] + sorted_values
        print("Forcing inclusion of class 0 (background) in the mapping")

    # Create mapping from original values to consecutive indices
    class_mapping = {str(orig): idx for idx, orig in enumerate(sorted_values)}

    # Also create reverse mapping for decoding predictions
    reverse_mapping = {str(idx): int(orig) for orig, idx in class_mapping.items()}

    return {
        "class_mapping": class_mapping,
        "reverse_mapping": reverse_mapping,
        "num_classes": len(class_mapping),
    }


def remap_mask(mask, class_mapping):
    """
    Remap class values in a mask according to the provided mapping.

    Parameters:
    -----------
    mask : numpy.ndarray
        Input mask with original class values
    class_mapping : dict
        Dictionary mapping original class values (as strings) to new values

    Returns:
    --------
    numpy.ndarray
        Mask with remapped class values
    """
    if class_mapping is None:
        return mask

    # Create a remapped copy of the mask
    remapped_mask = np.zeros_like(mask)

    # Apply mapping - convert keys to integers since they're stored as strings in the mapping
    for orig_class_str, new_class in class_mapping.items():
        orig_class = int(orig_class_str)
        remapped_mask[mask == orig_class] = new_class

    return remapped_mask


def analyze_class_distribution(mask_data, class_mapping=None):
    """
    Analyze the class distribution in the mask.

    Parameters:
    -----------
    mask_data : numpy.ndarray
        Mask data
    class_mapping : dict or None
        Class mapping dictionary if remapping is used

    Returns:
    --------
    dict
        Dictionary with class distribution information
    """
    # Get unique values and their counts
    unique_values, counts = np.unique(mask_data, return_counts=True)

    # Calculate percentages
    total_pixels = mask_data.size
    percentages = (counts / total_pixels) * 100

    # Create distribution info
    distribution = {}
    for val, count, percentage in zip(unique_values, counts, percentages):
        class_label = str(int(val))
        if class_mapping is not None and class_label in class_mapping:
            mapped_value = class_mapping[class_label]
            class_label = f"{val} → {mapped_value}"

        distribution[class_label] = {
            "count": int(count),
            "percentage": f"{percentage:.2f}%",
        }

    # Get number of classes
    num_classes = len(unique_values)

    return {
        "unique_values": unique_values.tolist(),
        "distribution": distribution,
        "num_classes": num_classes,
    }


def save_remapped_mask(mask_path, remapped_mask, output_path):
    """
    Save a remapped mask, preserving the original metadata.

    Parameters:
    -----------
    mask_path : str
        Path to the original mask
    remapped_mask : numpy.ndarray
        Remapped mask data
    output_path : str
        Path to save the remapped mask
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read metadata from original mask
    with rasterio.open(mask_path) as src:
        profile = src.profile.copy()

    # Save remapped mask with original metadata
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(remapped_mask, 1)


def create_remapped_dataset(args):
    """
    Create a remapped dataset with proper class mapping.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    dict
        Information about the created dataset
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Calculate statistics for the input image
    print("Calculating image statistics...")
    stats = get_sentinel2_statistics(args.image_path, sample_size=2000)
    print(f"Image statistics: {stats}")

    # Step 2: Detect classes and create mapping
    print("\nDetecting unique class values in mask...")
    class_info = find_unique_class_values(args.mask_path, args.sample_limit)

    print(
        f"Found {class_info['num_classes']} unique class values: {class_info['unique_values']}"
    )
    print("\nClass distribution:")
    for val, info in class_info["distribution"].items():
        print(f"  Class {val}: {info['percentage']}")

    # Load custom class mapping if provided
    custom_mapping = None
    if args.class_mapping_file is not None and os.path.exists(args.class_mapping_file):
        with open(args.class_mapping_file, "r") as f:
            custom_mapping = json.load(f)
        print(f"Loaded custom class mapping from {args.class_mapping_file}")

    # Determine if we need to remap and create mapping
    mapping_info = None

    if args.disable_class_remapping:
        print("\nClass remapping disabled - using original class values")
        need_remapping = False
        class_mapping = None
        remapped_mask_path = args.mask_path
    else:
        # Check if we need to remap
        if class_info["need_remapping"]:
            print(
                "\nDetected non-consecutive class values - creating mapping to consecutive indices"
            )
            if custom_mapping is not None:
                # Use custom mapping
                mapping_info = {
                    "class_mapping": custom_mapping,
                    "reverse_mapping": {
                        str(v): int(k) for k, v in custom_mapping.items()
                    },
                    "num_classes": len(custom_mapping),
                }
                print("Using custom class mapping")
            else:
                # Generate automatic mapping
                mapping_info = create_class_mapping(
                    class_info["unique_values"],
                    force_include_zero=args.force_include_zero,
                )

            # Print the mapping
            print("\nClass mapping:")
            for orig, new in mapping_info["class_mapping"].items():
                print(f"  Original class {orig} → New class {new}")

            need_remapping = True
            class_mapping = mapping_info["class_mapping"]
        else:
            print("\nClass values are already consecutive - no remapping needed")
            need_remapping = False
            class_mapping = None
            remapped_mask_path = args.mask_path

    # Step 3: Create a remapped mask if needed
    if need_remapping:
        print("\nCreating remapped mask...")
        remapped_mask_path = os.path.join(args.output_dir, "remapped_mask.tif")

        # Read original mask
        with rasterio.open(args.mask_path) as src:
            original_mask = src.read(1)

            # Remap the mask
            remapped_mask = remap_mask(original_mask, class_mapping)

            # Save remapped mask
            save_remapped_mask(args.mask_path, remapped_mask, remapped_mask_path)

        print(f"Remapped mask saved to {remapped_mask_path}")

        # Analyze class distribution in remapped mask
        print("\nClass distribution after remapping:")
        remapped_class_info = analyze_class_distribution(remapped_mask)

        for val, info in remapped_class_info["distribution"].items():
            print(f"  Class {val}: {info['percentage']}")

        # Save mapping files
        mapping_path = os.path.join(args.output_dir, "class_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(mapping_info["class_mapping"], f, indent=4)

        inverse_path = os.path.join(args.output_dir, "inverse_class_mapping.json")
        with open(inverse_path, "w") as f:
            json.dump(mapping_info["reverse_mapping"], f, indent=4)

        print(f"Class mapping saved to {mapping_path}")
        print(f"Inverse mapping saved to {inverse_path}")

    # Step 4: Create patch-based dataset splits
    print("\nCreating patch-based dataset splits...")
    splits_info = create_patch_based_splits(
        image_path=args.image_path,
        mask_path=remapped_mask_path,  # Use remapped mask if created
        validity_mask_path=args.validity_mask_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        overlap=args.overlap,
        stratify=True,
    )

    # Update splits info with class information for the model
    if need_remapping and mapping_info is not None:
        splits_info["num_classes"] = mapping_info["num_classes"]
        splits_info["class_values"] = list(range(mapping_info["num_classes"]))
        splits_info["original_class_values"] = class_info["unique_values"]
    else:
        splits_info["num_classes"] = class_info["num_classes"]
        splits_info["class_values"] = class_info["unique_values"]

    print(
        f"Dataset splits created with {splits_info['train_patches']} training patches"
    )

    # Step 5: Set up data augmentation
    print("\nSetting up data augmentation...")
    # Get RGB indices (assuming bands 2, 3, 4 are RGB for Sentinel-2)
    rgb_indices = [
        0,
        1,
        2,
    ]  # Default RGB indices (Sentinel-2 convention: B02=Blue, B03=Green, B04=Red)

    train_transform = get_train_transform(
        p=0.5,
        patch_size=args.patch_size,
        use_copy_paste=args.use_copy_paste,
        rgb_indices=rgb_indices,
    )
    val_transform = get_val_transform(patch_size=args.patch_size)

    # Step 6: Create data loaders (for validation)
    print("\nCreating data loaders for validation...")
    try:
        data_loaders = create_patch_data_loaders(
            patches_dir=args.output_dir,
            batch_size=args.batch_size,
            train_transform=train_transform,
            val_transform=val_transform,
            num_workers=4,
        )

        print(f"Created data loaders with:")
        print(f"  {len(data_loaders['train'].dataset)} training samples")
        print(f"  {len(data_loaders['val'].dataset)} validation samples")
        print(f"  {len(data_loaders['test'].dataset)} test samples")
    except Exception as e:
        print(f"Warning: Could not create data loaders for validation: {e}")
        print(
            "This is expected if the modules are not available, but the dataset is still created."
        )

    # Step 7: Create and save normalizer optimized for TorchGeo
    print("\nCreating and saving normalizer for TorchGeo compatibility...")
    try:
        # Always use pretrained method for TorchGeo models
        normalize_method = (
            "pretrained" if args.torchgeo_compatible else args.normalize_method
        )
        normalizer = Sentinel2Normalizer(
            method=normalize_method, rgb_indices=rgb_indices
        )

        # Optimize normalizer for TorchGeo compatibility
        if args.torchgeo_compatible:
            print("Configuring normalizer specifically for TorchGeo ResNet models...")
            normalizer.torchgeo_specific = True
            normalizer.raw_scale_factor = (
                10000.0  # Standard scale factor for Sentinel-2 reflectance values
            )

            # Using ImageNet stats for RGB bands and 0.5 mean/std for other bands
            # This matches what TorchGeo expects for transfer learning with pretrained ResNet
            normalizer.rgb_mean = [0.485, 0.456, 0.406]
            normalizer.rgb_std = [0.229, 0.224, 0.225]
            normalizer.other_mean = 0.5
            normalizer.other_std = 0.5

        # Fit the normalizer on the image
        normalizer.fit(None, args.image_path)

        # Save the normalizer
        normalizer_path = os.path.join(args.output_dir, "normalizer.pkl")
        with open(normalizer_path, "wb") as f:
            pickle.dump(normalizer, f)
        print(f"Normalizer saved to {normalizer_path}")
    except Exception as e:
        print(f"Warning: Could not create normalizer: {e}")
        print("This might happen if the Sentinel2Normalizer class is not available.")

    # Step 8: Create model class configuration for training
    model_class_config = {
        "num_classes": splits_info["num_classes"],
        "class_values": splits_info["class_values"],
        "ignore_index": -1,  # Default value for pixels to ignore during training
    }

    if need_remapping and mapping_info is not None:
        model_class_config["class_mapping"] = mapping_info["class_mapping"]
        model_class_config["reverse_mapping"] = mapping_info["reverse_mapping"]

    model_class_path = os.path.join(args.output_dir, "model_class_config.json")
    with open(model_class_path, "w") as f:
        json.dump(model_class_config, f, indent=4)

    print(f"Model class configuration saved to {model_class_path}")

    # Return dataset information
    return {
        "num_classes": splits_info["num_classes"],
        "class_values": splits_info["class_values"],
        "original_class_values": class_info["unique_values"],
        "class_mapping": (
            mapping_info["class_mapping"] if mapping_info is not None else None
        ),
        "class_distribution": class_info["distribution"],
        "remapped": need_remapping,
        "splits_info": splits_info,
    }


def main():
    """Run the dataset preparation pipeline."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("Sentinel-2 Remapped Dataset Preparation".center(80))
    print("=" * 80 + "\n")

    # Check input files
    for path, name in [(args.image_path, "image"), (args.mask_path, "mask")]:
        if not os.path.exists(path):
            print(f"Error: {name} file '{path}' does not exist")
            return

    if args.validity_mask_path and not os.path.exists(args.validity_mask_path):
        print(
            f"Warning: Validity mask '{args.validity_mask_path}' does not exist - proceeding without it"
        )
        args.validity_mask_path = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Create the remapped dataset
    dataset_info = create_remapped_dataset(args)

    # Save dataset information
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        json_info = {}
        for k, v in dataset_info.items():
            if isinstance(v, np.ndarray):
                json_info[k] = v.tolist()
            elif isinstance(v, dict):
                # Handle nested dicts
                json_info[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        json_info[k][kk] = vv.tolist()
                    else:
                        json_info[k][kk] = vv
            else:
                json_info[k] = v

        json.dump(json_info, f, indent=4)

    print(f"Dataset information saved to {info_path}")

    print("\n" + "=" * 80)
    print("Remapped dataset preparation completed successfully!".center(80))
    print("=" * 80 + "\n")

    print(f"Dataset location: {os.path.abspath(args.output_dir)}")
    print(f"Number of classes: {dataset_info['num_classes']}")

    if dataset_info.get("remapped", False):
        print(
            "Class values were remapped to consecutive indices for efficient training"
        )
    else:
        print("Class values were already consecutive - no remapping was needed")

    print("\nYou can now train your TorchGeo UNet model using the command:")
    print(
        f'python train.py --data_dir "{os.path.abspath(args.output_dir)}" --model_name sentinel2_unet --num_classes {dataset_info["num_classes"]} --use_normalizer --torchgeo_pretrained'
    )


if __name__ == "__main__":
    main()
