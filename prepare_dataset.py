"""
Script to prepare remapped datasets for training segmentation models with Sentinel-2 imagery.

This enhanced version includes:
1. Support for remapped class labels
2. Enhanced normalization for TorchGeo compatibility
3. Class distribution analysis
"""

import os
import argparse
import numpy as np
import json
import rasterio
import pickle

# Import custom modules
from dataset.dataset_splitter import create_patch_based_splits
from dataset.patch_dataset import create_patch_data_loaders
from dataset.augmentation import get_train_transform, get_val_transform
from dataset.normalizers import Sentinel2Normalizer, get_sentinel2_statistics


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
        default="standard",
        choices=["minmax", "standard", "pretrained"],
        help="Normalization method",
    )
    parser.add_argument(
        "--use_class_remapping", action="store_true", help="Use class remapping"
    )
    parser.add_argument(
        "--class_mapping_file",
        type=str,
        default=None,
        help="JSON file with class mapping (if using remapping)",
    )
    parser.add_argument(
        "--torchgeo_compatible",
        action="store_true",
        help="Make normalization compatible with TorchGeo models",
    )
    parser.add_argument(
        "--use_copy_paste", action="store_true", help="Use CopyPaste augmentation"
    )

    return parser.parse_args()


def analyze_class_distribution(mask_path, class_mapping=None):
    """
    Analyze the class distribution in the mask.

    Parameters:
    -----------
    mask_path : str
        Path to the mask GeoTIFF
    class_mapping : dict or None
        Class mapping dictionary if remapping is used

    Returns:
    --------
    dict
        Dictionary with class distribution information
    """
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)

    # Get unique values and their counts
    unique_values, counts = np.unique(mask_data, return_counts=True)

    # Calculate percentages
    total_pixels = mask_data.size
    percentages = (counts / total_pixels) * 100

    # Create distribution info
    distribution = {}
    for val, count, percentage in zip(unique_values, counts, percentages):
        class_label = str(val)
        if class_mapping is not None and str(val) in class_mapping:
            class_label = f"{val} → {class_mapping[str(val)]}"

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


def create_remapped_dataset(args, class_mapping=None):
    """
    Create a remapped dataset with class mapping.

    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    class_mapping : dict or None
        Class mapping dictionary for remapping

    Returns:
    --------
    dict
        Information about the created dataset
    """
    # Step 1: Calculate statistics for the input image
    print("Calculating image statistics...")
    stats = get_sentinel2_statistics(args.image_path, sample_size=2000)
    print(f"Image statistics: {stats}")

    # Analyze class distribution in the original mask
    print("\nAnalyzing class distribution in original mask...")
    class_info = analyze_class_distribution(args.mask_path, class_mapping)
    print(f"Found {class_info['num_classes']} unique classes")
    print("Class distribution:")
    for class_label, info in class_info["distribution"].items():
        print(f"  {class_label}: {info['percentage']}")

    # Step 2: Create patch-based dataset splits
    print("\nCreating patch-based dataset splits...")
    splits_info = create_patch_based_splits(
        image_path=args.image_path,
        mask_path=args.mask_path,
        validity_mask_path=args.validity_mask_path,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        overlap=args.overlap,
        stratify=True,
    )
    print(f"Dataset splits created: {splits_info}")

    # Step 3: Set up data augmentation
    print("\nSetting up data augmentation...")
    train_transform = get_train_transform(
        p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste
    )
    val_transform = get_val_transform(patch_size=args.patch_size)

    # Step 4: Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_patch_data_loaders(
        patches_dir=args.output_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=4,
    )

    # Step 5: Create and save normalizer (with TorchGeo compatibility if requested)
    print("\nCreating and saving normalizer...")
    normalizer = Sentinel2Normalizer(method=args.normalize_method)

    # Modify the normalizer for TorchGeo compatibility if requested
    if args.torchgeo_compatible:
        # TorchGeo expects data to be normalized to a specific range
        # For TorchGeo pretrained models, we'll use specific normalization values
        print("Setting up TorchGeo-compatible normalization")
        normalizer.other_mean = 0.0
        normalizer.other_std = 1.0
        # For torchgeo_specific method, we'll divide all values by 10000
        normalizer.torchgeo_specific = True

    # Fit the normalizer on the image
    normalizer.fit(None, args.image_path)

    # Save the normalizer
    normalizer_path = os.path.join(args.output_dir, "normalizer.pkl")
    with open(normalizer_path, "wb") as f:
        pickle.dump(normalizer, f)
    print(f"Normalizer saved to {normalizer_path}")

    # Step 6: Save class mapping information if applicable
    if class_mapping is not None:
        # Save the class mapping
        mapping_path = os.path.join(args.output_dir, "class_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=4)

        # Save the inverse mapping (for converting predictions back to original classes)
        inverse_mapping = {v: k for k, v in class_mapping.items()}
        inverse_path = os.path.join(args.output_dir, "inverse_class_mapping.json")
        with open(inverse_path, "w") as f:
            json.dump(inverse_mapping, f, indent=4)

        print(f"Class mappings saved to {mapping_path} and {inverse_path}")

    return {
        "normalizer_path": normalizer_path,
        "num_classes": class_info["num_classes"],
        "class_distribution": class_info["distribution"],
        "splits_info": splits_info,
    }


def main():
    """Run the dataset preparation pipeline."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("Sentinel-2 Remapped Dataset Preparation".center(80))
    print("=" * 80 + "\n")

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load class mapping if provided
    class_mapping = None
    if args.use_class_remapping and args.class_mapping_file is not None:
        if os.path.exists(args.class_mapping_file):
            with open(args.class_mapping_file, "r") as f:
                class_mapping = json.load(f)
            print(f"Loaded class mapping from {args.class_mapping_file}")
            print(f"Class mapping: {class_mapping}")
        else:
            print(f"Warning: Class mapping file {args.class_mapping_file} not found")

    # Create the remapped dataset
    dataset_info = create_remapped_dataset(args, class_mapping)

    # Save dataset information
    info_path = os.path.join(args.output_dir, "splits_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
    print(f"Dataset information saved to {info_path}")

    print("\nRemapped dataset preparation completed successfully!")
    print(f"Dataset location: {os.path.abspath(args.output_dir)}")
    print("\nYou can now train models using the command:")
    print(
        f'python train.py --data_dir "{os.path.abspath(args.output_dir)}" --model_name sentinel2_model --auto_detect_classes --use_normalizer'
    )


if __name__ == "__main__":
    main()
