"""
Script to prepare datasets for training segmentation models with Sentinel-2 imagery.

This script demonstrates the complete data preparation workflow:
1. Loading data
2. Splitting into train/val/test
3. Creating patch-based datasets
4. Normalizing data
5. Setting up data augmentation
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from dataset.dataset_splitter import create_patch_based_splits
from dataset.patch_dataset import create_patch_data_loaders
from dataset.augmentation import get_train_transform, get_val_transform
from dataset.normalizers import Sentinel2Normalizer, get_sentinel2_statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for training")

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
        default="datasets",
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
        default="pretrained",
        choices=["minmax", "standard", "pretrained"],
        help="Normalization method",
    )
    parser.add_argument(
        "--use_copy_paste", action="store_true", help="Use CopyPaste augmentation"
    )

    return parser.parse_args()


def main():
    """Run the dataset preparation pipeline."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("Sentinel-2 Dataset Preparation".center(80))
    print("=" * 80 + "\n")

    # Step 1: Calculate statistics for the input image
    print("Calculating image statistics...")
    stats = get_sentinel2_statistics(args.image_path, sample_size=2000)
    print(f"Image statistics: {stats}")

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

    if args.use_copy_paste:
        print("CopyPaste augmentation enabled")

    # Step 4: Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_patch_data_loaders(
        patches_dir=args.output_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=4,
    )

    # Step 5: Test data loaders and display sample batch
    print("\nTesting data loaders...")

    # Get a batch from the training loader
    train_batch = next(iter(data_loaders["train"]))
    images, masks = train_batch

    print(f"Batch shape: images {images.shape}, masks {masks.shape}")

    # Display a sample from the batch
    # Replace the visualization code in prepare_dataset.py with this safer version:

    # Display a sample from the batch
    plt.figure(figsize=(15, 8))

    # Get a sample from the batch
    sample_idx = 0
    sample_img = images[sample_idx].permute(1, 2, 0).numpy()  # CHW to HWC
    sample_mask = masks[sample_idx].numpy()

    # Normalize RGB channels for display (with safety checks)
    rgb_img = sample_img[:, :, :3].copy()  # Assuming first 3 channels are RGB
    for i in range(3):
        rgb_img_ch = rgb_img[:, :, i]
        p2, p98 = np.percentile(rgb_img_ch, (2, 98))

        # Check to avoid division by zero
        if p98 > p2:
            rgb_img[:, :, i] = np.clip((rgb_img_ch - p2) / (p98 - p2), 0, 1)
        else:
            # If there's not enough range, just normalize to [0,1]
            min_val = rgb_img_ch.min()
            max_val = rgb_img_ch.max()
            if max_val > min_val:
                rgb_img[:, :, i] = (rgb_img_ch - min_val) / (max_val - min_val)
            else:
                rgb_img[:, :, i] = 0  # Set to zero if constant value

    # Display RGB image and mask
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Sample RGB Image (Augmented)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask, cmap="viridis")
    plt.title("Sample Mask")
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "samples", "sample_batch.png"), dpi=300)
    plt.close()

    print(
        f"Sample visualization saved to {os.path.join(args.output_dir, 'samples', 'sample_batch.png')}"
    )

    # Step 6: Save normalizer for later use
    normalizer = Sentinel2Normalizer(method=args.normalize_method)
    normalizer.fit(None, args.image_path)

    import pickle

    normalizer_path = os.path.join(args.output_dir, "normalizer.pkl")
    with open(normalizer_path, "wb") as f:
        pickle.dump(normalizer, f)

    print(f"Normalizer saved to {normalizer_path}")
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
