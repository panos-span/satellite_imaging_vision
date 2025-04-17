"""
Main script for running the entire data preparation pipeline for Sentinel-2 land cover classification.

This script orchestrates:
1. Data checking and verification
2. Sentinel-2 data processing (pansharpening, alignment, etc.)
3. Dataset creation and organization

Usage:
    python run_data_preparation.py --sentinel_dir data/sentinel_data --ground_truth data/ground_truth.tif --output_dir output
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data preparation modules
from data_preparation.check_data import main as check_data_main
from data_preparation.sentinel_processor import SentinelProcessor
from data_preparation.visualization import visualize_rgb, visualize_false_color
from data_preparation.geospatial_utils import (
    get_ground_truth_info,
    align_raster_to_reference,
    merge_rasters,
    create_multiband_stack,
)


def setup_logging(output_dir):
    """Set up logging to file and console."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "data_preparation.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger("data_preparation")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Data Preparation for Land Cover Classification"
    )

    # Input/output directories
    parser.add_argument(
        "--sentinel_dir",
        type=str,
        required=True,
        help="Directory containing Sentinel-2 data (zip files or SAFE directories)",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth GeoTIFF file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory for output files"
    )

    # Processing options
    parser.add_argument(
        "--pansharpening",
        type=str,
        default="brovey",
        choices=["simple", "brovey", "hpf"],
        help="Pansharpening method to use",
    )
    parser.add_argument(
        "--validity_masks", action="store_true", help="Create validity masks"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )

    # Prediction area (optional)
    parser.add_argument(
        "--predict_dir",
        type=str,
        default=None,
        help="Directory containing Sentinel-2 data for prediction area",
    )
    parser.add_argument(
        "--predict_ground_truth",
        type=str,
        default=None,
        help="Path to ground truth for prediction area (optional)",
    )

    return parser.parse_args()


def main():
    """Run the data preparation pipeline."""
    args = parse_args()

    # Delete any existing output directory
    if os.path.exists(args.output_dir):
        import shutil

        shutil.rmtree(args.output_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_dir)

    # Print banner
    logger.info("=" * 80)
    logger.info("Sentinel-2 Data Preparation Pipeline".center(80))
    logger.info("=" * 80)

    # Step 1: Check data
    logger.info("\nStep 1: Checking Sentinel-2 data...")
    # Run check_data_main without arguments first
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [
            sys.argv[0],
            "--sentinel_dir",
            args.sentinel_dir,
            "--ground_truth",
            args.ground_truth,
            "--output_dir",
            args.output_dir,
        ]
        valid_safe_dirs = check_data_main()
    finally:
        sys.argv = original_argv

    if not valid_safe_dirs:
        logger.error("No valid Sentinel-2 data found. Exiting.")
        sys.exit(1)

    # Step 1.5: Get ground truth information
    logger.info("\nExtracting ground truth information...")
    ground_truth_info = get_ground_truth_info(args.ground_truth)
    logger.info(f"Ground truth CRS: {ground_truth_info['crs']}")
    logger.info(f"Ground truth shape: {ground_truth_info['shape']}")
    if "classes" in ground_truth_info:
        logger.info(f"Ground truth classes: {ground_truth_info['classes']}")
    logger.info(f"Ground truth unique values: {ground_truth_info['unique_values']}")

    # Step 2: Process Sentinel-2 data
    logger.info("\nStep 2: Processing Sentinel-2 data...")
    processor = SentinelProcessor(
        output_base_dir=os.path.join(args.output_dir, "processed_data")
    )

    # Process training area
    logger.info("Processing training area...")
    train_dataset_info = processor.process_all(
        valid_safe_dirs,
        args.ground_truth,
        pansharpening_method=args.pansharpening,
        create_validity_masks=args.validity_masks,
    )

    # Process prediction area if specified
    if args.predict_dir and (
        args.predict_ground_truth or Path(args.predict_dir).is_dir()
    ):
        logger.info("\nProcessing prediction area...")

        # First check the prediction data
        sys.argv = [
            sys.argv[0],
            "--sentinel_dir",
            args.predict_dir,
            "--ground_truth",
            args.predict_ground_truth or args.ground_truth,
            "--output_dir",
            args.output_dir,
        ]
        try:
            predict_safe_dirs = check_data_main()
        finally:
            sys.argv = original_argv

        if predict_safe_dirs:
            predict_processor = SentinelProcessor(
                output_base_dir=os.path.join(args.output_dir, "predict_processed")
            )

            predict_dataset_info = predict_processor.process_all(
                predict_safe_dirs,
                args.predict_ground_truth
                or args.ground_truth,  # Use prediction ground truth if available, otherwise training
                pansharpening_method=args.pansharpening,
                create_validity_masks=args.validity_masks,
            )

            logger.info(
                f"Prediction dataset created: {predict_dataset_info['image_path']}"
            )

            # Create manual alignment between training and prediction areas if needed
            if (
                args.predict_ground_truth is None
                and "image_path" in predict_dataset_info
            ):
                logger.info("\nAligning prediction data with training data...")
                prediction_aligned_path = os.path.join(
                    args.output_dir, "predict_aligned", "aligned_image.tif"
                )
                os.makedirs(os.path.dirname(prediction_aligned_path), exist_ok=True)

                # Align prediction image to training area
                align_raster_to_reference(
                    predict_dataset_info["image_path"],
                    ground_truth_info,
                    prediction_aligned_path,
                )
                logger.info(
                    f"Aligned prediction image saved to: {prediction_aligned_path}"
                )

                # Update the image path in the dataset info
                predict_dataset_info["aligned_image_path"] = prediction_aligned_path
        else:
            logger.warning("No valid Sentinel-2 data found for prediction area.")

    # Step 3: Create visualizations
    if args.visualize:
        logger.info("\nStep 3: Creating visualizations...")

        # Create output directory for visualizations
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # RGB visualization
        logger.info("Creating RGB visualization...")
        visualize_rgb(
            train_dataset_info["image_path"],
            output_path=os.path.join(vis_dir, "rgb_visualization.png"),
        )

        # False color visualizations
        for combo in ["vegetation", "urban", "geology"]:
            logger.info(f"Creating {combo} false color visualization...")
            visualize_false_color(
                train_dataset_info["image_path"],
                output_path=os.path.join(vis_dir, f"false_color_{combo}.png"),
                combination=combo,
            )

        # If we have prediction data, create comparison visualizations
        if (
            args.predict_dir
            and "predict_dataset_info" in locals()
            and "image_path" in predict_dataset_info
        ):
            logger.info("Creating prediction area visualizations...")

            # RGB for prediction area
            visualize_rgb(
                predict_dataset_info["image_path"],
                output_path=os.path.join(vis_dir, "predict_rgb.png"),
            )

            # If we have aligned prediction data, create that visualization too
            if "aligned_image_path" in predict_dataset_info:
                visualize_rgb(
                    predict_dataset_info["aligned_image_path"],
                    output_path=os.path.join(vis_dir, "predict_aligned_rgb.png"),
                )

    # Step 4: Create a multiband stack of selected bands for analysis if not already done
    if not os.path.exists(
        os.path.join(args.output_dir, "processed_data", "combined_bands.tif")
    ):
        logger.info("\nStep 4: Creating multiband stack of selected bands...")

        # Define bands of interest (e.g., RGB + NIR)
        bands_of_interest = ["B02", "B03", "B04", "B08"]

        # Get paths to merged bands
        band_paths = {}
        merged_band_paths = []
        for band in bands_of_interest:
            path = os.path.join(
                args.output_dir, "processed_data", "merged", f"{band}_merged.tif"
            )
            if os.path.exists(path):
                band_paths[band] = path
                merged_band_paths.append(path)

        if band_paths:
            # First merge any unmerged bands if needed
            for band in bands_of_interest:
                aligned_paths = []
                for safe_dir in valid_safe_dirs:
                    safe_id = os.path.basename(safe_dir).split("_")[
                        5
                    ]  # Extract tile ID
                    aligned_path = os.path.join(
                        args.output_dir,
                        "processed_data",
                        f"aligned_{safe_id}",
                        f"{band}_aligned.tif",
                    )
                    if os.path.exists(aligned_path):
                        aligned_paths.append(aligned_path)

                if aligned_paths and len(aligned_paths) > 1:
                    # Check if merged band already exists, if not create it
                    merged_path = os.path.join(
                        args.output_dir,
                        "processed_data",
                        "merged",
                        f"{band}_merged.tif",
                    )
                    if not os.path.exists(merged_path):
                        logger.info(
                            f"Merging {len(aligned_paths)} aligned images for band {band}..."
                        )
                        merged_path = merge_rasters(
                            aligned_paths,
                            merged_path,
                            reference_bounds=ground_truth_info["bounds"],
                            reference_crs=ground_truth_info["crs"],
                        )
                        logger.info(f"Merged band saved to: {merged_path}")

                    if (
                        os.path.exists(merged_path)
                        and merged_path not in merged_band_paths
                    ):
                        merged_band_paths.append(merged_path)
                        band_paths[band] = merged_path

            # Create multiband stack
            multiband_path = os.path.join(
                args.output_dir, "processed_data", "combined_bands.tif"
            )
            stack_path, band_order = create_multiband_stack(band_paths, multiband_path)
            logger.info(f"Multiband stack created: {stack_path}")
            logger.info(f"Band order: {band_order}")

            # Also create a visualization of the stack
            if args.visualize:
                vis_dir = os.path.join(args.output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)

                visualize_rgb(
                    stack_path, output_path=os.path.join(vis_dir, "combined_rgb.png")
                )

    # Step 5: Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Data Preparation Complete".center(80))
    logger.info("=" * 80)
    logger.info(f"Training dataset: {train_dataset_info['image_path']}")
    logger.info(f"Training masks: {train_dataset_info['mask_path']}")

    if args.validity_masks and train_dataset_info.get("validity_mask_path"):
        logger.info(f"Validity mask: {train_dataset_info['validity_mask_path']}")

    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Create training dataset splits using dataset/dataset_splitter.py")
    logger.info("2. Train the model using main.py")

    return train_dataset_info


if __name__ == "__main__":
    main()
