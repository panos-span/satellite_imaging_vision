
"""
Script to run hyperparameter tuning experiments for the Sentinel-2 U-Net model.

This script performs multiple training runs with different hyperparameters to find 
the optimal configuration for the segmentation model. It uses Weights & Biases
for experiment tracking and torchmetrics for evaluation metrics.
"""

import os
import argparse
import wandb
from train import run_hyperparameter_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
    
    # Add data validation options
    parser.add_argument("--validate_dataset", action="store_true", 
                        help="Validate dataset for invalid class indices")
    parser.add_argument("--ignore_index", type=int, default=-100, 
                        help="Index to ignore in loss calculation")
    parser.add_argument("--class_safety_margin", type=int, default=2, 
                        help="Add safety margin to auto-detected number of classes")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, 
                        help="Maximum number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=5, 
                        help="Number of epochs without improvement to trigger early stopping")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing the dataset patches")
    parser.add_argument("--patch_size", type=int, default=256, 
                        help="Size of input patches")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of worker threads for data loading")
    parser.add_argument("--use_copy_paste", action="store_true", 
                        help="Use CopyPaste augmentation technique")
    
    # Output parameters
    parser.add_argument("--save_dir", type=str, default="experiment_results", 
                        help="Directory to save experiment results")
    parser.add_argument("--model_name", type=str, default="unet_sentinel2", 
                        help="Base name for the model files")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="sentinel2-segmentation", 
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_group", type=str, default="hyperparameter-sweep", 
                        help="Weights & Biases experiment group")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Add auto-detect classes argument
    parser.add_argument("--auto_detect_classes", action="store_true", 
                        help="Auto-detect number of classes from dataset")
    
    # Model parameters for class detection
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="Number of classes (auto-detected if not specified)")    # Run the experiment
    
    args = parser.parse_args()
    
    # Make sure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Auto-detect classes if not specified
    if args.num_classes is None or args.auto_detect_classes:
        # We'll add this parameter to pass it to the train function
        args.auto_detect_classes = True
    
    # Initialize Weights & Biases for the hyperparameter sweep
    if args.wandb_project:
        print(f"Initializing Weights & Biases project: {args.wandb_project}")
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            config={
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "encoder_type": args.encoder_type,
                "skip_connections": args.skip_connections,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "early_stopping": args.early_stopping,
                "dataset": os.path.basename(args.data_dir),
                "experiment_type": "hyperparameter_sweep",
            },
            job_type="sweep-control"
        )
        
        # Log the different hyperparameter combinations
        wandb.run.summary["sweep_configs"] = [
            {"name": "lr_1e-3", "learning_rate": 1e-3, "optimizer": "adam", "scheduler": None},
            {"name": "lr_1e-4", "learning_rate": 1e-4, "optimizer": "adam", "scheduler": None},
            {"name": "lr_1e-5", "learning_rate": 1e-5, "optimizer": "adam", "scheduler": None},
            {"name": "sgd", "learning_rate": 1e-2, "optimizer": "sgd", "momentum": 0.9, "scheduler": None},
            {"name": "cosine", "learning_rate": 1e-3, "optimizer": "adam", "scheduler": "cosine"},
        ]
    
    # Force experiment mode
    args.run_experiment = True
    
    print("Starting hyperparameter experiments...")
    best_config = run_hyperparameter_experiment(args)
    
    print("\nExperiment complete!")
    print(f"Best configuration: {best_config}")

    # Log final best config to wandb
    if args.wandb_project and wandb.run is not None:
        wandb.run.summary["final_best_config"] = best_config
        wandb.finish()