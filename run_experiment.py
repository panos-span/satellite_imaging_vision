"""
Optimized script for hyperparameter tuning of Sentinel-2 U-Net segmentation model.

This script runs comprehensive hyperparameter experiments testing various
configurations of transfer learning, learning rates, optimizers, and schedulers.
The experiments are tracked with Weights & Biases for easy comparison.
"""

import os
import argparse
import json
import wandb
from train import run_hyperparameter_experiment
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optimized hyperparameter experiments"
    )

    # Model architecture parameters
    parser.add_argument(
        "--in_channels",
        type=int,
        default=13,
        help="Number of input channels (13 for Sentinel-2)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected if not specified)",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="best_performance",
        choices=["best_performance", "lightweight", "resnet50", "resnet18"],
        help="Type of encoder backbone to use",
    )
    parser.add_argument(
        "--skip_connections",
        type=int,
        default=4,
        help="Number of skip connections (max 4 for ResNet50)",
    )
    parser.add_argument(
        "--use_batchnorm",
        action="store_true",
        default=True,
        help="Use batch normalization in the model",
    )

    # Transfer learning parameters
    parser.add_argument(
        "--transfer_learning_mode",
        type=str,
        default="freeze_backbone",
        choices=["freeze_backbone", "progressive_unfreeze", "fine_tune_all"],
        help="Transfer learning strategy to use",
    )
    parser.add_argument(
        "--test_encoder_lr_factors",
        action="store_true",
        default=True,
        help="Test different encoder learning rate factors",
    )

    # Dataset validation options
    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Validate dataset for invalid class indices",
    )
    #parser.add_argument(
    #    "--ignore_index",
    #    type=int,
    #    default=-100,
    #    help="Index to ignore in loss calculation",
    #)
    #parser.add_argument(
    #    "--class_safety_margin",
    #    type=int,
    #    default=2,
    #    help="Add safety margin to auto-detected number of classes",
    #)
    parser.add_argument(
        "--auto_detect_classes",
        action="store_true",
        default=True,
        help="Auto-detect number of classes from dataset",
    )
    parser.add_argument(
        "--use_normalizer", 
        action="store_true", 
        default=True,  # Default to True for experiments
        help="Use the saved Sentinel-2 normalizer from dataset directory"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
        help="Number of epochs without improvement to trigger early stopping",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training",
    )

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset patches",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Size of input patches"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--use_copy_paste",
        action="store_true",
        help="Use CopyPaste augmentation technique",
    )

    # Output parameters
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiment_results",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet_sentinel2",
        help="Base name for the model files",
    )

    # Experiment customization
    parser.add_argument(
        "--custom_experiment",
        action="store_true",
        help="Use custom experiment configurations from JSON file",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="experiment_configs.json",
        help="JSON file with custom experiment configurations",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=5,
        help="Number of experiments to run when using default configurations",
    )

    # Wandb parameters
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sentinel2-segmentation",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="hyperparameter-sweep",
        help="Weights & Biases experiment group",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="transfer-learning,unet",
        help="Comma-separated tags for wandb",
    )

    # Other parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Make sure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Force experiment mode
    args.run_experiment = True

    # Define the experiment configurations
    if args.custom_experiment and os.path.exists(args.experiment_config):
        # Load custom experiment configurations from JSON file
        with open(args.experiment_config, "r") as f:
            experiment_configs = json.load(f)
        print(f"Loaded {len(experiment_configs)} custom experiment configurations")
    else:
        # Define default experiment configurations
        experiment_configs = []

        # Base learning rates to test
        learning_rates = [1e-3, 1e-4]

        # Encoder learning rate factors to test if requested
        encoder_lr_factors = [0.0, 0.1, 0.5] if args.test_encoder_lr_factors else [0.1]

        # Add progressive unfreezing configuration
        if args.transfer_learning_mode == "progressive_unfreeze":
            for lr in learning_rates:
                experiment_configs.append(
                    {
                        "name": f"progressive_lr_{lr}",
                        "learning_rate": lr,
                        "encoder_lr_factor": 0.1,
                        "optimizer": "adam",
                        "scheduler": "onecycle",
                        "freeze_backbone": True,
                        "progressive_unfreeze": True,
                        "weight_decay": 1e-4,
                    }
                )

        # Add frozen backbone configurations
        if (
            args.transfer_learning_mode == "freeze_backbone"
            or args.transfer_learning_mode == "progressive_unfreeze"
        ):
            for lr in learning_rates:
                for factor in encoder_lr_factors:
                    for scheduler in ["onecycle", "cosine"]:
                        experiment_configs.append(
                            {
                                "name": f"frozen_lr_{lr}_factor_{factor}_sched_{scheduler}",
                                "learning_rate": lr,
                                "encoder_lr_factor": factor,
                                "optimizer": "adam",
                                "scheduler": scheduler,
                                "freeze_backbone": True,
                                "progressive_unfreeze": False,
                                "weight_decay": 1e-4,
                            }
                        )

        # Add fine-tuning configurations
        if args.transfer_learning_mode == "fine_tune_all":
            for lr in learning_rates:
                for scheduler in ["onecycle", "cosine"]:
                    experiment_configs.append(
                        {
                            "name": f"finetune_lr_{lr}_sched_{scheduler}",
                            "learning_rate": lr,
                            "encoder_lr_factor": 0.1,  # Still use lower LR for encoder
                            "optimizer": "adam",
                            "scheduler": scheduler,
                            "freeze_backbone": False,
                            "progressive_unfreeze": False,
                            "weight_decay": 1e-4,
                        }
                    )

        # Add SGD optimization
        experiment_configs.append(
            {
                "name": "sgd_momentum",
                "learning_rate": 1e-2,
                "encoder_lr_factor": 0.1,
                "optimizer": "sgd",
                "momentum": 0.9,
                "scheduler": "onecycle",
                "freeze_backbone": True,
                "progressive_unfreeze": False,
                "weight_decay": 1e-4,
            }
        )

        # Limit to specified number of experiments if needed
        if len(experiment_configs) > args.num_experiments:
            experiment_configs = experiment_configs[: args.num_experiments]

    # Print experiment plan
    print(f"\nRunning {len(experiment_configs)} hyperparameter experiments:")
    for i, config in enumerate(experiment_configs):
        print(f"  {i+1}. {config['name']}")

    # Initialize Weights & Biases for the hyperparameter sweep
    if args.wandb_project:
        print(f"\nInitializing Weights & Biases project: {args.wandb_project}")
        tags = args.wandb_tags.split(",") if args.wandb_tags else []

        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            tags=tags,
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
                "transfer_learning_mode": args.transfer_learning_mode,
                "use_amp": args.use_amp,
                "total_experiments": len(experiment_configs),
                "normalizer": "Sentinel2Normalizer" if args.use_normalizer else "None",
            },
            job_type="sweep-control",
        )

        # Log the different hyperparameter combinations
        wandb.run.summary["sweep_configs"] = experiment_configs

    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(
            f"\nRunning experiments on {num_gpus} GPU(s): {torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "\nWARNING: No GPU detected. Experiments will run on CPU and may be very slow!"
        )

    # Save experiment plan
    with open(os.path.join(args.save_dir, "experiment_plan.json"), "w") as f:
        json.dump({"args": vars(args), "experiments": experiment_configs}, f, indent=2)

    print("\nStarting hyperparameter experiments...")
    best_config = run_hyperparameter_experiment(args, experiment_configs)

    print("\nExperiment complete!")
    print(f"Best configuration: {best_config}")

    # Save final results
    with open(os.path.join(args.save_dir, "experiment_results.json"), "w") as f:
        json.dump(
            {"best_config": best_config, "all_configs": experiment_configs}, f, indent=2
        )

    # Log final best config to wandb
    if args.wandb_project and wandb.run is not None:
        wandb.run.summary["final_best_config"] = best_config
        wandb.finish()
