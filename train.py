"""
Training and evaluation script for Sentinel-2 segmentation with TorchGeo integration.

This script implements a complete training pipeline with:
1. Differential learning rates for backbone vs. decoder/head
2. Progressive unfreezing for transfer learning
3. Mixed precision training for better performance
4. Multiple evaluation metrics (IoU, Dice, F1)
5. Weights & Biases logging
6. Best model saving
7. Automatic class detection from dataset
"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from dataset.augmentation import get_train_transform, get_val_transform
from dataset.class_remapper import create_class_remapper
from dataset.mask_handler import clean_mask, inspect_dataset_masks
from dataset.metrics import create_metrics
from dataset.patch_dataset import create_patch_data_loaders
from dataset.utils import detect_classes_from_dataset
from models.unet import UNet

# TODO: Check normalization of data (check prepare_dataset.py for normalization)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#python train.py --data_dir "F:\\output\remapped_dataset" 
#    --save_dir "F:\\output\model_results\stable" 
#    --model_name unet_sentinel2_stable 
#    --encoder_type best_performance 
#    --freeze_backbone 
#    --encoder_lr_factor 0.1 
#    --learning_rate 5e-4 
#    --scheduler onecycle 
#    --use_amp


def normalize_sentinel2_data(data, max_value=10000.0):
    """
    Normalize Sentinel-2 data to the range [0,1].
    
    Parameters:
    -----------
    data : torch.Tensor
        Input data tensor of shape (B, C, H, W)
    max_value : float
        Value to divide by (default 10000.0 for standard Sentinel-2 reflectance)
        
    Returns:
    --------
    torch.Tensor
        Normalized data tensor
    """
    # Check if normalization is needed
    if data.max() > 100:  # Heuristic to detect unnormalized data
        #print(f"Normalizing Sentinel-2 data from range [{data.min().item():.2f}, {data.max().item():.2f}]")
        
        # Handle very large values and NaN/Inf values
        data = torch.clamp(data, min=0.0, max=max_value * 3.0)  # Clip extreme outliers
        
        # Perform normalization to [0,1]
        data = data / max_value
        
        #print(f"After normalization: range [{data.min().item():.4f}, {data.max().item():.4f}]")
    
    return data


def create_optimizer_with_differential_lr(
    model, base_lr, encoder_lr_factor=0.1, weight_decay=1e-5, optimizer_type="adam"
):
    """
    Create an optimizer with different learning rates for different parts of the model.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to optimize
    base_lr : float
        Base learning rate for decoder and head
    encoder_lr_factor : float
        Factor to multiply base_lr for encoder (0 to freeze completely)
    weight_decay : float
        Weight decay for regularization
    optimizer_type : str
        Type of optimizer to use ('adam' or 'sgd')

    Returns:
    --------
    torch.optim.Optimizer
        Configured optimizer with parameter groups
    """
    # Define parameter groups with different learning rates
    params = [
        # Encoder parameters (lower learning rate or frozen)
        {
            "params": model.encoder.parameters(),
            "lr": base_lr * encoder_lr_factor if encoder_lr_factor > 0 else 0,
            "weight_decay": weight_decay,
        },
        # Decoder parameters (base learning rate)
        {
            "params": model.decoder.parameters(),
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        # Final classification layer (higher learning rate)
        {
            "params": model.final_conv.parameters(),
            "lr": base_lr * 1.0,  # Can use base_lr * X for higher rate if needed
            "weight_decay": weight_decay,
        },
    ]

    # Create optimizer with parameter groups
    if optimizer_type.lower() == "adam":
        return optim.AdamW(params)  # AdamW handles weight decay better
    else:  # sgd
        return optim.SGD(params, momentum=0.9)


def progressive_unfreeze_backbone(model, current_epoch, unfreeze_schedule=None):
    """
    Progressively unfreeze backbone layers as training progresses.

    Parameters:
    -----------
    model : UNet
        The UNet model
    current_epoch : int
        Current training epoch
    unfreeze_schedule : dict
        Dictionary mapping epoch numbers to layers to unfreeze
        Example: {10: 'layer4', 15: 'layer3', 20: 'layer2', 25: 'layer1'}
    """
    if unfreeze_schedule is None:
        return

    # Check if this epoch should trigger unfreezing
    for epoch, layer_name in unfreeze_schedule.items():
        if current_epoch == epoch:
            print(f"Epoch {current_epoch}: Unfreezing {layer_name}")

            # Unfreeze specified layer
            for name, param in model.encoder.backbone.named_parameters():
                if name.startswith(f"{layer_name}."):
                    param.requires_grad = True

            # Log the number of trainable parameters
            trainable_params = sum(
                p.numel()
                for p in model.encoder.backbone.parameters()
                if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.encoder.backbone.parameters())
            percentage = 100 * trainable_params / total_params

            print(
                f"Backbone now has {trainable_params:,}/{total_params:,} trainable parameters ({percentage:.1f}%)"
            )

            # Log to wandb if available
            if wandb.run is not None:
                wandb.log(
                    {
                        f"unfreeze/{layer_name}": current_epoch,
                        "trainable_backbone_params": trainable_params,
                        "trainable_backbone_percentage": percentage,
                    }
                )


"""
Updated train_epoch function with loss stabilization and debugging.
Replace the train_epoch function in your train.py with this improved version.
"""


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    metrics,
    device,
    num_classes,
    scaler=None,
    use_amp=False,
    remapper=None,
):
    """Train for one epoch with loss stabilization and debugging."""
    model.train()
    epoch_loss = 0
    batch_count = 0

    # Reset metrics at the start of epoch
    metrics.reset()

    with tqdm(dataloader, desc="Training") as pbar:
        for images, masks in pbar:
            batch_count += 1
            images = images.to(device)
            
            images = normalize_sentinel2_data(images)

            # Apply class remapping if provided
            if remapper is not None:
                masks = remapper.remap_mask(masks)

            # Clean mask to ensure valid class indices
            masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
            masks = masks.long().to(device)

            # Debug: Check masks for invalid values
            if torch.isnan(masks).any() or torch.isinf(masks).any():
                print(f"WARNING: NaN or Inf values found in masks, batch {batch_count}")
                print(f"Mask range: [{masks.min().item()}, {masks.max().item()}]")
                print(f"Mask unique values: {torch.unique(masks).cpu().numpy()}")
                # Skip this batch if there are issues
                continue

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with autocast(device_type="cuda"):
                    # Forward pass
                    outputs = model(images)

                    # Debug: Check outputs for extreme values
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(
                            f"WARNING: NaN or Inf values found in model outputs, batch {batch_count}"
                        )
                        # Skip this batch if there are issues
                        continue

                    # Add a small epsilon to prevent log(0) in softmax-based losses
                    if num_classes > 1:  # Multiclass segmentation
                        # Apply loss stabilization
                        outputs = torch.clamp(outputs, min=-50.0, max=50.0)

                    # Compute loss
                    try:
                        loss = criterion(outputs, masks)

                        # Check if loss is NaN or Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(
                                f"WARNING: NaN or Inf loss detected in batch {batch_count}"
                            )
                            print(
                                f"Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}"
                            )
                            # Skip this batch
                            continue

                    except Exception as e:
                        print(f"Error computing loss: {e}")
                        continue

                # Backward pass with gradient scaling
                try:
                    scaler.scale(loss).backward()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                except Exception as e:
                    print(f"Error in backward pass: {e}")
                    continue
            else:
                # Standard precision training
                outputs = model(images)

                # Debug: Check outputs for extreme values
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(
                        f"WARNING: NaN or Inf values found in model outputs, batch {batch_count}"
                    )
                    # Skip this batch if there are issues
                    continue

                # Add a small epsilon to prevent log(0) in softmax-based losses
                if num_classes > 1:  # Multiclass segmentation
                    # Apply loss stabilization
                    outputs = torch.clamp(outputs, min=-50.0, max=50.0)

                # Compute loss
                try:
                    loss = criterion(outputs, masks)

                    # Check if loss is NaN or Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(
                            f"WARNING: NaN or Inf loss detected in batch {batch_count}"
                        )
                        print(
                            f"Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}"
                        )
                        # Skip this batch
                        continue

                except Exception as e:
                    print(f"Error computing loss: {e}")
                    continue

                # Backward pass
                try:
                    loss.backward()

                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                except Exception as e:
                    print(f"Error in backward pass: {e}")
                    continue

            # Update metrics
            with torch.no_grad():
                metrics.update(outputs, masks)

            # Update loss statistics
            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(loss=loss.item())

            # Log batch metrics to wandb (optional)
            if batch_count % 10 == 0 and wandb.run is not None:
                wandb.log({"batch/train_loss": loss.item()})

    # Compute average metrics
    metric_values = metrics.compute()
    epoch_loss /= max(1, len(dataloader))  # Prevent division by zero

    return epoch_loss, metric_values


def validate(
    model,
    dataloader,
    criterion,
    metrics,
    device,
    num_classes,
    use_amp=False,
    remapper=None,
):
    """Validate the model with mixed precision support."""
    model.eval()
    val_loss = 0

    # Reset metrics at the start of validation
    metrics.reset()

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                
                images = normalize_sentinel2_data(images)

                # Apply class remapping if provided
                if remapper is not None:
                    masks = remapper.remap_mask(masks)

                # Clean mask to ensure valid class indices
                masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
                masks = masks.long().to(device)

                # Mixed precision inference
                if use_amp:
                    with autocast(device_type="cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # Update metrics
                metrics.update(outputs, masks)

                # Update loss statistics
                val_loss += loss.item()

                # Update progress bar
                pbar.set_postfix(loss=loss.item())

    # Compute average metrics
    metric_values = metrics.compute()
    val_loss /= len(dataloader)

    return val_loss, metric_values


def plot_learning_curves(history, best_epoch, save_path):
    """Plot learning curves for loss and metrics."""
    # Plot loss, IoU, Dice, and F1 in a 2x2 grid
    plt.figure(figsize=(18, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot IoU
    plt.subplot(2, 2, 2)
    plt.plot(history["train_iou"], label="Train IoU")
    plt.plot(history["val_iou"], label="Validation IoU")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU")
    plt.legend()
    plt.grid(True)

    # Plot Dice
    plt.subplot(2, 2, 3)
    plt.plot(history["train_dice"], label="Train Dice")
    plt.plot(history["val_dice"], label="Validation Dice")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training and Validation Dice")
    plt.legend()
    plt.grid(True)

    # Plot Learning Rates
    plt.subplot(2, 2, 4)
    plt.plot(history["learning_rates"], label="Decoder LR")
    plt.plot(history["encoder_learning_rates"], label="Encoder LR")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    # Log plots to wandb if it's being used
    if wandb.run is not None:
        wandb.log({"learning_curves": wandb.Image(save_path)})

    plt.close("all")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    scheduler=None,
    save_dir="models",
    model_name="unet_sentinel2",
    early_stopping=10,
    wandb_project=None,
    config=None,
    num_classes=None,
    remapper=None,
    use_amp=True,
    unfreeze_schedule=None,
):
    """Train the model with monitoring, validation, and optimization features."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Ensure config is a valid dictionary
    if config is None:
        config = {
            "num_classes": num_classes,
            "batch_size": (
                train_loader.batch_size if hasattr(train_loader, "batch_size") else 8
            ),
            "learning_rate": (
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else optimizer.param_groups[0]["lr"]
            ),
            "scheduler": scheduler.__class__.__name__ if scheduler else None,
        }

    # Setup Weights & Biases if a project is specified
    if wandb_project and wandb.run is None:
        wandb.init(project=wandb_project, config=config)
        wandb.watch(model, log="all", log_freq=100)

    # Initialize metrics using the explicit num_classes parameter instead of config
    train_metrics = create_metrics(
        num_classes=num_classes, threshold=0.5, device=device
    )
    val_metrics = create_metrics(num_classes=num_classes, threshold=0.5, device=device)

    # Initialize mixed precision training if requested
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # Initialize history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rates": [],
        "encoder_learning_rates": [],
    }

    # Initialize early stopping
    early_stop_counter = 0
    best_val_metric = float("-inf")
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Progressive unfreezing if scheduled
        if unfreeze_schedule:
            progressive_unfreeze_backbone(model, epoch, unfreeze_schedule)

        # Train for one epoch
        train_loss, train_metrics_values = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            train_metrics,
            device,
            num_classes,
            scaler,
            use_amp,
            remapper,
        )

        # Validate
        val_loss, val_metrics_values = validate(
            model,
            val_loader,
            criterion,
            val_metrics,
            device,
            num_classes,
            use_amp,
            remapper,
        )

        # Get current learning rates
        current_lr = optimizer.param_groups[1]["lr"]  # Decoder LR
        encoder_lr = optimizer.param_groups[0]["lr"]  # Encoder LR

        # Update learning rate if scheduler is provided
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_metrics_values["iou"])
        history["val_iou"].append(val_metrics_values["iou"])
        history["train_dice"].append(train_metrics_values["dice"])
        history["val_dice"].append(val_metrics_values["dice"])
        history["train_f1"].append(train_metrics_values["f1"])
        history["val_f1"].append(val_metrics_values["f1"])
        history["learning_rates"].append(current_lr)
        history["encoder_learning_rates"].append(encoder_lr)

        # Print epoch statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(
            f"Train IoU: {train_metrics_values['iou']:.4f}, Val IoU: {val_metrics_values['iou']:.4f}"
        )
        print(
            f"Train Dice: {train_metrics_values['dice']:.4f}, Val Dice: {val_metrics_values['dice']:.4f}"
        )
        print(f"Decoder LR: {current_lr:.6f}, Encoder LR: {encoder_lr:.6f}")

        # Log metrics to wandb
        if wandb_project and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/iou": train_metrics_values["iou"],
                    "train/dice": train_metrics_values["dice"],
                    "train/f1": train_metrics_values["f1"],
                    "val/loss": val_loss,
                    "val/iou": val_metrics_values["iou"],
                    "val/dice": val_metrics_values["dice"],
                    "val/f1": val_metrics_values["f1"],
                    "learning_rate/decoder": current_lr,
                    "learning_rate/encoder": encoder_lr,
                    "backbone/trainable_params": sum(
                        p.numel() for p in model.encoder.parameters() if p.requires_grad
                    ),
                }
            )

        # Check if this is the best model
        if val_metrics_values["iou"] > best_val_metric:
            best_val_metric = val_metrics_values["iou"]
            best_epoch = epoch

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_metrics_values["iou"],
                "val_dice": val_metrics_values["dice"],
                "val_f1": val_metrics_values["f1"],
                "num_classes": num_classes,
                "config": config,
            }

            # Save remapper if used
            if remapper is not None:
                checkpoint["remapper"] = {
                    "class_mapping": remapper.class_mapping,
                    "reverse_mapping": remapper.reverse_mapping,
                    "class_values": remapper.class_values,
                }

            torch.save(
                checkpoint,
                os.path.join(save_dir, f"{model_name}_best.pth"),
            )
            print(f"Saved best model with Val IoU: {best_val_metric:.4f}")

            # Reset early stopping counter
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stopping:
            print(f"Early stopping after {early_stopping} epochs without improvement")
            if wandb_project and wandb.run is not None:
                wandb.run.summary["stopped_early"] = True
                wandb.run.summary["early_stop_epoch"] = epoch + 1
            break

    # Save training history
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        json_history = {}
        for k, v in history.items():
            json_history[k] = [float(x) for x in v]

        json_history["best_epoch"] = best_epoch
        json_history["best_val_metric"] = float(best_val_metric)

        json.dump(json_history, f, indent=4)

    # Close wandb run if it was initialized
    if wandb_project and wandb.run is not None:
        wandb.finish()

    return history, best_epoch


def run_hyperparameter_experiment(args, experiment_configs=None):
    """Run multiple training experiments with different hyperparameters."""
    # Create experiment directory
    experiment_dir = os.path.join(args.save_dir, "experiments")
    os.makedirs(experiment_dir, exist_ok=True)

    # Get data loaders
    train_transform = get_train_transform(
        p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste
    )
    val_transform = get_val_transform(patch_size=args.patch_size)

    data_loaders = create_patch_data_loaders(
        patches_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=args.num_workers,
    )

    # Auto-detect classes if not specified
    class_remapper = None
    if args.auto_detect_classes or args.num_classes is None:
        print("Auto-detecting classes from dataset...")
        class_info = detect_classes_from_dataset(
            data_loaders["train"].dataset, sample_size=100
        )
        args.num_classes = class_info["num_classes"]
        print(f"Auto-detected {args.num_classes} classes")

    # Define default hyperparameter configurations if not provided
    if experiment_configs is None:
        experiment_configs = [
            {
                "name": "lr_1e-3",
                "learning_rate": 1e-3,
                "encoder_lr_factor": 0.1,
                "optimizer": "adam",
                "weight_decay": 1e-4,
                "scheduler": "onecycle",
                "freeze_backbone": True,
                "progressive_unfreeze": False,
            },
            {
                "name": "lr_1e-4",
                "learning_rate": 1e-4,
                "encoder_lr_factor": 0.1,
                "optimizer": "adam",
                "weight_decay": 1e-4,
                "scheduler": "onecycle",
                "freeze_backbone": True,
                "progressive_unfreeze": False,
            },
            {
                "name": "sgd",
                "learning_rate": 1e-2,
                "encoder_lr_factor": 0.1,
                "optimizer": "sgd",
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "scheduler": "onecycle",
                "freeze_backbone": True,
                "progressive_unfreeze": False,
            },
            {
                "name": "progressive",
                "learning_rate": 1e-3,
                "encoder_lr_factor": 0.1,
                "optimizer": "adam",
                "weight_decay": 1e-4,
                "scheduler": "onecycle",
                "freeze_backbone": True,
                "progressive_unfreeze": True,
            },
            {
                "name": "finetune",
                "learning_rate": 5e-4,
                "encoder_lr_factor": 0.1,
                "optimizer": "adam",
                "weight_decay": 1e-4,
                "scheduler": "onecycle",
                "freeze_backbone": False,
                "progressive_unfreeze": False,
            },
        ]

    # Store results
    results = []

    # Initialize wandb sweep if it's being used
    wandb_project = getattr(args, "wandb_project", None)

    # Loop through hyperparameter combinations
    for i, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiment_configs)}: {config['name']}")
        print(f"{'='*80}\n")

        # Create model
        model = UNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            encoder_type=args.encoder_type,
            use_batchnorm=args.use_batchnorm,
            skip_connections=args.skip_connections,
            freeze_backbone=config.get("freeze_backbone", True),
        ).to(device)

        # Create optimizer with differential learning rates
        optimizer = create_optimizer_with_differential_lr(
            model=model,
            base_lr=config["learning_rate"],
            encoder_lr_factor=config.get("encoder_lr_factor", 0.1),
            weight_decay=config.get("weight_decay", 1e-4),
            optimizer_type=config.get("optimizer", "adam"),
        )

        # Create loss function
        criterion = (
            nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
        )

        # Create scheduler
        scheduler = None
        if config.get("scheduler") == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs, eta_min=config.get("min_lr", 1e-6)
            )
        elif config.get("scheduler") == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 10),
                gamma=config.get("gamma", 0.1),
            )
        elif config.get("scheduler") == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.get("factor", 0.1),
                patience=config.get("patience", 5),
            )
        elif config.get("scheduler") == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[
                    config["learning_rate"] * config.get("encoder_lr_factor", 0.1),
                    config["learning_rate"],
                    config["learning_rate"],
                ],
                total_steps=args.num_epochs * len(data_loaders["train"]),
                pct_start=0.3,
            )

        # Create progressive unfreezing schedule if requested
        unfreeze_schedule = None
        if config.get("progressive_unfreeze", False):
            # Define unfreeze schedule based on total epochs
            total_epochs = args.num_epochs
            unfreeze_schedule = {
                int(total_epochs * 0.3): "layer4",  # Unfreeze at 30% of training
                int(total_epochs * 0.5): "layer3",  # Unfreeze at 50% of training
                int(total_epochs * 0.7): "layer2",  # Unfreeze at 70% of training
                int(total_epochs * 0.9): "layer1",  # Unfreeze at 90% of training
            }
            print(f"Progressive unfreezing schedule: {unfreeze_schedule}")

        # Save directory for this experiment
        exp_save_dir = os.path.join(experiment_dir, config["name"])
        os.makedirs(exp_save_dir, exist_ok=True)

        # Prepare wandb config
        wandb_config = {
            **config,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "encoder_type": args.encoder_type,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "early_stopping": args.early_stopping,
            "dataset": os.path.basename(args.data_dir),
            "experiment": config["name"],
        }

        # Train model
        print(f"Training with config: {config}")
        history, best_epoch = train_model(
            model,
            data_loaders["train"],
            data_loaders["val"],
            criterion,
            optimizer,
            args.num_epochs,
            device,
            scheduler=scheduler,
            save_dir=exp_save_dir,
            model_name=args.model_name,
            early_stopping=args.early_stopping,
            wandb_project=wandb_project,
            config=wandb_config,
            num_classes=args.num_classes,
            remapper=class_remapper,
            use_amp=getattr(args, "use_amp", True),
            unfreeze_schedule=unfreeze_schedule,
        )

        # Plot learning curves
        plot_learning_curves(
            history,
            best_epoch,
            save_path=os.path.join(exp_save_dir, f"{args.model_name}_curves.png"),
        )

        # Store results
        checkpoint = torch.load(
            os.path.join(exp_save_dir, f"{args.model_name}_best.pth")
        )

        results.append(
            {
                "config": config,
                "best_epoch": best_epoch + 1,
                "val_iou": checkpoint["val_iou"],
                "val_dice": checkpoint["val_dice"],
                "val_loss": checkpoint["val_loss"],
            }
        )

    # Find best configuration
    best_config = max(results, key=lambda x: x["val_iou"])

    # Print summary
    print("\nHyperparameter Experiment Results:")
    print("=" * 80)
    for result in results:
        print(f"Config: {result['config']['name']}")
        print(f"Val IoU: {result['val_iou']:.4f}, Val Dice: {result['val_dice']:.4f}")
        print(f"Best Epoch: {result['best_epoch']}")
        print("-" * 80)

    print("\nBest Configuration:")
    print(f"Config: {best_config['config']['name']}")
    print(
        f"Val IoU: {best_config['val_iou']:.4f}, Val Dice: {best_config['val_dice']:.4f}"
    )
    print(f"Best Epoch: {best_config['best_epoch']}")

    # Save summary
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, default=str)

    # Log summary table to wandb if it's being used
    if wandb_project and wandb.run is not None:
        # Create a table to log all results
        columns = [
            "config_name",
            "learning_rate",
            "encoder_lr_factor",
            "optimizer",
            "scheduler",
            "freeze_backbone",
            "progressive_unfreeze",
            "best_epoch",
            "val_iou",
            "val_dice",
            "val_loss",
        ]
        data = []

        for result in results:
            config = result["config"]
            data.append(
                [
                    config["name"],
                    config["learning_rate"],
                    config.get("encoder_lr_factor", 0.1),
                    config["optimizer"],
                    config.get("scheduler", "None"),
                    config.get("freeze_backbone", True),
                    config.get("progressive_unfreeze", False),
                    result["best_epoch"],
                    result["val_iou"],
                    result["val_dice"],
                    result["val_loss"],
                ]
            )

        # Log table
        results_table = wandb.Table(columns=columns, data=data)
        wandb.log({"hyperparameter_results": results_table})

        # Log best config
        wandb.run.summary["best_config"] = best_config["config"]["name"]
        wandb.run.summary["best_config_val_iou"] = best_config["val_iou"]

        # Finish the wandb run
        wandb.finish()

    return best_config["config"]


def main(args):
    """Main training function."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    train_transform = get_train_transform(
        p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste
    )
    val_transform = get_val_transform(patch_size=args.patch_size)

    data_loaders = create_patch_data_loaders(
        patches_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=args.num_workers,
    )

    print(f"Training on {len(data_loaders['train'].dataset)} samples")
    print(f"Validating on {len(data_loaders['val'].dataset)} samples")

    # Inspect dataset masks to detect issues
    print("\nInspecting dataset for mask issues...")
    train_inspection = inspect_dataset_masks(
        data_loaders["train"].dataset, num_samples=50
    )

    # Initialize class remapper if sparse indices detected
    class_remapper = None
    if train_inspection.get("sparse_indices", False):
        print("\nDetected sparse class indices. Creating class remapper...")
        class_remapper = create_class_remapper(
            data_loaders["train"].dataset, num_samples=100
        )

        # Update number of classes based on remapped classes
        args.num_classes = class_remapper.num_classes
        print(f"Using {args.num_classes} classes after remapping")
    # Detect classes if not specified
    elif args.auto_detect_classes or args.num_classes is None:
        print("Auto-detecting classes from dataset...")
        class_info = detect_classes_from_dataset(
            data_loaders["train"].dataset, sample_size=100
        )
        detected_num_classes = class_info["num_classes"]

        # Check if auto-detected classes match with inspection results
        if "recommended_num_classes" in train_inspection:
            recommended = train_inspection["recommended_num_classes"]
            if recommended > detected_num_classes:
                print(
                    f"WARNING: Inspection suggests {recommended} classes but auto-detection found {detected_num_classes}"
                )
                print(f"Using the larger value ({recommended}) to be safe")
                args.num_classes = recommended
            else:
                args.num_classes = detected_num_classes
        else:
            args.num_classes = detected_num_classes

        print(f"Auto-detected {args.num_classes} classes")

        # Update wandb config with detected classes
        if args.wandb_project and not args.run_experiment:
            wandb.config.update(
                {
                    "num_classes": args.num_classes,
                    "class_values": class_info["class_values"],
                    "class_distribution": class_info["class_distribution"],
                    "is_binary_segmentation": class_info["is_binary"],
                }
            )

    # Initialize wandb
    if args.wandb_project and not args.run_experiment:
        print(f"Initializing Weights & Biases project: {args.wandb_project}")
        config = {
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "encoder_type": args.encoder_type,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "encoder_lr_factor": args.encoder_lr_factor,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "num_epochs": args.num_epochs,
            "dataset": os.path.basename(args.data_dir),
            "use_amp": getattr(args, "use_amp", True),
            "freeze_backbone": getattr(args, "freeze_backbone", True),
            "progressive_unfreeze": getattr(args, "progressive_unfreeze", False),
        }

        # Add class remapping info if used
        if class_remapper is not None:
            config["using_class_remapper"] = True
            config["original_class_values"] = sorted(
                list(class_remapper.class_mapping.keys())
            )
            config["num_original_classes"] = len(class_remapper.class_mapping)

        wandb.init(project=args.wandb_project, config=config)

    # Log dataset info to wandb
    if args.wandb_project and wandb.run is not None:
        wandb.run.summary["train_samples"] = len(data_loaders["train"].dataset)
        wandb.run.summary["val_samples"] = len(data_loaders["val"].dataset)
        wandb.run.summary["mask_inspection"] = train_inspection

    # Run hyperparameter experiment if requested
    if args.run_experiment:
        print("Running hyperparameter experiment...")
        best_config = run_hyperparameter_experiment(args)

        # Set best hyperparameters for final training
        args.learning_rate = best_config["learning_rate"]
        args.optimizer = best_config["optimizer"]
        if "momentum" in best_config:
            args.momentum = best_config["momentum"]
        args.weight_decay = best_config["weight_decay"]
        args.scheduler = best_config.get("scheduler")
        args.freeze_backbone = best_config.get("freeze_backbone", True)
        args.progressive_unfreeze = best_config.get("progressive_unfreeze", False)

        print(f"Using best hyperparameters for final training: {best_config}")

    # Create model
    print(f"Creating U-Net model with {args.num_classes} output classes...")
    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        encoder_type=args.encoder_type,
        use_batchnorm=args.use_batchnorm,
        skip_connections=args.skip_connections,
        freeze_backbone=getattr(args, "freeze_backbone", True),
    ).to(device)

    # Create optimizer with differential learning rates
    optimizer = create_optimizer_with_differential_lr(
        model=model,
        base_lr=args.learning_rate,
        encoder_lr_factor=args.encoder_lr_factor,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
    )

    # Create loss function with ignore_index for invalid labels
    if args.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Create scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.factor, patience=args.patience
        )
    elif args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                args.learning_rate * args.encoder_lr_factor,
                args.learning_rate,
                args.learning_rate,
            ],
            total_steps=args.num_epochs * len(data_loaders["train"]),
            pct_start=0.3,
        )

    # Create progressive unfreezing schedule if requested
    unfreeze_schedule = None
    if getattr(args, "progressive_unfreeze", False):
        # Define unfreeze schedule based on total epochs
        total_epochs = args.num_epochs
        unfreeze_schedule = {
            int(total_epochs * 0.3): "layer4",  # Unfreeze at 30% of training
            int(total_epochs * 0.5): "layer3",  # Unfreeze at 50% of training
            int(total_epochs * 0.7): "layer2",  # Unfreeze at 70% of training
            int(total_epochs * 0.9): "layer1",  # Unfreeze at 90% of training
        }
        print(f"Progressive unfreezing schedule: {unfreeze_schedule}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Train model
    print("Starting training...")
    history, best_epoch = train_model(
        model,
        data_loaders["train"],
        data_loaders["val"],
        criterion,
        optimizer,
        args.num_epochs,
        device,
        scheduler=scheduler,
        save_dir=args.save_dir,
        model_name=args.model_name,
        early_stopping=args.early_stopping,
        wandb_project=args.wandb_project,
        config=config if "config" in locals() else None,
        num_classes=args.num_classes,
        remapper=class_remapper,
        use_amp=getattr(args, "use_amp", True),
        unfreeze_schedule=unfreeze_schedule,
    )

    # Plot learning curves
    print("Plotting learning curves...")
    plot_learning_curves(
        history,
        best_epoch,
        save_path=os.path.join(args.save_dir, f"{args.model_name}_curves.png"),
    )

    print(f"Training completed. Best model saved at epoch {best_epoch+1}")

    # Save class remapper if used
    if class_remapper is not None:
        remapper_path = os.path.join(args.save_dir, f"{args.model_name}_remapper.json")
        class_remapper.save(remapper_path)
        print(f"Saved class remapper to {remapper_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Sentinel-2 segmentation model with TorchGeo integration"
    )

    # Model parameters
    parser.add_argument(
        "--in_channels", type=int, default=13, help="Number of input channels"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected if not specified)",
    )
    parser.add_argument(
        "--encoder_type", type=str, default="best_performance", help="Encoder type"
    )
    parser.add_argument(
        "--use_batchnorm", action="store_true", help="Use batch normalization"
    )
    parser.add_argument(
        "--skip_connections", type=int, default=4, help="Number of skip connections"
    )
    parser.add_argument(
        "--auto_detect_classes",
        action="store_true",
        help="Auto-detect number of classes from dataset",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze backbone for transfer learning",
    )
    parser.add_argument(
        "--progressive_unfreeze",
        action="store_true",
        help="Progressively unfreeze backbone layers",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--encoder_lr_factor",
        type=float,
        default=0.1,
        help="Learning rate factor for encoder",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum (for SGD)"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )

    # Scheduler parameters
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=[None, "plateau", "cosine", "step", "onecycle"],
        help="LR scheduler",
    )
    parser.add_argument(
        "--factor", type=float, default=0.1, help="Factor for ReduceLROnPlateau"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for ReduceLROnPlateau"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Min LR for CosineAnnealingLR"
    )
    parser.add_argument(
        "--step_size", type=int, default=10, help="Step size for StepLR"
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR")

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Data directory with patches"
    )
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--use_copy_paste", action="store_true", help="Use CopyPaste augmentation"
    )

    # Experiment parameters
    parser.add_argument(
        "--run_experiment", action="store_true", help="Run hyperparameter experiment"
    )

    # Saving parameters
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--model_name", type=str, default="unet_sentinel2", help="Model name"
    )

    # Wandb parameters
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Weights & Biases project name"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(args)
