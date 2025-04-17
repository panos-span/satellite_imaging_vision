"""
Training and evaluation script for Sentinel-2 segmentation models.

This script implements a complete training pipeline with:
1. Multiple evaluation metrics (IoU, Dice, F1) using torchmetrics
2. Training and validation curves with Weights & Biases logging
3. Hyperparameter experiments
4. Best model saving
5. Automatic class detection from dataset
6. Class remapping for sparse class indices
"""

import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models import UNet
from dataset.patch_dataset import create_patch_data_loaders
from dataset.augmentation import get_train_transform, get_val_transform
from dataset.metrics import create_metrics
from dataset.utils import detect_classes_from_dataset
from dataset.mask_handler import validate_mask, clean_mask, inspect_dataset_masks
from dataset.class_remapper import create_class_remapper


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
python train.py    --data_dir "F:\\output\dataset"    --save_dir "F:\\output\train"    --auto_detect_classes    --in_channels 13    --encoder_type "best_performance"    --batch_size 8    --num_epochs 50    --learning_rate 1e-4    --optimizer adam    --scheduler cosine    --use_batchnorm    --wandb_project "sentinel2-segmentation"
"""


def train_epoch(model, dataloader, criterion, optimizer, metrics, device, num_classes, remapper=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    # Check if wandb is initialized
    wandb_initialized = wandb.run is not None
    
    # Reset metrics at the start of epoch
    metrics.reset()
    
    with tqdm(dataloader, desc="Training") as pbar:
        for images, masks in pbar:
            batch_count += 1
            images = images.to(device)
            
            # Apply class remapping if provided
            if remapper is not None:
                masks = remapper.remap_mask(masks)
            
            # Clean mask to ensure valid class indices
            # This is critical to avoid "Assertion `t >= 0 && t < n_classes` failed" errors
            masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
            
            # Convert masks to Long type for CrossEntropyLoss
            masks = masks.long().to(device)
            
            # Debug: Check for invalid values
            if batch_count == 1 or batch_count % 100 == 0:
                validate_mask(masks, num_classes, name=f"Batch {batch_count} masks")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            metrics.update(outputs, masks)
            
            # Update loss statistics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
            
            # Log batch metrics to wandb (optional, but useful for detailed tracking)
            if wandb_initialized and batch_count % 10 == 0: # Log every 10 batches to avoid flooding
                
                wandb.log({
                    "batch/train_loss": loss.item(),
                })
    
    # Compute average metrics
    metric_values = metrics.compute()
    epoch_loss /= len(dataloader)
    
    # Return all metrics
    return epoch_loss, metric_values


def validate(model, dataloader, criterion, metrics, device, num_classes, remapper=None):
    """Validate the model."""
    model.eval()
    val_loss = 0
    
    # Reset metrics at the start of validation
    metrics.reset()
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                
                # Apply class remapping if provided
                if remapper is not None:
                    masks = remapper.remap_mask(masks)
                
                # Clean mask to ensure valid class indices
                masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
                
                # Convert masks to Long type for CrossEntropyLoss
                masks = masks.long().to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
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
    
    # Return all metrics
    return val_loss, metric_values


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
    remapper=None
):
    """Train the model with monitoring and validation."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup Weights & Biases if a project is specified
    if wandb_project:
        if wandb.run is None:  # Check if wandb is already initialized
            wandb.init(project=wandb_project, config=config)
        
        # Log model architecture as a summary
        wandb.watch(model, log="all", log_freq=100)
        
        # Log class remapper if used
        if remapper is not None:
            wandb.config.update({
                "remapped_classes": remapper.num_classes,
                "original_class_values": remapper.class_values,
                "class_mapping": remapper.class_mapping
            })
    
    # Initialize metrics
    train_metrics = create_metrics(
        num_classes=config.get("num_classes", 1), 
        threshold=0.5, 
        device=device
    )
    val_metrics = create_metrics(
        num_classes=config.get("num_classes", 1), 
        threshold=0.5, 
        device=device
    )
    
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
        "learning_rates": []
    }
    
    # Initialize early stopping counter
    early_stop_counter = 0
    best_val_metric = float('-inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_metrics_values = train_epoch(
            model, train_loader, criterion, optimizer, train_metrics, device, num_classes, remapper
        )
        
        # Validate
        val_loss, val_metrics_values = validate(
            model, val_loader, criterion, val_metrics, device, num_classes, remapper
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
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
        
        # Print epoch statistics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train IoU: {train_metrics_values['iou']:.4f}, Val IoU: {val_metrics_values['iou']:.4f}")
        print(f"Train Dice: {train_metrics_values['dice']:.4f}, Val Dice: {val_metrics_values['dice']:.4f}")
        print(f"Train F1: {train_metrics_values['f1']:.4f}, Val F1: {val_metrics_values['f1']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log metrics to wandb
        if wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/iou": train_metrics_values["iou"],
                "train/dice": train_metrics_values["dice"],
                "train/f1": train_metrics_values["f1"],
                "train/accuracy": train_metrics_values["accuracy"],
                "val/loss": val_loss,
                "val/iou": val_metrics_values["iou"],
                "val/dice": val_metrics_values["dice"],
                "val/f1": val_metrics_values["f1"],
                "val/accuracy": val_metrics_values["accuracy"],
                "learning_rate": current_lr
            })
        
        # Check if this is the best model (using IoU as primary metric)
        if val_metrics_values["iou"] > best_val_metric:
            best_val_metric = val_metrics_values["iou"]
            best_epoch = epoch
            
            # Save best model (only weights, not the full model as requested)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_metrics_values["iou"],
                "val_dice": val_metrics_values["dice"],
                "val_f1": val_metrics_values["f1"],
                "num_classes": num_classes
            }
            
            # Save class remapper if used
            if remapper is not None:
                checkpoint["remapper"] = {
                    "class_mapping": remapper.class_mapping,
                    "reverse_mapping": remapper.reverse_mapping,
                    "class_values": remapper.class_values
                }
            
            torch.save(
                checkpoint,
                os.path.join(save_dir, f"{model_name}_best.pth"),
            )
            print(f"Saved best model with Val IoU: {best_val_metric:.4f}")
            
            # Also save remapper if used
            if remapper is not None:
                remapper_path = os.path.join(save_dir, f"{model_name}_remapper.json")
                remapper.save(remapper_path)
                print(f"Saved class remapper to {remapper_path}")
            
            # Reset early stopping counter
            early_stop_counter = 0
            
            # Log best model metrics to wandb
            if wandb_project:
                wandb.run.summary["best_epoch"] = epoch + 1
                wandb.run.summary["best_val_iou"] = val_metrics_values["iou"]
                wandb.run.summary["best_val_dice"] = val_metrics_values["dice"]
                wandb.run.summary["best_val_f1"] = val_metrics_values["f1"]
                wandb.run.summary["best_val_loss"] = val_loss
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= early_stopping:
            print(f"Early stopping after {early_stopping} epochs without improvement")
            if wandb_project:
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
    
    # Plot F1
    plt.subplot(2, 2, 4)
    plt.plot(history["train_f1"], label="Train F1")
    plt.plot(history["val_f1"], label="Validation F1")
    plt.axvline(x=best_epoch, color="r", linestyle="--", label="Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Training and Validation F1")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    # Plot learning rate separately
    plt.figure(figsize=(8, 5))
    plt.plot(history["learning_rates"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.savefig(save_path.replace(".png", "_lr.png"), dpi=300)
    
    # Log plots to wandb if it's being used
    if wandb.run is not None:
        wandb.log({
            "learning_curves": wandb.Image(save_path),
            "learning_rate_curve": wandb.Image(save_path.replace(".png", "_lr.png"))
        })
    
    plt.close('all')


def run_hyperparameter_experiment(args):
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
    
    # Define hyperparameter configurations
    configs = []
    
    # Experiment with learning rate
    for lr in [1e-3, 1e-4, 1e-5]:
        configs.append({
            "name": f"lr_{lr}",
            "learning_rate": lr,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "scheduler": None,
        })
    
    # Experiment with optimizer
    configs.append({
        "name": "sgd",
        "learning_rate": 1e-2,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler": None,
    })
    
    # Experiment with scheduler
    configs.append({
        "name": "cosine",
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "min_lr": 1e-6,
    })
    
    # Store results
    results = []
    
    # Initialize wandb sweep if it's being used
    wandb_project = getattr(args, "wandb_project", None)
    
    # Loop through hyperparameter combinations
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(configs)}: {config['name']}")
        print(f"{'='*80}\n")
        
        # Create model
        model = UNet(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            encoder_type=args.encoder_type,
            use_batchnorm=args.use_batchnorm,
            skip_connections=args.skip_connections,
        ).to(device)
        
        # Create loss function
        criterion = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
        
        # Create optimizer
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config.get("weight_decay", 0),
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(),
                lr=config["learning_rate"],
                momentum=config.get("momentum", 0.9),
                weight_decay=config.get("weight_decay", 0),
            )
        
        # Create scheduler
        if config.get("scheduler") == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.num_epochs,
                eta_min=config.get("min_lr", 1e-6)
            )
        elif config.get("scheduler") == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 10),
                gamma=config.get("gamma", 0.1)
            )
        elif config.get("scheduler") == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.get("factor", 0.1),
                patience=config.get("patience", 5)
            )
        else:
            scheduler = None
        
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
            "experiment": config["name"]
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
            config=wandb_config
        )
        
        # Plot learning curves
        plot_learning_curves(
            history,
            best_epoch,
            save_path=os.path.join(exp_save_dir, f"{args.model_name}_curves.png"),
        )
        
        # Store results
        checkpoint = torch.load(os.path.join(exp_save_dir, f"{args.model_name}_best.pth"))
        
        results.append({
            "config": config,
            "best_epoch": best_epoch + 1,
            "val_iou": checkpoint["val_iou"],
            "val_dice": checkpoint["val_dice"],
            "val_loss": checkpoint["val_loss"],
        })
    
    # Find best configuration
    best_config = max(results, key=lambda x: x["val_iou"])
    
    # Print summary
    print("\nHyperparameter Experiment Results:")
    print("="*80)
    for result in results:
        print(f"Config: {result['config']['name']}")
        print(f"Val IoU: {result['val_iou']:.4f}, Val Dice: {result['val_dice']:.4f}")
        print(f"Best Epoch: {result['best_epoch']}")
        print("-"*80)
    
    print("\nBest Configuration:")
    print(f"Config: {best_config['config']['name']}")
    print(f"Val IoU: {best_config['val_iou']:.4f}, Val Dice: {best_config['val_dice']:.4f}")
    print(f"Best Epoch: {best_config['best_epoch']}")
    
    # Save summary
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    # Log summary table to wandb if it's being used
    if wandb_project and wandb.run is not None:
        # Create a table to log all results
        columns = ["config_name", "learning_rate", "optimizer", "scheduler", 
                   "best_epoch", "val_iou", "val_dice", "val_loss"]
        data = []
        
        for result in results:
            config = result["config"]
            data.append([
                config["name"],
                config["learning_rate"],
                config["optimizer"],
                config.get("scheduler", "None"),
                result["best_epoch"],
                result["val_iou"],
                result["val_dice"],
                result["val_loss"]
            ])
        
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
    train_inspection = inspect_dataset_masks(data_loaders['train'].dataset, num_samples=50)
    
    # Initialize class remapper if sparse indices detected
    class_remapper = None
    if train_inspection.get("sparse_indices", False):
        print("\nDetected sparse class indices. Creating class remapper...")
        class_remapper = create_class_remapper(data_loaders['train'].dataset, num_samples=100)
        
        # Update number of classes based on remapped classes
        args.num_classes = class_remapper.num_classes
        print(f"Using {args.num_classes} classes after remapping")
    # Detect classes if not specified
    elif args.auto_detect_classes or args.num_classes is None:
        print("Auto-detecting classes from dataset...")
        class_info = detect_classes_from_dataset(data_loaders['train'].dataset)
        detected_num_classes = class_info['num_classes']
        
        # Check if auto-detected classes match with inspection results
        if 'recommended_num_classes' in train_inspection:
            recommended = train_inspection['recommended_num_classes']
            if recommended > detected_num_classes:
                print(f"WARNING: Inspection suggests {recommended} classes but auto-detection found {detected_num_classes}")
                print(f"Using the larger value ({recommended}) to be safe")
                args.num_classes = recommended
            else:
                args.num_classes = detected_num_classes
        else:
            args.num_classes = detected_num_classes
            
        print(f"Auto-detected {args.num_classes} classes")
        
        # Update wandb config with detected classes
        if args.wandb_project and not args.run_experiment:
            config = {
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "encoder_type": args.encoder_type,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "num_epochs": args.num_epochs,
                "dataset": os.path.basename(args.data_dir),
            }
            
            # Add class remapping info if used
            if class_remapper is not None:
                config["using_class_remapper"] = True
                config["original_class_values"] = sorted(list(class_remapper.class_mapping.keys()))
                config["num_original_classes"] = len(class_remapper.class_mapping)
            
            wandb.init(project=args.wandb_project, config=config)
            wandb.config.update({"num_classes": args.num_classes, 
                               "class_values": class_info['class_values'],
                               "class_distribution": class_info['class_distribution'],
                               "is_binary_segmentation": class_info['is_binary']})
    
    # Initialize wandb
    if args.wandb_project:
        print(f"Initializing Weights & Biases project: {args.wandb_project}")
        if not args.run_experiment:  # For single run (not experiment)
            config = {
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "encoder_type": args.encoder_type,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "num_epochs": args.num_epochs,
                "dataset": os.path.basename(args.data_dir),
            }
            
            # Add class remapping info if used
            if class_remapper is not None:
                config["using_class_remapper"] = True
                config["original_class_values"] = sorted(list(class_remapper.class_mapping.keys()))
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
        
        print(f"Using best hyperparameters for final training: {best_config}")
    
    # Create model
    print(f"Creating model with {args.num_classes} output classes...")
    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        encoder_type=args.encoder_type,
        use_batchnorm=args.use_batchnorm,
        skip_connections=args.skip_connections,
    ).to(device)
    
    # Create optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    
    # Create loss function with ignore_index for invalid labels
    if args.num_classes > 1:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.factor,
            patience=args.patience
        )
    else:
        scheduler = None
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare wandb config for training
    wandb_config = {
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "encoder_type": args.encoder_type,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "num_epochs": args.num_epochs,
        "dataset": os.path.basename(args.data_dir),
    }
    
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
        config=wandb_config,
        num_classes=args.num_classes,
        remapper=class_remapper
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
    
    parser = argparse.ArgumentParser(description="Train Sentinel-2 segmentation model")
    
    # Model parameters
    parser.add_argument("--in_channels", type=int, default=13, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detected if not specified)")
    parser.add_argument("--encoder_type", type=str, default="best_performance", help="Encoder type")
    parser.add_argument("--use_batchnorm", action="store_true", help="Use batch normalization")
    parser.add_argument("--skip_connections", type=int, default=4, help="Number of skip connections")
    parser.add_argument("--auto_detect_classes", action="store_true", help="Auto-detect number of classes from dataset")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD)")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping patience")
    
    # Scheduler parameters
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "plateau", "cosine", "step"], help="LR scheduler")
    parser.add_argument("--factor", type=float, default=0.1, help="Factor for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=5, help="Patience for ReduceLROnPlateau")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Min LR for CosineAnnealingLR")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory with patches")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--use_copy_paste", action="store_true", help="Use CopyPaste augmentation")
    
    # Experiment parameters
    parser.add_argument("--run_experiment", action="store_true", help="Run hyperparameter experiment")
    
    # Saving parameters
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="unet_sentinel2", help="Model name")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args)