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

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from dataset.augmentation import get_train_transform, get_val_transform
from dataset.class_remapper import create_class_remapper
from dataset.mask_handler import clean_mask, inspect_dataset_masks
from dataset.metrics import (
    CombinedCEDiceLoss,
    calculate_sqrt_inverse_weights,
    create_metrics,
)
from dataset.normalizers import load_sentinel2_normalizer, normalize_batch,  Sentinel2Normalizer
from dataset.patch_dataset import create_patch_data_loaders, calculate_class_pixel_counts, SentinelPatchDataset
from dataset.utils import detect_classes_from_dataset
from models.unet import UNet

import torch._dynamo
torch._dynamo.config.suppress_errors = True

"""
python run_experiment.py --data_dir F:\processed_data\training_dataset_128 --encoder_type best_performance --use_normalizer --batch_size 16 --num_epochs 5 --early_stopping 2 --save_dir F:\processed_data\model_experiments_special --wandb_project sentinel2_landcover --patch_size 128 --use_copy_paste --custom_experiment --experiment_config experiments_config.json --use_improved_sampler --use_copy_paste --num_workers 8
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In train.py, modify the data loading section:

def get_data_loaders(data_dir, batch_size=8, num_workers=4):
    """Create data loaders that respect stratified splits"""
    
    # Check if stratified split info exists
    split_info_path = os.path.join(data_dir, 'split_info.json')
    if os.path.exists(split_info_path):
        print("Loading stratified split information...")
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)
            
        # Create datasets from the stratified splits
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')
        
        # Create transforms
        train_transform = get_train_transform(p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste)
        val_transform = get_val_transform(patch_size=args.patch_size)
        
        # Create datasets using the existing directory structure
        train_dataset = SentinelPatchDataset(train_dir, transform=train_transform)
        val_dataset = SentinelPatchDataset(val_dir, transform=val_transform)
        test_dataset = SentinelPatchDataset(test_dir, transform=val_transform)
        
        # Print class distribution information
        print("\nClass distribution after stratified split:")
        for split, dist in split_info.get('class_distribution', {}).items():
            print(f"  {split}: {dist}")
    else:
        # Fall back to non-stratified loading
        print("No stratified split information found. Using regular dataset loading.")
        return create_patch_data_loaders(
            patches_dir=data_dir,
            batch_size=batch_size,
            train_transform=get_train_transform(p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste),
            val_transform=get_val_transform(patch_size=args.patch_size),
            num_workers=num_workers
        )
    
    # Create and return data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    print(f"Created data loaders with:")
    print(f"  {len(train_dataset)} training samples")
    print(f"  {len(val_dataset)} validation samples")
    print(f"  {len(test_dataset)} test samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


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
    normalizer=None,
):
    """Train for one epoch with mixed precision support and normalization."""
    model.train()
    epoch_loss = 0
    batch_count = 0

    # Reset metrics at the start of epoch
    metrics.reset()
    
    # Zero gradients at the beginning
    optimizer.zero_grad(set_to_none=True)  # Memory efficient

    with tqdm(dataloader, desc="Training") as pbar:
        for images, masks in pbar:
            batch_count += 1
            images = images.to(device, non_blocking=True)  # Non-blocking for speed

            # Apply normalization - this is the key addition
            if normalizer is not None:
                images = normalize_batch(images, normalizer, device)

            # Apply class remapping if provided
            if remapper is not None:
                masks = remapper.remap_mask(masks)

            # Clean mask to ensure valid class indices
            masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
            masks = masks.long().to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp and scaler is not None:
                with autocast(device_type="cuda", dtype=torch.float32):
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Clip gradients to prevent explosion
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()

                # Clip gradients
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            # Update metrics
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
    epoch_loss /= len(dataloader)

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
    normalizer=None,
):
    """Validate the model with mixed precision support and normalization."""
    model.eval()
    val_loss = 0

    # Reset metrics at the start of validation
    metrics.reset()

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for images, masks in pbar:
                images = images.to(device)

                # Apply normalization - this is the key addition
                if normalizer is not None:
                    images = normalize_batch(images, normalizer, device)
                else:
                    # Simple check for high-value Sentinel-2 data
                    if images.max() > 100:
                        images = images / 10000.0

                # Apply class remapping if provided
                if remapper is not None:
                    masks = remapper.remap_mask(masks)

                # Clean mask to ensure valid class indices
                masks = clean_mask(masks, num_classes=num_classes, ignore_index=-100)
                masks = masks.long().to(device)

                # Mixed precision inference
                if use_amp:
                    with autocast(device_type="cuda", dtype=torch.float32):
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
    normalizer=None,
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
            normalizer,
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
            normalizer,
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


def setup_data_and_preprocessing(args):
    """
    Setup data loaders and preprocessing components.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Dictionary containing data loaders, normalizer, class remapper, and other setup info
    """
    # Setup data augmentation
    train_transform = get_train_transform(
        p=0.5, patch_size=args.patch_size, use_copy_paste=args.use_copy_paste
    )
    val_transform = get_val_transform(patch_size=args.patch_size)
    
    # Load or create normalizer
    normalizer = None
    if args.use_normalizer:
        normalizer_path = os.path.join(args.data_dir, "normalizer.pkl")
        if os.path.exists(normalizer_path):
            print(f"Loading Sentinel-2 normalizer from {normalizer_path}")
            try:
                normalizer = load_sentinel2_normalizer(normalizer_path)
                print(f"Loaded normalizer with method: {normalizer.method}")
                
                # Verify normalizer is properly configured
                if not normalizer.is_fitted or normalizer.band_means is None:
                    print("Normalizer missing band statistics. Fitting on dataset...")
                    normalizer.fit(args.data_dir)
                    
                    # Save updated normalizer
                    with open(normalizer_path, "wb") as f:
                        pickle.dump(normalizer, f)
                    print("Saved updated normalizer with band-specific statistics")
            except Exception as e:
                print(f"Error loading normalizer: {e}")
                print("Creating new band-specific normalizer")
                normalizer = Sentinel2Normalizer(method="band_specific")
                normalizer.fit(args.data_dir)
                
                # Save new normalizer
                with open(normalizer_path, "wb") as f:
                    pickle.dump(normalizer, f)
        else:
            print("Creating new band-specific normalizer...")
            normalizer = Sentinel2Normalizer(method="band_specific")
            normalizer.fit(args.data_dir)
            
            # Save new normalizer
            with open(normalizer_path, "wb") as f:
                pickle.dump(normalizer, f)

    # Create data loaders with optimized settings
    data_loaders = create_patch_data_loaders(
        patches_dir=args.data_dir,
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=args.num_workers,
        use_rare_class_sampler=True,
        use_improved_sampler=args.use_improved_sampler,
    )

    print(f"Training on {len(data_loaders['train'].dataset)} samples")
    print(f"Validating on {len(data_loaders['val'].dataset)} samples")

    # Inspect dataset masks (using reduced sample size for efficiency)
    print("\nInspecting dataset for mask issues...")
    train_inspection = inspect_dataset_masks(
        data_loaders['train'].dataset, 
        num_samples=min(50, len(data_loaders['train'].dataset))
    )

    # Initialize class remapper if needed
    class_remapper = None
    remapper_cache_path = os.path.join(args.data_dir, "remapper_cache.pkl")
    
    if train_inspection.get("sparse_indices", False):
        print("\nDetected sparse class indices. Looking for cached remapper first...")
        if os.path.exists(remapper_cache_path):
            try:
                with open(remapper_cache_path, 'rb') as f:
                    class_remapper = pickle.load(f)
                print("Loaded class remapper from cache")
            except Exception:
                print("Failed to load cached remapper, creating new one")
                class_remapper = create_class_remapper(
                    data_loaders["train"].dataset, num_samples=100
                )
                # Cache the remapper
                with open(remapper_cache_path, 'wb') as f:
                    pickle.dump(class_remapper, f)
        else:
            print("Creating class remapper and caching for future use...")
            class_remapper = create_class_remapper(
                data_loaders["train"].dataset, num_samples=100
            )
            # Cache the remapper
            with open(remapper_cache_path, 'wb') as f:
                pickle.dump(class_remapper, f)

        # Update number of classes based on remapped classes
        args.num_classes = class_remapper.num_classes
        print(f"Using {args.num_classes} classes after remapping")
        
    # Auto-detect classes if needed
    elif args.auto_detect_classes or args.num_classes is None:
        print("Auto-detecting classes from dataset...")
        # Check for cached class info
        class_info_path = os.path.join(args.data_dir, "class_info_cache.json")
        if os.path.exists(class_info_path):
            try:
                with open(class_info_path, 'r') as f:
                    class_info = json.load(f)
                print("Loaded class info from cache")
            except Exception:
                print("Failed to load cached class info, detecting classes...")
                class_info = detect_classes_from_dataset(
                    data_loaders["train"].dataset, sample_size=100
                )
                # Cache the class info
                with open(class_info_path, 'w') as f:
                    json.dump(class_info, f)
        else:
            class_info = detect_classes_from_dataset(
                data_loaders["train"].dataset, sample_size=100
            )
            # Cache the class info
            with open(class_info_path, 'w') as f:
                json.dump(class_info, f)
                
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
    
    return {
        "data_loaders": data_loaders,
        "normalizer": normalizer,
        "class_remapper": class_remapper,
        "train_inspection": train_inspection
    }


def compute_class_weights(dataset, num_classes, cache_dir=None, sample_size=500):
    """
    Compute class weights based on class distribution for loss weighting.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        Dataset to analyze
    num_classes : int
        Number of classes
    cache_dir : str or None
        Directory to cache weights
    sample_size : int
        Number of samples to analyze
        
    Returns:
    --------
    torch.Tensor
        Class weights tensor
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "class_weights_cache.pt")
        
    # Try to load cached weights
    if cache_path and os.path.exists(cache_path):
        try:
            class_weights = torch.load(cache_path)
            print("Loaded cached class weights")
            return class_weights
        except Exception:
            print("Failed to load cached weights, computing new ones...")
    
    # Calculate class distribution
    print("Calculating class distribution for loss weighting...")
    class_counts, total_pixels = calculate_class_pixel_counts(dataset, num_samples=sample_size)
    
    # Display class distribution
    print(f"Total pixels analyzed: {total_pixels}")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count:,} pixels ({count/total_pixels*100:.2f}%)")
    
    # Calculate sqrt-inverse class weights
    class_weights = calculate_sqrt_inverse_weights(class_counts)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Cache weights if path provided
    if cache_path:
        torch.save(class_weights, cache_path)
        
    return class_weights


def setup_model_and_training(args, config=None, data_loaders=None, class_remapper=None):
    """
    Setup model, optimizer, scheduler, and loss function based on config.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    config : dict or None
        Hyperparameter configuration (for experiments)
    data_loaders : dict or None
        Data loaders dict containing 'train' and 'val'
    class_remapper : ClassRemapper or None
        Class remapper object if used
        
    Returns:
    --------
    dict
        Dictionary containing model, criterion, optimizer, scheduler, and other config
    """
    # Use config values if provided, otherwise use args
    if config is None:
        config = {
            "learning_rate": args.learning_rate,
            "encoder_lr_factor": args.encoder_lr_factor,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "freeze_backbone": getattr(args, "freeze_backbone", True),
            "progressive_unfreeze": getattr(args, "progressive_unfreeze", False),
        }
    
    # Create model
    print(f"Creating U-Net model with {args.num_classes} output classes...")
    model = UNet(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        encoder_type=args.encoder_type,
        use_batchnorm=args.use_batchnorm,
        skip_connections=args.skip_connections,
        freeze_backbone=config.get("freeze_backbone", True),
    ).to(device)
    
    # Compute class weights for loss function
    class_weights = compute_class_weights(
        dataset=data_loaders["train"].dataset if data_loaders else None,
        num_classes=args.num_classes,
        cache_dir=args.data_dir,
        sample_size=500
    )
    
    # Create loss function
    ce_weight = 0.5  # Default to balanced weights
    dice_weight = 0.5
    if hasattr(args, "ce_weight") and hasattr(args, "dice_weight"):
        ce_weight = args.ce_weight
        dice_weight = args.dice_weight
        
    criterion = CombinedCEDiceLoss(
        weights=class_weights,
        num_classes=args.num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight
    )
    print(f"Using combined CE+Dice loss (CE: {ce_weight}, Dice: {dice_weight}) with class weights")
    
    # Create optimizer
    optimizer = create_optimizer_with_differential_lr(
        model=model,
        base_lr=config["learning_rate"],
        encoder_lr_factor=config.get("encoder_lr_factor", 0.1),
        weight_decay=config.get("weight_decay", 1e-4),
        optimizer_type=config.get("optimizer", "adam"),
    )
    
    # Create scheduler
    scheduler = None
    scheduler_type = config.get("scheduler", "onecycle")
    
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs, 
            eta_min=getattr(args, "min_lr", 1e-6)
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=getattr(args, "step_size", 10), 
            gamma=getattr(args, "gamma", 0.1)
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=getattr(args, "factor", 0.1), 
            patience=getattr(args, "patience", 5)
        )
    elif scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                config["learning_rate"] * config.get("encoder_lr_factor", 0.1),
                config["learning_rate"],
                config["learning_rate"],
            ],
            total_steps=args.num_epochs * len(data_loaders["train"]) if data_loaders else None,
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
    
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "unfreeze_schedule": unfreeze_schedule,
        "config": config
    }


def init_wandb(args, class_remapper=None, train_inspection=None, config=None):
    """
    Initialize Weights & Biases with optimized settings.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    class_remapper : ClassRemapper or None
        Class remapper object if used
    train_inspection : dict or None
        Dataset inspection results
    config : dict or None
        Custom configuration dict
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if not args.wandb_project:
        return None
        
    print(f"Initializing Weights & Biases project with optimized settings: {args.wandb_project}")
    
    if config is None:
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

    # Initialize with optimized settings
    os.environ["WANDB_CONSOLE"] = "off"  # Reduce console output
    wandb.init(
        project=args.wandb_project, 
        config=config,
        # Lower log frequency to reduce overhead
        settings=wandb.Settings(log_internal=60)  # Log internals every 60 seconds
    )
    
    # Log dataset info
    if train_inspection:
        wandb.run.summary["sparse_indices"] = train_inspection.get("sparse_indices", False)
        wandb.run.summary["num_classes"] = args.num_classes
    
    return config


def run_hyperparameter_experiment(args, experiment_configs=None):
    """Run multiple training experiments with different hyperparameters."""
    # Create experiment directory
    experiment_dir = os.path.join(args.save_dir, "experiments")
    os.makedirs(experiment_dir, exist_ok=True)

    # Setup data and preprocessing
    setup_data = setup_data_and_preprocessing(args)
    data_loaders = setup_data["data_loaders"]
    normalizer = setup_data["normalizer"]
    class_remapper = setup_data["class_remapper"]
    train_inspection = setup_data["train_inspection"]

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
    
    # Initialize wandb
    wandb_project = getattr(args, "wandb_project", None)

    # Pre-compute class weights once
    class_weights = compute_class_weights(
        dataset=data_loaders["train"].dataset,
        num_classes=args.num_classes,
        cache_dir=args.data_dir
    )

    # Loop through hyperparameter combinations
    for i, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiment_configs)}: {config['name']}")
        print(f"{'='*80}\n")
        
        # Setup model and training components with this config
        training_setup = setup_model_and_training(
            args=args,
            config=config,
            data_loaders=data_loaders,
            class_remapper=class_remapper
        )
        
        model = training_setup["model"]
        criterion = training_setup["criterion"]
        optimizer = training_setup["optimizer"]
        scheduler = training_setup["scheduler"]
        unfreeze_schedule = training_setup["unfreeze_schedule"]

        # Create experiment-specific wandb config
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

        # Save directory for this experiment
        exp_save_dir = os.path.join(experiment_dir, config["name"])
        os.makedirs(exp_save_dir, exist_ok=True)

        # Train model with this configuration
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
            normalizer=normalizer,
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
        wandb.config.update(
            {
                "using_normalizer": normalizer is not None,
                "normalizer_type": (
                    getattr(normalizer, "method", "default") if normalizer else "none"
                ),
            }
        )
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
    """Main training function with optimizations for faster training."""
    # Enable cuDNN benchmark mode for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {device}")

    # Optimize CPU thread settings for faster data loading
    if args.num_workers < 12:
        args.num_workers = min(8, os.cpu_count())
        print(f"Setting num_workers to {args.num_workers} for better performance")

    # Setup data and preprocessing - unified function
    setup_data = setup_data_and_preprocessing(args)
    data_loaders = setup_data["data_loaders"]
    normalizer = setup_data["normalizer"]
    class_remapper = setup_data["class_remapper"]
    train_inspection = setup_data["train_inspection"]

    # Initialize wandb if not in experiment mode
    config = None
    if args.wandb_project and not args.run_experiment:
        config = init_wandb(args, class_remapper, train_inspection)

    # Run hyperparameter experiment if requested
    if args.run_experiment:
        print("Running hyperparameter experiment with optimized settings...")
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

    # Setup model and training components - unified function
    training_setup = setup_model_and_training(
        args=args,
        data_loaders=data_loaders,
        class_remapper=class_remapper
    )
    
    model = training_setup["model"]
    criterion = training_setup["criterion"]
    optimizer = training_setup["optimizer"]
    scheduler = training_setup["scheduler"]
    unfreeze_schedule = training_setup["unfreeze_schedule"]
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Train model with optimized settings
    print("Starting training with optimized settings...")
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
        config=config,
        num_classes=args.num_classes,
        remapper=class_remapper,
        use_amp=getattr(args, "use_amp", True),
        unfreeze_schedule=unfreeze_schedule,
        normalizer=normalizer,
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
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
    parser.add_argument(
        "--use_normalizer",
        action="store_true",
        default=True,
        help="Use the saved Sentinel-2 normalizer from dataset directory",
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
        "--num_workers", type=int, default=2, help="Number of workers for data loading"
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
    
    parser.add_argument(
        "--use_improved_sampler",
        action="store_true",
        default=True,  # Making it true by default
        help="Use improved rare class sampling with dynamic weighting (default: True)"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(args)
