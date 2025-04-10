"""
Main script for land cover classification using Sentinel-2 imagery and U-Net.

This script orchestrates the entire pipeline:
1. Data preparation
2. Dataset creation
3. Model training
4. Evaluation
5. Prediction on new area
"""
import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add the current directory to the path so we can import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_preparation.check_data import main as check_data
from data_preparation.sentinel_processor import SentinelProcessor
from dataset.data_loader import SentinelDataset, create_data_loaders
from dataset.augmentation import get_train_transform, get_val_transform
from models.unet import UNetWithResnet
from models.train_utils import train_model, validate_model
from models.metrics import calculate_metrics
from prediction.predictor import LandCoverPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Land Cover Classification with Sentinel-2 and U-Net')
    
    # Data preparation arguments
    parser.add_argument('--sentinel_dir', type=str, default='data/sentinel_data',
                        help='Directory containing Sentinel-2 data')
    parser.add_argument('--ground_truth', type=str, default='data/ground_truth.tif',
                        help='Path to ground truth GeoTIFF file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--pansharp_method', type=str, default='brovey',
                        choices=['simple', 'brovey', 'hpf'],
                        help='Pansharpening method')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--encoder', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Encoder architecture for U-Net')
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip data preparation if already done')
    
    # Prediction arguments
    parser.add_argument('--predict_tile', type=str, default='T34SEH',
                        help='Tile ID for prediction (without SAFE extension)')
    parser.add_argument('--predict_dir', type=str, default='data/predict_data',
                        help='Directory containing Sentinel-2 data for prediction')
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare data for training."""
    print("\nStep 1: Data Preparation")
    
    # Check Sentinel-2 data coverage and cloud coverage
    print("\nChecking Sentinel-2 data...")
    valid_safe_dirs = check_data()
    
    if not valid_safe_dirs:
        print("No valid Sentinel-2 data found.")
        sys.exit(1)
    
    # Process Sentinel-2 data
    print("\nProcessing Sentinel-2 data...")
    processor = SentinelProcessor(output_base_dir='processed_data')
    
    dataset_info = processor.process_all(
        valid_safe_dirs,
        args.ground_truth,
        pansharpening_method='brovey',
        create_validity_masks=True
    )
    
    return dataset_info


def create_datasets(dataset_info, args):
    """Create datasets for training, validation, and testing."""
    print("\n" + "="*80)
    print("Step 2: Dataset Creation".center(80))
    print("="*80)
    
    # Get paths from dataset_info
    image_path = dataset_info['image_path']
    mask_path = dataset_info['mask_path']
    validity_mask_path = dataset_info.get('validity_mask_path')
    
    # Create transformations for train and validation
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        image_path=image_path,
        mask_path=mask_path,
        validity_mask_path=validity_mask_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, dataset_info


def train(train_loader, val_loader, dataset_info, args):
    """Train the model."""
    print("\n" + "="*80)
    print("Step 3: Model Training".center(80))
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get number of input channels and classes
    num_channels = dataset_info['image_shape'][2]
    num_classes = len(dataset_info['class_values'])
    
    print(f"Input channels: {num_channels}")
    print(f"Output classes: {num_classes}")
    
    # Create model
    model = UNetWithResnet(
        encoder_name=args.encoder,
        in_channels=num_channels,
        num_classes=num_classes
    )
    model.to(device)
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the model
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        output_dir=os.path.join(args.output_dir, 'model')
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'plots'))
    
    return model, history


def evaluate(model, test_loader, dataset_info, args):
    """Evaluate the model on the test set."""
    print("\n" + "="*80)
    print("Step 4: Model Evaluation".center(80))
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate on test set
    test_results = validate_model(
        model=model,
        data_loader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        class_values=dataset_info['class_values']
    )
    
    # Print evaluation metrics
    print("\nTest Set Metrics:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Mean IoU: {test_results['mean_iou']:.4f}")
    
    # Print per-class IoU
    print("\nPer-class IoU:")
    for i, class_iou in enumerate(test_results['class_iou']):
        if 'class_names' in dataset_info and dataset_info['class_names']:
            class_name = dataset_info['class_names'][i]
            print(f"  Class {i} ({class_name}): {class_iou:.4f}")
        else:
            print(f"  Class {i}: {class_iou:.4f}")
    
    # Create confusion matrix visualization
    if 'confusion_matrix' in test_results:
        cm = test_results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot non-normalized confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Set labels
        if 'class_names' in dataset_info and dataset_info['class_names']:
            class_names = dataset_info['class_names']
        else:
            class_names = [f'Class {i}' for i in dataset_info['class_values']]
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, 'plots', 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    return test_results


def predict_new_area(model, dataset_info, args):
    """Make predictions on a new area."""
    print("\n" + "="*80)
    print("Step 5: Prediction on New Area".center(80))
    print("="*80)
    
    # Find SAFE directory for the specified tile
    predict_safe_dirs = []
    for safe_dir in Path(args.predict_dir).glob(f"*{args.predict_tile}*.SAFE"):
        predict_safe_dirs.append(str(safe_dir))
    
    if not predict_safe_dirs:
        print(f"No SAFE directory found for tile {args.predict_tile}. Checking for zip files...")
        
        # Check for zip files
        for zip_file in Path(args.predict_dir).glob(f"*{args.predict_tile}*.zip"):
            from zipfile import ZipFile
            safe_name = zip_file.name.replace('.zip', '')
            safe_dir = Path(args.predict_dir) / safe_name
            
            if not safe_dir.exists():
                print(f"Extracting {zip_file}...")
                with ZipFile(zip_file, 'r') as zipf:
                    zipf.extractall(args.predict_dir)
                
                predict_safe_dirs.append(str(safe_dir))
    
    if not predict_safe_dirs:
        print(f"No data found for tile {args.predict_tile}. Please download the data and try again.")
        return
    
    # Get ground truth for evaluation if available
    ground_truth_path = os.path.join(args.predict_dir, f"{args.predict_tile}_ground_truth.tif")
    if not os.path.exists(ground_truth_path):
        print(f"No ground truth found at {ground_truth_path}. Will only generate predictions.")
        ground_truth_path = None
    
    # Create predictor
    predictor = LandCoverPredictor(
        model=model,
        dataset_info=dataset_info,
        output_dir=os.path.join(args.output_dir, 'predictions')
    )
    
    # Process the new area
    print("\nProcessing new area...")
    processor = SentinelProcessor(output_base_dir=os.path.join(args.output_dir, 'predict_processed'))
    
    if ground_truth_path:
        prediction_dataset_info = processor.process_all(
            predict_safe_dirs,
            ground_truth_path,
            pansharpening_method=args.pansharp_method,
            create_validity_masks=True
        )
    else:
        # Create a dummy ground truth
        from data_preparation.geospatial_utils import create_dummy_ground_truth
        dummy_gt_path = os.path.join(args.output_dir, 'predict_processed', 'dummy_gt.tif')
        
        # First, process one band to get the dimensions
        first_safe = predict_safe_dirs[0]
        img_data_path = list(Path(first_safe).glob('GRANULE/*/IMG_DATA'))[0]
        band_file = list(img_data_path.glob('*B02.jp2'))[0]
        
        # Create dummy ground truth based on the dimensions
        create_dummy_ground_truth(str(band_file), dummy_gt_path)
        
        prediction_dataset_info = processor.process_all(
            predict_safe_dirs,
            dummy_gt_path,
            pansharpening_method=args.pansharp_method,
            create_validity_masks=True
        )
    
    # Generate predictions
    print("\nGenerating predictions...")
    prediction_results = predictor.predict(
        image_path=prediction_dataset_info['image_path'],
        output_path=os.path.join(args.output_dir, 'predictions', f"{args.predict_tile}_prediction.tif"),
        patch_size=256,
        overlap=32
    )
    
    # Evaluate predictions if ground truth is available
    if ground_truth_path and ground_truth_path != dummy_gt_path:
        print("\nEvaluating predictions...")
        evaluation_results = predictor.evaluate(
            prediction_path=prediction_results['prediction_path'],
            ground_truth_path=ground_truth_path
        )
        
        print("\nPrediction Metrics:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Mean IoU: {evaluation_results['mean_iou']:.4f}")
        
        # Print per-class IoU
        print("\nPer-class IoU:")
        for i, class_iou in enumerate(evaluation_results['class_iou']):
            if 'class_names' in dataset_info and dataset_info['class_names']:
                class_name = dataset_info['class_names'][i]
                print(f"  Class {i} ({class_name}): {class_iou:.4f}")
            else:
                print(f"  Class {i}: {class_iou:.4f}")
    
    return prediction_results


def plot_training_history(history, output_dir):
    """Plot training history."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and validation IoU
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    # Plot additional metrics if available
    if 'train_acc' in history and 'val_acc' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_history.png'), dpi=300)
        plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Data preparation
    if not args.skip_data_prep:
        dataset_info = prepare_data(args)
    else:
        print("\nSkipping data preparation...")
        dataset_info_path = os.path.join(args.output_dir, 'processed_data', 'dataset_info.pt')
        if os.path.exists(dataset_info_path):
            dataset_info = torch.load(dataset_info_path)
        else:
            print(f"Dataset info not found at {dataset_info_path}. Running data preparation...")
            dataset_info = prepare_data(args)
    
    # Save dataset_info
    torch.save(dataset_info, os.path.join(args.output_dir, 'processed_data', 'dataset_info.pt'))
    
    # Step 2: Create datasets
    train_loader, val_loader, test_loader, dataset_info = create_datasets(dataset_info, args)
    
    # Step 3: Train model
    model, history = train(train_loader, val_loader, dataset_info, args)
    
    # Step 4: Evaluate model
    test_results = evaluate(model, test_loader, dataset_info, args)
    
    # Step 5: Predict on new area
    prediction_results = predict_new_area(model, dataset_info, args)
    
    # End timer
    end_time = time.time()
    
    # Print execution time
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("Execution completed".center(80))
    print("="*80)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()