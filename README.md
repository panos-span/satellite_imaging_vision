# Sentinel-2 Land Cover Mapping with Deep Learning

## Project Overview

This repository contains a comprehensive deep learning pipeline for large-scale land cover mapping using Sentinel-2 satellite imagery. The project implements a U-Net architecture with transfer learning from pretrained ResNet models to perform semantic segmentation of multispectral satellite data into distinct land cover classes.

## Key Features

- Complete preprocessing pipeline for Sentinel-2 Level-1C imagery
- Optimized U-Net architecture with ResNet-50 encoder using transfer learning
- Efficient patch-based training and prediction (256×256 pixels with 96-pixel overlap)
- Advanced data augmentation techniques for multispectral satellite imagery
- Specialized loss functions for handling class imbalance in land cover data
- Memory-efficient prediction pipeline for large-area mapping with overlap handling
- Sophisticated post-processing with spatial context refinement and small object removal

## Project Structure

```
.
├── dataset/                          # Dataset handling modules
│   ├── __init__.py
│   ├── augmentation.py               # Data augmentation functions
│   ├── class_remapper.py             # Class mapping utilities
│   ├── copy_paste_augmentation.py    # Copy-paste augmentation technique
│   ├── data_loader.py                # Data loading utilities
│   ├── data_validation.py            # Dataset validation tools
│   ├── dataset_splitter.py           # Dataset splitting functions
│   ├── mask_handler.py               # Segmentation mask processing
│   ├── metrics.py                    # Evaluation metrics
│   ├── normalizers.py                # Data normalization utilities
│   ├── patch_dataset.py              # Patch-based dataset implementation
│   └── utils.py                      # Utility functions
│
├── data_preparation/                  # Data preprocessing utilities
│   ├── __init__.py
│   ├── check_data.py                 # Data integrity checking
│   ├── cloud_coverage.py             # Cloud coverage assessment
│   ├── geospatial_utils.py           # Geospatial processing utilities
│   ├── pansharpening.py              # Image pansharpening implementation
│   ├── sentinel_processor.py         # Sentinel-2 data processor
│   ├── validity_masks.py             # Validity mask generation
│   ├── visualization.py              # Data visualization utilities
│   ├── output/                       # Output directory for processed data
│   └── processed_data/               # Storage for processed datasets
│
├── models/                            # Model architecture definitions
│   ├── __init__.py
│   ├── blocks.py                     # Neural network building blocks
│   ├── encoder.py                    # Encoder architecture
│   ├── receptive_field.py            # Receptive field utilities
│   └── unet.py                       # U-Net model implementation
│
├── data_preparation.py               # Main script for data preparation pipeline
├── prepare_dataset.py                # Dataset preparation with class remapping
├── run_experiment.py                 # Hyperparameter optimization framework
├── train.py                          # Complete training pipeline
├── prediction.py                     # Advanced prediction with post-processing
├── experiments_config.json           # Configuration for hyperparameter experiments
│
├── requirements.txt                   # Dependencies
├── LICENSE                            # License information
└── README.md                          # This file
```

## Installation


### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16+ GB RAM for large area prediction

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/username/sentinel2-landcover-mapping.git
   cd sentinel2-landcover-mapping
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download sample data (optional):
   ```
   python scripts/download_sample_data.py
   ```

## Usage

### Data Preprocessing

To preprocess Sentinel-2 data and ground truth:

```
python scripts/preprocess.py --sentinel2-dir /path/to/sentinel2 --ground-truth /path/to/ground_truth.tif --output-dir /path/to/output
```

### Model Training

To train the model with default parameters:

```
python scripts/train.py --data-dir /path/to/processed_data --output-dir /path/to/models
```

Advanced training configuration:

```
python scripts/train.py --data-dir /path/to/processed_data --output-dir /path/to/models --model resnet50 --batch-size 8 --epochs 50 --lr 5e-4 --patch-size 256 --patch-overlap 96 --skip-connections 4 --encoder-lr-factor 0
```

### Prediction

To apply the trained model to a new Sentinel-2 image:

```
python scripts/predict.py --model-path /path/to/best_model.pth --input-image /path/to/sentinel2_image.tif --output-map /path/to/output_landcover.tif
```

## Dataset

The model is designed to work with Sentinel-2 Level-1C (Top-of-Atmosphere reflectance) products. The training dataset consists of:

- **Input**: 13-band Sentinel-2 imagery at 10m resolution (after pansharpening)
- **Ground Truth**: Land cover classification maps with the following classes:
  - Class 0: Background/No Data
  - Class 10: Agricultural Areas
  - Class 20: Forests
  - Class 30: Semi-natural Areas
  - Class 40: Urban Areas
  - Class 50: Wetlands
  - Class 60: Water Bodies
  - Class 80: Natural Vegetation
  - Class 90: Other

## Model Architecture

The implemented architecture is a U-Net with:
- Encoder: Pre-trained ResNet-50 backbone (modified to accept 13 input bands)
- Skip connections: Minimum of 4 connections from encoder to decoder
- Decoder: Transposed convolutions for upsampling with concatenated skip features
- Output: Pixel-wise classification across land cover classes

Training employs a combined loss function with weighted cross-entropy and Dice loss to handle class imbalance.

## Patch-Based Processing Strategy

The model employs an optimized patch-based approach for both training and prediction:
- Patch size: 256×256 pixels
- Overlap: 96 pixels between adjacent patches
- Weighted blending: Radial weighting scheme for seamless merging of predictions

This configuration balances computational efficiency with the need for spatial context, and ensures consistent performance across large geographic areas.

## Results

On the Achaia test region (T34SEH tile), the model achieves:
- Accuracy: 0.77
- Precision: 0.77
- Recall: 0.77
- F1 Score: 0.73
- IoU: 0.63

Class-specific performance highlights:
- Natural Vegetation (Class 80): 99% accuracy
- Agricultural Areas (Class 10): 90% accuracy
- Urban Areas (Class 40): 69% accuracy
