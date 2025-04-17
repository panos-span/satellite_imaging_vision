"""
Fixed encoder implementation for U-Net architecture with Sentinel-2 pretrained weights.

This module provides ResNet-based encoder implementations (ResNet50 or ResNet18)
that use Sentinel-2 specific pretrained weights from the TorchGeo library, which
significantly outperform general ImageNet pretrained weights for remote sensing tasks.

The encoder part of the U-Net consists exclusively of layers from a pre-trained
ResNet model as required for proper transfer learning.
"""

import torch.nn as nn
import torchgeo.models as tgm


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder (ResNet50 or ResNet18) with Sentinel-2 specific pretrained weights.

    This encoder consists exclusively of layers from a pre-trained ResNet model,
    as required for proper transfer learning in the U-Net architecture.

    Parameters:
    -----------
    model_type : str
        Type of ResNet model to use ('resnet18' or 'resnet50')
    weight_name : str
        Name of the TorchGeo weight to use
    in_channels : int
        Number of input channels (should match the pretrained model's channels)
    """

    def __init__(self, model_type="resnet50", weight_name=None, in_channels=13):
        super().__init__()

        # Ensure we're using a ResNet model
        if model_type not in ["resnet18", "resnet50"]:
            raise ValueError(
                f"Only 'resnet18' and 'resnet50' are supported, got: {model_type}"
            )

        self.model_type = model_type
        self.weight_name = weight_name
        self.in_channels = in_channels

        # Get the appropriate weights
        if weight_name is None:
            # Use default weights based on model type and input channels
            weight_name = self._get_default_weights()

        # Create the backbone using TorchGeo
        self.backbone, self.weights = self._create_backbone(tgm)

        # Extract feature info for skip connections
        self.feature_channels = self._extract_feature_channels()

    def _get_default_weights(self):
        """Get default weights based on model type and input channels."""
        # Default to best performing models for ResNet
        if self.model_type == "resnet18":
            return (
                "SENTINEL2_ALL_MOCO" if self.in_channels == 13 else "SENTINEL2_RGB_MOCO"
            )
        else:  # resnet50
            return (
                "SENTINEL2_ALL_MOCO" if self.in_channels == 13 else "SENTINEL2_RGB_MOCO"
            )

    def _create_backbone(self, tgm):
        """Create the ResNet backbone with appropriate weights."""
        if self.model_type == "resnet18":
            weights_enum = getattr(tgm.ResNet18_Weights, self.weight_name)
            backbone = tgm.resnet18(weights=weights_enum)
        else:  # resnet50
            weights_enum = getattr(tgm.ResNet50_Weights, self.weight_name)
            backbone = tgm.resnet50(weights=weights_enum)

        return backbone, weights_enum

    def _extract_feature_channels(self):
        """Extract feature channel information for skip connections."""
        if hasattr(self.backbone, "inplanes"):
            base_width = getattr(self.backbone, "inplanes", 64)
        else:
            base_width = 64

        if self.model_type == "resnet18":
            return [
                base_width,
                base_width,
                base_width * 2,
                base_width * 4,
                base_width * 8,
            ]
        else:  # resnet50
            return [
                base_width,
                base_width * 4,
                base_width * 8,
                base_width * 16,
                base_width * 32,
            ]

    def forward(self, x):
        """
        Forward pass through the ResNet encoder.

        Extracts intermediate features for skip connections between
        encoder and decoder in the U-Net architecture.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        dict
            Dictionary with intermediate features for skip connections
        """
        features = {}

        # Initial layers - adapted to match torchgeo's ResNet structure
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        # In torchgeo's ResNet, relu might be accessed differently
        if hasattr(self.backbone, "relu"):
            x = self.backbone.relu(x)
        else:
            # For models where relu might be part of the activation function
            x = nn.functional.relu(x, inplace=True)
        
        features["stage0"] = x

        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        features["stage1"] = x

        x = self.backbone.layer2(x)
        features["stage2"] = x

        x = self.backbone.layer3(x)
        features["stage3"] = x

        x = self.backbone.layer4(x)
        features["stage4"] = x

        return features


# Dictionary of recommended ResNet models for different purposes
SENTINEL2_RECOMMENDED_MODELS = {
    "best_performance": ("resnet50", "SENTINEL2_ALL_MOCO"),  # 91.8% on BigEarthNet
    "all_bands": (
        "resnet50",
        "SENTINEL2_ALL_DINO",
    ),  # Good performance with all 13 bands
    "lightweight": (
        "resnet18",
        "SENTINEL2_ALL_MOCO",
    ),  # Smaller model, still good performance
    "balanced": (
        "resnet50",
        "SENTINEL2_ALL_DINO",
    ),  # Good balance of size and performance
}


def create_encoder(encoder_type="best_performance", in_channels=13):
    """
    Create a ResNet encoder with Sentinel-2 specific pretrained weights.

    Parameters:
    -----------
    encoder_type : str
        Type of encoder to create ('best_performance', 'all_bands', 'lightweight', 'balanced')
        Or a specific model type and weight name tuple: ('resnet50', 'SENTINEL2_ALL_MOCO')
    in_channels : int
        Number of input channels

    Returns:
    --------
    ResNetEncoder
        ResNet encoder with Sentinel-2 specific pretrained weights
    """
    if isinstance(encoder_type, tuple) and len(encoder_type) == 2:
        # Custom model and weight specification
        model_type, weight_name = encoder_type
    else:
        # Use recommended model for the specified type
        model_type, weight_name = SENTINEL2_RECOMMENDED_MODELS.get(
            encoder_type, SENTINEL2_RECOMMENDED_MODELS["best_performance"]
        )

    # Ensure we're using a ResNet model
    if model_type not in ["resnet18", "resnet50"]:
        raise ValueError(
            "Only 'resnet18' and 'resnet50' are supported as per requirements"
        )

    # Adjust channels if needed
    if "RGB" in weight_name and in_channels != 3:
        in_channels = 3
    elif "MS" in weight_name and in_channels != 9:
        in_channels = 9
    elif "ALL" in weight_name and in_channels != 13:
        in_channels = 13

    return ResNetEncoder(
        model_type=model_type, weight_name=weight_name, in_channels=in_channels
    )


def list_available_encoders():
    """
    List available ResNet encoder types and their characteristics.

    Returns:
    --------
    dict
        Dictionary of encoder types and their characteristics
    """
    encoders = {
        "best_performance": {
            "model": "ResNet50",
            "weights": "SENTINEL2_ALL_MOCO",
            "channels": 13,
            "performance": "Excellent (91.8% on BigEarthNet)",
        },
        "all_bands": {
            "model": "ResNet50",
            "weights": "SENTINEL2_ALL_DINO",
            "channels": 13,
            "performance": "Excellent (90.7% on BigEarthNet)",
        },
        "lightweight": {
            "model": "ResNet18",
            "weights": "SENTINEL2_ALL_MOCO",
            "channels": 13,
            "performance": "Good",
        },
        "balanced": {
            "model": "ResNet50",
            "weights": "SENTINEL2_ALL_DINO",
            "channels": 13,
            "performance": "Excellent (90.7% on BigEarthNet)",
        },
    }

    return encoders