"""
U-Net implementation for Sentinel-2 imagery segmentation with TorchGeo pretrained weights.

This module provides a U-Net architecture specifically designed for Sentinel-2 imagery
with 13 input channels at 10m spatial resolution. Key features:

1. Transfer Learning: The encoder part consists exclusively of layers from a pre-trained
   ResNet model (ResNet50 or ResNet18) with Sentinel-2 specific weights from TorchGeo.

2. Skip Connections: The architecture includes at least 2 skip connections between the
   encoder and decoder for transferring features of specific scales.

3. Differential Learning Rates: Enables different learning rates for different parts of the model.

4. Accurate Receptive Field Calculation: Implements precise calculation of the model's receptive field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import resnet18, resnet50
from torchgeo.models.resnet import ResNet18_Weights, ResNet50_Weights

from .blocks import UpBlock


def calculate_receptive_field(model_type="resnet50"):
    """
    Calculate the theoretical receptive field of the U-Net architecture.

    Parameters:
    -----------
    model_type : str
        The type of ResNet backbone ('resnet18' or 'resnet50')

    Returns:
    --------
    dict
        Dictionary with receptive field size at different stages
    """
    # Initialize tracking variables
    rf = 1  # Initial receptive field is 1 pixel
    jump = 1  # Initial jump is 1 pixel
    layer_info = []

    # Define the ResNet architecture parameters
    if "resnet18" in model_type.lower():
        # ResNet18 architecture definition
        layers = [
            # name, kernel, stride, dilation, repetitions
            ("conv1", 7, 2, 1, 1),
            ("maxpool", 3, 2, 1, 1),
            ("layer1", 3, 1, 1, 4),  # 2 blocks × 2 convs each = 4 convs
            ("layer2.0", 3, 2, 1, 2),  # First block with stride 2
            ("layer2.1", 3, 1, 1, 2),  # Second block
            ("layer3.0", 3, 2, 1, 2),  # First block with stride 2
            ("layer3.1", 3, 1, 1, 2),  # Second block
            ("layer4.0", 3, 2, 1, 2),  # First block with stride 2
            ("layer4.1", 3, 1, 1, 2),  # Second block
        ]
    else:  # ResNet50
        # ResNet50 architecture definition (bottleneck blocks)
        layers = [
            # name, kernel, stride, dilation, repetitions
            ("conv1", 7, 2, 1, 1),
            ("maxpool", 3, 2, 1, 1),
            ("layer1", 3, 1, 1, 3),  # 3 bottleneck blocks, focusing on the 3×3 conv
            ("layer2.0", 3, 2, 1, 1),  # First block with stride 2
            ("layer2", 3, 1, 1, 3),  # Remaining blocks
            ("layer3.0", 3, 2, 1, 1),  # First block with stride 2
            ("layer3", 3, 1, 1, 5),  # Remaining blocks
            ("layer4.0", 3, 2, 1, 1),  # First block with stride 2
            ("layer4", 3, 1, 1, 2),  # Remaining blocks
        ]

    # Calculate receptive field layer by layer
    for name, kernel, stride, dilation, reps in layers:
        for i in range(reps):
            old_jump = jump
            old_rf = rf

            # Update jump and receptive field
            jump *= stride if i == 0 else 1  # Apply stride only for first repetition
            rf += (kernel - 1) * old_jump * dilation

            layer_info.append(
                {
                    "layer": f"{name}{'.' + str(i) if reps > 1 else ''}",
                    "rf": rf,
                    "jump": jump,
                }
            )

    # Add decoder contributions
    # Each decoder block includes multiple 3×3 convolutions
    decoder_layers = [
        ("decoder1", 3, 1, 1, 2),  # First decoder block (2 convs)
        ("decoder2", 3, 1, 1, 2),
        ("decoder3", 3, 1, 1, 2),
        ("decoder4", 3, 1, 1, 2),
        ("decoder5", 3, 1, 1, 2),
    ]

    for name, kernel, stride, dilation, reps in decoder_layers:
        for i in range(reps):
            old_rf = rf
            rf += (kernel - 1) * jump * dilation

            layer_info.append({"layer": f"{name}.conv{i+1}", "rf": rf, "jump": jump})

    # Calculate effective receptive field (typically 2/3 of theoretical)
    effective_rf = int(rf * 2 / 3)

    return {
        "theoretical_rf": rf,
        "effective_rf": effective_rf,
        "layer_info": layer_info,
        "backbone": model_type,
    }


class Encoder(nn.Module):
    """
    Encoder for U-Net using transfer learning from TorchGeo's pretrained ResNet models.

    Based on the confirmed structure of TorchGeo's ResNet implementation.

    Parameters:
    -----------
    encoder_type : str
        Type of encoder to use ('resnet50', 'resnet18', 'best_performance', etc.)
    in_channels : int
        Number of input channels (default: 13 for Sentinel-2)
    freeze_backbone : bool
        Whether to freeze the backbone layers for true transfer learning (default: True)
    """

    def __init__(
        self, encoder_type="best_performance", in_channels=13, freeze_backbone=True
    ):
        super().__init__()
        self.encoder_type = encoder_type

        # Select appropriate pretrained model based on encoder type
        if "resnet18" in str(encoder_type).lower() or encoder_type == "lightweight":
            # Use ResNet18 with weights pretrained on Sentinel-2 imagery
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            self.backbone = resnet18(weights=weights)
            self.feature_channels = [64, 64, 128, 256, 512]
        else:  # default to ResNet50 for 'best_performance', 'all_bands', 'balanced'
            # Use ResNet50 weights with good performance on BigEarthNet
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            self.backbone = resnet50(weights=weights)
            self.feature_channels = [64, 256, 512, 1024, 2048]

        # Validate input channels match the pretrained model's expectation
        if in_channels != 13:
            raise ValueError(
                f"TorchGeo's Sentinel-2 pretrained models expect 13 channels, got {in_channels}"
            )

        # Freeze backbone layers if requested (transfer learning approach)
        if freeze_backbone:
            self._freeze_backbone()
            print("Transfer learning: backbone layers frozen")
        else:
            print("Fine-tuning: all encoder layers will be updated")

    def _freeze_backbone(self):
        """Freeze backbone layers to preserve pretrained features"""
        # Freeze all backbone parameters
        for name, param in self.backbone.named_parameters():
            # Skip the 'fc' layer parameters as we don't use it
            if not name.startswith("fc."):
                param.requires_grad = False

        # Optionally unfreeze the last layer for fine-tuning
        for name, param in self.backbone.named_parameters():
            if name.startswith("layer4."):
                param.requires_grad = True

        print("Backbone frozen except for layer4")

    def forward(self, x):
        """
        Forward pass through the encoder, extracting features at different scales.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        dict
            Dictionary with feature maps at different stages of the encoder
        """
        features = {}

        # Initial convolution block
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)  # Note: It's 'act1' not 'relu' in TorchGeo's model
        features["stage0"] = x

        # Apply max pooling
        x = self.backbone.maxpool(x)

        # ResNet blocks - proceed layer by layer to extract features
        x = self.backbone.layer1(x)
        features["stage1"] = x

        x = self.backbone.layer2(x)
        features["stage2"] = x

        x = self.backbone.layer3(x)
        features["stage3"] = x

        x = self.backbone.layer4(x)
        features["stage4"] = x

        return features


class Decoder(nn.Module):
    """
    Decoder for U-Net architecture that progressively upsamples and combines
    encoder features through skip connections.

    Parameters:
    -----------
    encoder_channels : list
        List of channel sizes from the encoder stages
    decoder_channels : list
        List of channel sizes for each decoder stage
    use_batchnorm : bool
        Whether to use batch normalization
    skip_connections : int
        Number of skip connections to use from the encoder
    use_bilinear : bool
        Whether to use bilinear upsampling (True) or transposed convolution (False)
    """

    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
        skip_connections=4,
        use_bilinear=True,
    ):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.skip_connections = min(skip_connections, len(encoder_channels) - 1)

        # Create upsampling blocks
        self.up_blocks = nn.ModuleList()

        # Calculate number of decoder blocks needed
        num_blocks = len(decoder_channels)

        for i in range(num_blocks):
            # Input channels come from either bottleneck or previous decoder block
            in_channels = encoder_channels[-1] if i == 0 else decoder_channels[i - 1]

            # Skip connection channels (0 if no skip connection for this stage)
            use_skip = i < self.skip_connections
            skip_idx = -(i + 2) if i < len(encoder_channels) - 1 else 0
            skip_channels = encoder_channels[skip_idx] if use_skip else 0

            # Output channels for this decoder stage
            out_channels = decoder_channels[i]

            # Create up block
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_batchnorm=use_batchnorm,
                    bilinear=use_bilinear,
                )
            )

    def forward(self, encoder_features):
        """
        Forward pass through the decoder with skip connections.

        Parameters:
        -----------
        encoder_features : dict
            Dictionary with feature maps from the encoder

        Returns:
        --------
        torch.Tensor
            Output tensor from the decoder
        """
        # Convert encoder features to a list from deepest to shallowest
        features_list = [
            encoder_features[f"stage{i}"] for i in range(len(encoder_features))
        ]
        features_list.reverse()  # Deepest features first

        # Start with bottleneck feature
        x = features_list[0]

        # Apply decoder blocks with skip connections
        for i, up_block in enumerate(self.up_blocks):
            # Get appropriate skip connection if applicable
            skip = None
            if i < self.skip_connections and i + 1 < len(features_list):
                skip = features_list[i + 1]

            # Pass through decoder block
            x = up_block(x, skip)

        return x


class UNet(nn.Module):
    """
    U-Net architecture for Sentinel-2 imagery segmentation using transfer learning.

    This model uses TorchGeo's pretrained ResNet backbones specifically designed
    for multi-spectral Sentinel-2 imagery, allowing effective transfer learning.

    Parameters:
    -----------
    in_channels : int
        Number of input channels (default: 13 for Sentinel-2)
    num_classes : int
        Number of output classes for segmentation
    encoder_type : str
        Type of encoder to use ('resnet50', 'resnet18', 'best_performance', etc.)
    decoder_channels : list
        Number of channels in each decoder stage
    use_batchnorm : bool
        Whether to use batch normalization
    skip_connections : int
        Number of skip connections to use (minimum 2)
    freeze_backbone : bool
        Whether to freeze the backbone for transfer learning
    """

    def __init__(
        self,
        in_channels=13,
        num_classes=1,
        encoder_type="best_performance",
        decoder_channels=[256, 128, 64, 32, 16],
        use_batchnorm=True,
        skip_connections=4,
        freeze_backbone=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Ensure we have at least 2 skip connections
        if skip_connections < 2:
            raise ValueError(
                f"At least 2 skip connections are required, got {skip_connections}"
            )

        # Create encoder with transfer learning
        self.encoder = Encoder(
            encoder_type=encoder_type,
            in_channels=in_channels,
            freeze_backbone=freeze_backbone,
        )

        # Get encoder channel dimensions
        encoder_channels = self.encoder.feature_channels

        # Create decoder
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=use_batchnorm,
            skip_connections=skip_connections,
            use_bilinear=True,  # Bilinear upsampling is memory efficient
        )

        # Final classification layer (always trained from scratch)
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Calculate receptive field
        encoder_name = (
            "resnet18" if "resnet18" in str(encoder_type).lower() else "resnet50"
        )
        self.receptive_field_info = calculate_receptive_field(encoder_name)

        # Print architecture summary
        print("U-Net initialized with:")
        print(f"- Encoder: {encoder_type} (pretrained on Sentinel-2)")
        print(
            f"- Transfer learning mode: {'Enabled' if freeze_backbone else 'Disabled'}"
        )
        print(f"- Skip connections: {min(skip_connections, len(encoder_channels)-1)}")
        print(f"- Output classes: {num_classes}")
        print(
            f"- Theoretical receptive field: {self.receptive_field_info['theoretical_rf']} pixels"
        )
        print(
            f"- Effective receptive field: {self.receptive_field_info['effective_rf']} pixels"
        )

    def forward(self, x):
        """
        Forward pass through the U-Net.
#
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        torch.Tensor
            Output segmentation logits
        """
        # Store input dimensions
        input_size = x.shape[2:]

        # Get encoder features using transfer learning
        encoder_features = self.encoder(x)

        # Apply decoder with skip connections
        x = self.decoder(encoder_features)

        # Final classification
        logits = self.final_conv(x)

        # Resize to original input dimensions if necessary
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=True
            )

        return logits

    def get_receptive_field_size(self):
        """
        Get the receptive field size of the model.

        Returns:
        --------
        dict
            Dictionary with receptive field information
        """
        return self.receptive_field_info

    def get_skip_connection_count(self):
        """
        Get the number of skip connections used in the model.

        Returns:
        --------
        int
            Number of skip connections between encoder and decoder
        """
        return self.decoder.skip_connections
    
    """
    Debug functions to identify NaN issues in UNet model.
    Add these functions to your models/unet.py file.
    """

    def check_tensor(self,tensor, name="tensor"):
        """Check for NaN or Inf values in a tensor."""
        if tensor is None:
            print(f"WARNING: {name} is None!")
            return False
            
        if torch.isnan(tensor).any():
            print(f"WARNING: NaN values detected in {name}")
            return False
            
        if torch.isinf(tensor).any():
            print(f"WARNING: Inf values detected in {name}")
            return False
            
        if tensor.min() == tensor.max():
            print(f"WARNING: {name} has constant values {tensor.min().item()}")
            return False
            
        # All checks passed
        return True


    # Add this debug forward method to your UNet class to identify where NaN values appear
    #def forward(self, x):
    #    """Forward pass with NaN checking at each stage."""
    #    # Check input
    #    print(f"Input shape: {x.shape}, range: [{x.min().item()}, {x.max().item()}]")
    #    if not self.check_tensor(x, "input"):
    #        return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    
    #    # Store original input size
    #    input_size = (x.shape[2], x.shape[3])
    #    
    #    # Encoder forward pass
    #    try:
    #        features = self.encoder.backbone.conv1(x)
    #        if not self.check_tensor(features, "encoder.conv1"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #            
    #        features = self.encoder.backbone.bn1(features)
    #        if not self.check_tensor(features, "encoder.bn1"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #            
    #        features = self.encoder.backbone.act1(features)
    #        if not self.check_tensor(features, "encoder.act1"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        
    #        # Get all encoder features
    #        encoder_features = {}
    #        encoder_features['stage0'] = features
    #        
    #        # Apply max pooling
    #        features = self.encoder.backbone.maxpool(features)
    #        if not self.check_tensor(features, "encoder.maxpool"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        
    #        # Process through encoder layers
    #        features = self.encoder.backbone.layer1(features)
    #        if not self.check_tensor(features, "encoder.layer1"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        encoder_features['stage1'] = features
    #        
    #        features = self.encoder.backbone.layer2(features)
    #        if not self.check_tensor(features, "encoder.layer2"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        encoder_features['stage2'] = features
    #        
    #        features = self.encoder.backbone.layer3(features)
    #        if not self.check_tensor(features, "encoder.layer3"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        encoder_features['stage3'] = features
    #        
    #        features = self.encoder.backbone.layer4(features)
    #        if not self.check_tensor(features, "encoder.layer4"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        encoder_features['stage4'] = features
    #    except Exception as e:
    #        print(f"Error in encoder: {e}")
    #        return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    
    #    # Convert encoder features to a list from deepest to shallowest
    #    try:
    #        features_list = [encoder_features[f'stage{i}'] for i in range(len(encoder_features))]
    #        features_list.reverse()  # Deepest features first
    #        
    #        # Start with bottleneck feature
    #        x = features_list[0]
    #        if not self.check_tensor(x, "bottleneck"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        
    #        # Apply decoder blocks with skip connections
    #        for i, up_block in enumerate(self.decoder.up_blocks):
    #            # Get appropriate skip connection if applicable
    #            skip = None
    #            if i < self.decoder.skip_connections and i+1 < len(features_list):
    #                skip = features_list[i+1]
    #                if skip is not None and not self.check_tensor(skip, f"skip_{i}"):
    #                    skip = None
    #            
    #            # Pass through decoder block
    #            try:
    #                x = up_block(x, skip)
    #                if not self.check_tensor(x, f"decoder_block_{i}"):
    #                    return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #            except Exception as e:
    #                print(f"Error in decoder block {i}: {e}")
    #                return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    except Exception as e:
    #        print(f"Error in decoder: {e}")
    #        return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    
    #    # Final classification
    #    try:
    #        logits = self.final_conv(x)
    #        if not self.check_tensor(logits, "logits"):
    #            return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #        
    #        # Resize to original input size if necessary
    #        if logits.shape[2:] != input_size:
    #            logits = F.interpolate(
    #                logits, size=input_size, mode='bilinear', align_corners=True
    #            )
    #            if not self.check_tensor(logits, "resized_logits"):
    #                return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    except Exception as e:
    #        print(f"Error in final convolution: {e}")
    #        return torch.zeros((x.size(0), self.num_classes, x.size(2), x.size(3)), device=x.device)
    #    
    #    return logits
#