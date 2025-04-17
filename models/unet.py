"""
U-Net implementation for Sentinel-2 imagery segmentation with TorchGeo pretrained weights.

This module provides a U-Net architecture specifically designed for Sentinel-2 imagery
with 13 input channels at 10m spatial resolution. Key features:

1. Transfer Learning: The encoder part consists exclusively of layers from a pre-trained
   ResNet model (ResNet50 or ResNet18) with Sentinel-2 specific weights.

2. Skip Connections: The architecture includes at least 2 skip connections between the
   encoder and decoder for transferring features of specific scales. These skip connections
   are separate from and in addition to any skip connections within the ResNet itself.
"""

import torch.nn.functional as F

from .blocks import UpBlock
from .encoder import create_encoder
import torch.nn as nn


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation of Sentinel-2 imagery.

    Key features:
    1. Transfer Learning: The encoder part consists exclusively of layers from a
       pre-trained ResNet model (ResNet50 or ResNet18) with Sentinel-2 specific weights.

    2. Skip Connections: The architecture includes at least 2 explicit skip connections
       between the encoder and decoder for transferring features of specific scales.

    Parameters:
    -----------
    in_channels : int
        Number of input channels (default: 13 for Sentinel-2)
    num_classes : int
        Number of output classes
    encoder_type : str
        Type of encoder to use ('best_performance', 'all_bands', 'lightweight', 'balanced')
    decoder_channels : list
        Number of channels in each decoder block
    use_batchnorm : bool
        Whether to use batch normalization
    skip_connections : int
        Minimum number of skip connections to use (default: 2, must be at least 2)
    """

    def __init__(
        self,
        in_channels=13,
        num_classes=1,
        encoder_type="best_performance",
        decoder_channels=[256, 128, 64, 32],
        use_batchnorm=True,
        skip_connections=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_type = encoder_type

        # Ensure we have at least 2 skip connections
        if skip_connections < 2:
            print("Minimum number of skip connections is 2.")
            skip_connections = 2
        self.skip_connections = skip_connections

        # Create ResNet encoder with Sentinel-2 specific pretrained weights
        self.encoder = create_encoder(encoder_type, in_channels)

        # Get encoder channels for skip connections
        encoder_channels = self.encoder.feature_channels

        # Create encoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList()

        # Calculate how many decoder blocks we need
        num_decoder_blocks = len(decoder_channels)

        # Create decoder blocks with skip connections
        for i in range(num_decoder_blocks):
            # Input channels come from either the encoder bottleneck or the previous decoder block
            in_ch = encoder_channels[-1] if i == 0 else decoder_channels[i - 1]

            # Skip connection channels - use 0 if this decoder block doesn't have a skip connection
            use_skip = i < self.skip_connections
            skip_idx = -i - 2 if i < len(encoder_channels) - 1 else 0
            skip_ch = encoder_channels[skip_idx] if use_skip else 0

            # Output channels for this decoder block
            out_ch = decoder_channels[i]

            # Create the decoder block and append it to the list
            self.decoder_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm)
            )

        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Print information about skip connections during initialization
        skip_indices = list(
            range(min(self.skip_connections, len(encoder_channels) - 1))
        )
        print(
            f"U-Net initialized with {self.skip_connections} skip connections from encoder stages: {skip_indices}"
        )

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes, height, width)
        """
        # Store original input size for later use
        input_size = (x.shape[2], x.shape[3])

        # Encoder forward pass
        features = self.encoder(x)

        # Convert features dict to a list (from deepest to shallowest)
        encoder_features = [features[f"stage{i}"] for i in range(len(features))]
        encoder_features.reverse()  # from deepest to shallowest

        # Start decoder forward pass
        x = encoder_features[0]

        # Apply decoder blocks with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Determine if this block should use a skip connection
            use_skip = i < self.skip_connections
            skip = (
                encoder_features[i + 1]
                if use_skip and i + 1 < len(encoder_features)
                else None
            )

            # Apply the decoder block
            x = decoder_block(x, skip)
    
        # Final convolution to get logits
        logits = self.final_conv(x)
        
        # Resize to original input size if needed
        if logits.shape[2] != input_size[0] or logits.shape[3] != input_size[1]:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        
        return logits
    
    def get_receptive_field_size(self):
        """
        Estimate the receptive field size of the model.
        
        Returns:
        --------
        int
            Approximate receptive field size in pixels
        """
        if 'resnet18' in str(self.encoder_type).lower():
            return 212  # Approximate receptive field for ResNet-18
        else:  # resnet50
            return 256  # Approximate receptive field for ResNet-50
            
    def get_skip_connection_count(self):
        """
        Get the number of skip connections used in the model.
        
        Returns:
        --------
        int
            Number of skip connections between encoder and decoder
        """
        return self.skip_connections