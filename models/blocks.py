"""
Building blocks for constructing U-Net segmentation networks.

This module provides optimized building blocks for the U-Net architecture,
including convolutional blocks and upsampling blocks with efficient skip connection handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolution kernel
    padding : int
        Padding for the convolution
    use_batchnorm : bool
        Whether to use batch normalization (True) or group normalization (False)
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=True
    ):
        super().__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batchnorm,  # No bias when using normalization
        )

        # Normalization layer
        if use_batchnorm:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            # Group normalization with optimal groups
            num_groups = min(8, out_channels) if out_channels % 8 == 0 else 1
            self.norm = nn.GroupNorm(num_groups, out_channels)

        # Activation function (using inplace for memory efficiency)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass with efficient operations"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DoubleConvBlock(nn.Module):
    """
    Double convolutional block with shared activation.

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    use_batchnorm : bool
        Whether to use batch normalization
    """

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        # Define the double convolutional block
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm),
            ConvBlock(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        """Forward pass through the block."""
        return self.double_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder with efficient skip connection handling.

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    skip_channels : int
        Number of channels from the skip connection
    out_channels : int
        Number of output channels
    use_batchnorm : bool
        Whether to use batch normalization
    bilinear : bool
        Whether to use bilinear upsampling or transposed convolution
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        bilinear=True,
    ):
        super().__init__()

        self.has_skip = skip_channels > 0

        # Upsampling operation
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_channels, in_channels // 2, kernel_size=1
                ),  # Channel reduction
            )
            up_out_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            up_out_channels = in_channels // 2

        # Calculate combined channels after skip connection
        combined_channels = (
            up_out_channels + skip_channels if self.has_skip else up_out_channels
        )

        # Double convolution after upsampling and potential concatenation
        self.conv_block = DoubleConvBlock(
            combined_channels, out_channels, use_batchnorm=use_batchnorm
        )

    def forward(self, x, skip=None):
        """
        Forward pass with efficient skip connection handling.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor from the previous decoder stage or encoder bottleneck
        skip : torch.Tensor or None
            Skip connection feature from the encoder

        Returns:
        --------
        torch.Tensor
            Output tensor after upsampling and processing
        """
        # Upsampling
        x = self.up(x)

        # Skip connection handling
        if self.has_skip and skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            # Concatenate along channel dimension
            x = torch.cat([skip, x], dim=1)

        # Apply convolution block
        x = self.conv_block(x)

        return x
