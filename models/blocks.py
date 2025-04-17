"""
Building blocks for constructing U-Net segmentation networks.

This module provides common building blocks like convolutional blocks
and upsampling blocks used in U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard convolutional block with batch normalization and activation.

    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolutional kernel
    padding : int
        Padding size
    use_batchnorm : bool
        Whether to use batch normalization
    activation : torch.nn.Module
        Activation function to use
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        use_batchnorm=True,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not use_batchnorm,  # No bias when using batch norm
        )

        # Define batch normalization layer if requested
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        # Store activation function
        self.activation = activation

    def forward(self, x):
        """Forward pass through the block."""
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class DoubleConvBlock(nn.Module):
    """
    Double convolutional block used in U-Net.

    This block consists of two consecutive convolutional blocks with
    batch normalization and activation.

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
    Upsampling block used in U-Net decoder.

    This block consists of an upsampling operation followed by a double
    convolutional block. It also handles the skip connection concatenation.

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
        Whether to use bilinear interpolation for upsampling
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

        # Define the upsampling operation
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )

        # Define the convolutional blocks
        self.conv1 = ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm)

        # Combined channels accounting for skip connection
        combined_channels = (
            out_channels + skip_channels if skip_channels > 0 else out_channels
        )
        self.conv2 = ConvBlock(
            combined_channels, out_channels, use_batchnorm=use_batchnorm
        )

    def forward(self, x, skip=None):
        """
        Forward pass through the block.

        Parameters:
        -----------
        x : torch.Tensor
            Tensor from the encoder path
        skip : torch.Tensor
            Tensor from the skip connection

        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        # Upsample x
        x = self.up(x)
        x = self.conv1(x)

        # Skip connection
        if skip is not None:
            # Check if dimensions match
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]

            # Resize skip connection if dimensions don't match
            if diffY != 0 or diffX != 0:
                x = F.pad(
                    x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )

            # Concatenate along the channel dimension
            x = torch.cat([skip, x], dim=1)

        # Apply second convolution
        x = self.conv2(x)

        return x
