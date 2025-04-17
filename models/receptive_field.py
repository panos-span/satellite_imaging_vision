"""
Receptive field calculation utilities for convolutional neural networks.

This module provides functions to calculate the theoretical receptive field size
of different CNN architectures, including ResNet backbones used in U-Net.
"""

def resnet_block_receptive_field(block_type, previous_rf, stride=1):
    """
    Calculate receptive field after a ResNet block.
    
    Parameters:
    -----------
    block_type : str
        Type of ResNet block ('basic' for ResNet18/34, 'bottleneck' for ResNet50+)
    previous_rf : int
        Receptive field size before this block
    stride : int
        Stride used in this block
        
    Returns:
    --------
    int
        Updated receptive field size
    """
    if block_type == 'basic':
        # Basic block has two 3x3 convolutions
        rf = previous_rf + 2 * (3 - 1) * stride
    elif block_type == 'bottleneck':
        # Bottleneck block has 1x1, 3x3, 1x1 convolutions
        # Only the 3x3 convolution increases the receptive field
        rf = previous_rf + (3 - 1) * stride
    else:
        raise ValueError(f"Unknown block type: {block_type}")
    
    return rf


def calculate_resnet_receptive_field(model_type):
    """
    Calculate the theoretical receptive field size of a ResNet model.
    
    Parameters:
    -----------
    model_type : str
        ResNet model type ('resnet18', 'resnet50')
        
    Returns:
    --------
    int
        Receptive field size in pixels
    """
    # Initial receptive field from first conv layer
    # ResNet starts with a 7x7 conv with stride 2
    rf = 1 + (7 - 1)  # Starting with 1 pixel, kernel size 7
    
    # Initial stride effect
    stride_product = 2  # First conv has stride 2
    
    # Effect of max pooling (3x3 with stride 2)
    rf += (3 - 1) * stride_product
    stride_product *= 2  # MaxPool stride
    
    # Different configurations for different ResNet variants
    if model_type == 'resnet18':
        block_type = 'basic'
        
        # ResNet18 configuration: [2, 2, 2, 2] blocks per layer
        blocks_per_layer = [2, 2, 2, 2]
        
        # Calculate RF through each layer
        for i, num_blocks in enumerate(blocks_per_layer):
            # First block in the layer might have stride 2 (except for first layer)
            layer_stride = 2 if i > 0 else 1
            
            # First block of the layer
            rf = resnet_block_receptive_field(block_type, rf, layer_stride)
            stride_product *= layer_stride
            
            # Remaining blocks in the layer
            for _ in range(1, num_blocks):
                rf = resnet_block_receptive_field(block_type, rf, 1)
    
    elif model_type == 'resnet50':
        block_type = 'bottleneck'
        
        # ResNet50 configuration: [3, 4, 6, 3] bottleneck blocks per layer
        blocks_per_layer = [3, 4, 6, 3]
        
        # Calculate RF through each layer
        for i, num_blocks in enumerate(blocks_per_layer):
            # First block in the layer might have stride 2 (except for first layer)
            layer_stride = 2 if i > 0 else 1
            
            # First block of the layer
            rf = resnet_block_receptive_field(block_type, rf, layer_stride)
            stride_product *= layer_stride
            
            # Remaining blocks in the layer
            for _ in range(1, num_blocks):
                rf = resnet_block_receptive_field(block_type, rf, 1)
    
    else:
        raise ValueError(f"Unsupported ResNet model: {model_type}")
    
    return rf


def calculate_unet_receptive_field(encoder_type, num_decoder_blocks=5):
    """
    Calculate the receptive field size of a U-Net model with a ResNet encoder.
    
    Parameters:
    -----------
    encoder_type : str
        Type of encoder ('resnet18', 'resnet50', etc.)
    num_decoder_blocks : int
        Number of decoder blocks in the U-Net
        
    Returns:
    --------
    int
        Receptive field size in pixels
    """
    # Calculate encoder receptive field
    encoder_rf = calculate_resnet_receptive_field(encoder_type)
    
    # Decoder blocks typically don't increase the receptive field
    # in the same way since they're upsampling, but the convolutions
    # after concatenation do have a small impact
    decoder_impact = num_decoder_blocks * 2  # Each decoder has typically 2 3x3 convs
    
    # Total receptive field
    total_rf = encoder_rf + decoder_impact
    
    return total_rf