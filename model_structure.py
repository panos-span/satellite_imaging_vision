"""
Script to inspect the structure of TorchGeo's ResNet models.
Run this to understand how to access the model components.
"""
import torch
from torchgeo.models import resnet18, resnet50
from torchgeo.models.resnet import ResNet18_Weights, ResNet50_Weights
import pprint

# Function to print model structure
def print_model_structure(model, max_depth=3, prefix=''):
    """Print the structure of a PyTorch model up to a specified depth."""
    if max_depth <= 0:
        return
    
    # Print modules in the current level
    for name, module in model.named_children():
        print(f"{prefix}{name}: {module.__class__.__name__}")
        # Recursively print sub-modules
        print_model_structure(module, max_depth - 1, prefix + '  ')

# Function to examine features output
def examine_features(model, input_shape=(1, 13, 224, 224)):
    """Examine the feature maps returned by the model in features_only mode."""
    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)
    
    # Get features
    features = model(dummy_input)
    
    # Print information about the features
    print("Features information:")
    if isinstance(features, list):
        for i, feature in enumerate(features):
            print(f"  Feature {i}: shape={feature.shape}")
    else:
        print(f"  Output shape: {features.shape}")
    
    return features

# Main inspection code
if __name__ == "__main__":
    print("Examining ResNet50 from TorchGeo")
    
    # Create model with features_only=True to get intermediate outputs
    model_features = resnet50(
        weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, 
        features_only=True
    )
    
    # Create regular model
    model_regular = resnet50(
        weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, 
        features_only=False
    )
    
    # Print high-level structure
    print("\nModel structure (features_only=True):")
    print_model_structure(model_features, max_depth=2)
    
    print("\nModel structure (features_only=False):")
    print_model_structure(model_regular, max_depth=2)
    
    # Examine output features
    print("\nExamining output features:")
    features = examine_features(model_features)
    
    # Print model's feature info if available
    if hasattr(model_features, 'feature_info'):
        print("\nModel feature_info:")
        pprint.pprint(model_features.feature_info)
    
    # Inspect parameters
    print("\nParameter groups:")
    param_groups = []
    for name, _ in model_features.named_parameters():
        # Get the top-level parameter group
        group = name.split('.')[0]
        if group not in param_groups:
            param_groups.append(group)
    
    print(f"Top-level parameter groups: {param_groups}")
    
    # Count parameters in each group
    for group in param_groups:
        params = sum(p.numel() for name, p in model_features.named_parameters() if name.startswith(f"{group}."))
        print(f"  {group}: {params:,} parameters")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model_features.parameters()):,}")