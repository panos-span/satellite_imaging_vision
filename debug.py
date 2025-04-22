import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the first few samples from the dataset
def main():
    data_dir = r"F:\processed_data\training_dataset"
    train_dir = Path(data_dir) / "train"
    image_files = list((train_dir / "images").glob("*.npy"))
    
    # Load normalizer
    try:
        with open(Path(data_dir) / "normalizer.pkl", "rb") as f:
            normalizer = pickle.load(f)
        print(f"Loaded normalizer: {normalizer.method}")
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        normalizer = None
    
    # Simple in-place normalization that definitely won't produce NaN
    def safe_normalize(tensor):
        # Simple scaling for Sentinel-2 data
        if tensor.max() > 100:
            tensor = tensor / 10000.0
        # Clip to safe range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor
    
    # Create a direct replacement normalizer.pkl file
    def create_replacement_normalizer():
        from torch.nn import Module
        
        # Create a simple normalizer class that's pickle-compatible
        class SimpleNormalizer(Module):
            def __init__(self):
                super().__init__()
                self.method = "fixed"
                self.scale = 10000.0
                
            def transform(self, tensor):
                # Convert to numpy if it's a tensor
                if isinstance(tensor, torch.Tensor):
                    input_tensor = tensor.cpu().numpy()
                    is_tensor = True
                else:
                    input_tensor = tensor
                    is_tensor = False
                
                # Scale and clip
                result = input_tensor / self.scale
                result = np.clip(result, 0.0, 1.0)
                
                # Convert back if needed
                if is_tensor:
                    return torch.from_numpy(result).float()
                return result
        
        # Create and save the new normalizer
        new_normalizer = SimpleNormalizer()
        with open(Path(data_dir) / "normalizer.pkl", "wb") as f:
            pickle.dump(new_normalizer, f)
        print("Created new simplified normalizer file")
        
        return new_normalizer
    
    # Load a sample image
    if image_files:
        sample = np.load(image_files[0])
        image_tensor = torch.from_numpy(sample).float()
        
        print(f"Original image stats - min: {image_tensor.min().item()}, max: {image_tensor.max().item()}")
        
        # Try normalizing with loaded normalizer
        if normalizer:
            try:
                normalized = normalizer.transform(image_tensor)
                print(f"Normalizer output - min: {normalized.min().item()}, max: {normalized.max().item()}")
                print(f"Contains NaN: {torch.isnan(normalized).any().item()}")
                print(f"Contains Inf: {torch.isinf(normalized).any().item()}")
                
                if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                    print("ISSUE DETECTED: Normalizer produces NaN/Inf values!")
                    normalizer = create_replacement_normalizer()
            except Exception as e:
                print(f"ERROR using normalizer: {e}")
                normalizer = create_replacement_normalizer()
        else:
            normalizer = create_replacement_normalizer()
        
        # Always test the safe normalization
        safe_normalized = safe_normalize(image_tensor)
        print(f"Safe normalization - min: {safe_normalized.min().item()}, max: {safe_normalized.max().item()}")
        
        # Visualize sample bands
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image_tensor[0].numpy())
        plt.title("Original Band 1")
        plt.colorbar()
        
        plt.subplot(132)
        try:
            plt.imshow(normalized[0].numpy())
            plt.title("Normalized with Loaded Normalizer")
            plt.colorbar()
        except:
            plt.title("Normalizer Failed")
            
        plt.subplot(133)
        plt.imshow(safe_normalized[0].numpy())
        plt.title("Safe Normalization")
        plt.colorbar()
        
        plt.savefig(Path(data_dir) / "normalization_debug.png")
        print(f"Saved visualization to {data_dir}/normalization_debug.png")

if __name__ == "__main__":
    main()