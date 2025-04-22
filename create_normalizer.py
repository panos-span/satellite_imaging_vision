from dataset.normalizers import Sentinel2Normalizer
import pickle

# Create a normalizer with default settings for TorchGeo
normalizer = Sentinel2Normalizer(method="pretrained", rgb_indices=[0, 1, 2])
normalizer.torchgeo_specific = True
normalizer.is_fitted = True
normalizer.raw_scale_factor = 10000.0
normalizer.rgb_mean = [0.485, 0.456, 0.406]
normalizer.rgb_std = [0.229, 0.224, 0.225]
normalizer.other_mean = 0.5
normalizer.other_std = 0.5

# Save the normalizer
with open("F:/processed_data/training_dataset/normalizer.pkl", "wb") as f:
    pickle.dump(normalizer, f)
    
print("Created new normalizer file.")