"""
Implementation of CopyPaste augmentation for semantic segmentation.

CopyPaste is particularly useful for segmentation tasks as it allows the model
to see objects in different contexts, which can improve generalization.
"""
import random
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import skimage.measure
from albumentations.core.transforms_interface import DualTransform


class CopyPaste(DualTransform):
    """
    CopyPaste augmentation for segmentation tasks.
    
    This transform copies objects from one image to another based on segmentation masks.
    It's particularly useful for satellite imagery where different land cover types 
    can appear in various contexts.
    
    Parameters:
    -----------
    object_classes : List[int]
        List of class IDs that should be considered as objects for copying
    p : float
        Probability of applying the transform
    max_objects_per_image : int
        Maximum number of objects to paste in a single image
    min_object_area : int
        Minimum area (in pixels) required for an object to be considered for copying
    max_object_area : int
        Maximum area (in pixels) for an object to be considered for copying
    """
    def __init__(
        self,
        object_classes: List[int],
        p: float = 0.5,
        max_objects_per_image: int = 3,
        min_object_area: int = 100,
        max_object_area: int = 10000,
    ):
        super().__init__(p=p)
        self.object_classes = object_classes
        self.max_objects_per_image = max_objects_per_image
        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
    
    def apply(self, img, paste_img=None, paste_mask=None, **params):
        """Apply the transform to the image."""
        if paste_img is None or paste_mask is None:
            return img
        
        # Create a copy of the image to avoid modifying the original
        result_img = img.copy()
        
        # Get the coordinates where to paste
        coords = params.get('coords', [])
        
        for i, (x, y, obj_img, obj_mask) in enumerate(coords):
            h, w = obj_mask.shape[:2]
            
            # Make sure we don't go out of bounds
            if y + h > result_img.shape[0] or x + w > result_img.shape[1]:
                continue
            
            # Create masks for blending
            mask_3d = np.stack([obj_mask] * 3, axis=2) if len(result_img.shape) == 3 else obj_mask
            
            # Paste the object onto the image
            result_img[y:y+h, x:x+w] = obj_img * mask_3d + result_img[y:y+h, x:x+w] * (1 - mask_3d)
            
        return result_img
    
    def apply_to_mask(self, mask, paste_mask=None, **params):
        """Apply the transform to the mask."""
        if paste_mask is None:
            return mask
        
        # Create a copy of the mask to avoid modifying the original
        result_mask = mask.copy()
        
        # Get the coordinates where to paste
        coords = params.get('coords', [])
        
        for i, (x, y, _, obj_mask) in enumerate(coords):
            h, w = obj_mask.shape[:2]
            
            # Make sure we don't go out of bounds
            if y + h > result_mask.shape[0] or x + w > result_mask.shape[1]:
                continue
            
            # Paste the object onto the mask
            result_mask[y:y+h, x:x+w] = obj_mask * 1 + result_mask[y:y+h, x:x+w] * (1 - obj_mask)
            
        return result_mask
    
    def _extract_objects(self, image, mask):
        """Extract objects from the image based on the mask."""
        objects = []
        
        for class_id in self.object_classes:
            # Create a binary mask for this class
            class_mask = (mask == class_id).astype(np.uint8)
            
            if np.sum(class_mask) == 0:
                continue
            
            # Label connected components in the mask
            labeled_mask, num_labels = skimage.measure.label(class_mask, return_num=True, connectivity=2)
            
            for label_id in range(1, num_labels + 1):
                # Get the object mask
                obj_mask = (labeled_mask == label_id).astype(np.uint8)
                
                # Calculate object area
                obj_area = np.sum(obj_mask)
                
                # Check if the object size is within the desired range
                if self.min_object_area <= obj_area <= self.max_object_area:
                    # Find the bounding box of the object
                    y_indices, x_indices = np.where(obj_mask > 0)
                    
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue
                    
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    # Extract the object and its mask
                    h, w = y_max - y_min + 1, x_max - x_min + 1
                    obj_img = image[y_min:y_max+1, x_min:x_max+1].copy()
                    obj_mask = obj_mask[y_min:y_max+1, x_min:x_max+1].copy()
                    
                    objects.append((obj_img, obj_mask, class_id, h, w))
        
        return objects
    
    def _get_random_paste_coordinates(self, canvas_height, canvas_width, obj_height, obj_width):
        """Get random coordinates where to paste an object."""
        y = random.randint(0, max(0, canvas_height - obj_height))
        x = random.randint(0, max(0, canvas_width - obj_width))
        return x, y
    
    def get_params_dependent_on_targets(self, params):
        """Get parameters for the transform based on the targets."""
        img = params["image"]
        mask = params["mask"]
        
        height, width = img.shape[:2]
        
        # Extract objects from the current image
        objects = self._extract_objects(img, mask)
        
        if not objects:
            return {"paste_img": None, "paste_mask": None, "coords": []}
        
        # Determine how many objects to paste
        num_objects = min(len(objects), random.randint(1, self.max_objects_per_image))
        
        # Randomly select objects to paste
        selected_objects = random.sample(objects, num_objects)
        
        # Generate coordinates for each object
        coords = []
        for obj_img, obj_mask, class_id, h, w in selected_objects:
            x, y = self._get_random_paste_coordinates(height, width, h, w)
            coords.append((x, y, obj_img, obj_mask))
        
        return {
            "paste_img": img,
            "paste_mask": mask,
            "coords": coords
        }


class LandCoverCopyPaste(CopyPaste):
    """
    Specialized CopyPaste augmentation for land cover classification.
    
    This version includes additional logic specifically for handling land cover
    classes in satellite imagery.
    
    Parameters:
    -----------
    object_classes : List[int]
        List of class IDs that should be considered as objects for copying
    background_classes : List[int], optional
        List of class IDs that are considered as background and can be replaced by objects
    p : float
        Probability of applying the transform
    max_objects_per_image : int
        Maximum number of objects to paste in a single image
    min_object_area : int
        Minimum area (in pixels) required for an object to be considered for copying
    max_object_area : int
        Maximum area (in pixels) for an object to be considered for copying
    blend_mode : str
        Blending mode for pasting objects ('normal', 'gaussian', or 'poisson')
    """
    def __init__(
        self,
        object_classes: List[int],
        background_classes: Optional[List[int]] = None,
        p: float = 0.5,
        max_objects_per_image: int = 3,
        min_object_area: int = 100,
        max_object_area: int = 10000,
        blend_mode: str = 'normal'
    ):
        super().__init__(
            object_classes=object_classes,
            p=p,
            max_objects_per_image=max_objects_per_image,
            min_object_area=min_object_area,
            max_object_area=max_object_area
        )
        self.background_classes = background_classes or []
        self.blend_mode = blend_mode
    
    def apply(self, img, paste_img=None, paste_mask=None, **params):
        """Apply the transform to the image with the specified blend mode."""
        if paste_img is None or paste_mask is None:
            return img
        
        # Create a copy of the image to avoid modifying the original
        result_img = img.copy()
        
        # Get the coordinates where to paste
        coords = params.get('coords', [])
        is_multispectral = len(img.shape) == 3 and img.shape[2] > 3
        
        for i, (x, y, obj_img, obj_mask) in enumerate(coords):
            h, w = obj_mask.shape[:2]
            
            # Make sure we don't go out of bounds
            if y + h > result_img.shape[0] or x + w > result_img.shape[1]:
                continue
            
            # For multispectral images, use normal blending for all bands
            if is_multispectral:
                # Create a 3D mask with the correct number of channels
                mask_nd = np.expand_dims(obj_mask, axis=-1) if len(obj_mask.shape) == 2 else obj_mask
                mask_nd = np.repeat(mask_nd, img.shape[2], axis=-1)
                
                # Paste the object onto the image
                result_img[y:y+h, x:x+w] = obj_img * mask_nd + result_img[y:y+h, x:x+w] * (1 - mask_nd)
                continue
            
            # For RGB images, we can use more advanced blending techniques
            if self.blend_mode == 'gaussian':
                # Apply Gaussian blending
                mask_3d = np.stack([obj_mask] * 3, axis=2) if len(obj_mask.shape) == 2 else obj_mask
                # Blur the mask for softer edges
                blurred_mask = cv2.GaussianBlur(mask_3d, (5, 5), 2)
                result_img[y:y+h, x:x+w] = obj_img * blurred_mask + result_img[y:y+h, x:x+w] * (1 - blurred_mask)
            
            elif self.blend_mode == 'poisson':
                # Attempt to use Poisson blending if available
                try:
                    # Create a mask for blending (binary)
                    blend_mask = (obj_mask * 255).astype(np.uint8)
                    
                    # Define the center point for seamless cloning
                    center = (x + w // 2, y + h // 2)
                    
                    # Apply seamless cloning
                    # Note: this modifies result_img in-place
                    cv2.seamlessClone(
                        obj_img, 
                        result_img, 
                        blend_mask, 
                        center, 
                        cv2.NORMAL_CLONE
                    )
                except Exception:
                    # Fallback to normal blending if Poisson fails
                    mask_3d = np.stack([obj_mask] * 3, axis=2) if len(obj_mask.shape) == 2 else obj_mask
                    result_img[y:y+h, x:x+w] = obj_img * mask_3d + result_img[y:y+h, x:x+w] * (1 - mask_3d)
            
            else:  # 'normal' blending
                # Standard alpha blending
                mask_3d = np.stack([obj_mask] * 3, axis=2) if len(obj_mask.shape) == 2 else obj_mask
                result_img[y:y+h, x:x+w] = obj_img * mask_3d + result_img[y:y+h, x:x+w] * (1 - mask_3d)
            
        return result_img
    
    def apply_to_mask(self, mask, paste_mask=None, **params):
        """
        Apply the transform to the mask, considering background classes.
        """
        if paste_mask is None:
            return mask
        
        # Create a copy of the mask to avoid modifying the original
        result_mask = mask.copy()
        
        # Get the coordinates where to paste
        coords = params.get('coords', [])
        
        for i, (x, y, _, obj_mask) in enumerate(coords):
            h, w = obj_mask.shape[:2]
            
            # Make sure we don't go out of bounds
            if y + h > result_mask.shape[0] or x + w > result_mask.shape[1]:
                continue
            
            # Get the region where we'll paste
            target_region = result_mask[y:y+h, x:x+w].copy()
            
            # Only paste where the target region is a background class
            if self.background_classes:
                background_mask = np.zeros_like(target_region, dtype=bool)
                for bg_class in self.background_classes:
                    background_mask = np.logical_or(background_mask, target_region == bg_class)
                
                # Combine masks: paste only where obj_mask is 1 AND target is background
                combined_mask = obj_mask.astype(bool) & background_mask
                
                # Apply to result mask
                temp_mask = result_mask[y:y+h, x:x+w]
                class_id = params.get(f'class_id_{i}', 1)  # Default to class 1 if not specified
                temp_mask[combined_mask] = class_id
                result_mask[y:y+h, x:x+w] = temp_mask
            else:
                # Paste the object onto the mask
                result_mask[y:y+h, x:x+w] = obj_mask * params.get(f'class_id_{i}', 1) + \
                                           result_mask[y:y+h, x:x+w] * (1 - obj_mask)
            
        return result_mask