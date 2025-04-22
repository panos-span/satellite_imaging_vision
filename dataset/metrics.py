"""
Revised evaluation metrics for semantic segmentation using torchmetrics.

This module provides metrics for semantic segmentation using the torchmetrics library,
with improved handling for multiclass segmentation and edge cases.
"""

import torch
import torchmetrics
from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import (
    BinaryF1Score, 
    MulticlassF1Score,
    BinaryAccuracy, 
    MulticlassAccuracy
)

class SegmentationMetrics:
    """Class to handle both binary and multiclass segmentation metrics with improved stability."""
    
    def __init__(self, num_classes=1, threshold=0.5, device=None):
        """
        Initialize segmentation metrics.
        
        Parameters:
        -----------
        num_classes : int
            Number of classes (1 for binary segmentation, >1 for multiclass)
        threshold : float
            Threshold for binary segmentation
        device : torch.device
            Device to use for metric computation (CPU or CUDA)
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.device = device if device is not None else torch.device("cpu")
        
        # Initialize metrics based on number of classes
        if num_classes == 1:
            # Binary segmentation metrics
            self.iou_score = MeanIoU(num_classes=2).to(device)
            self.dice_score = DiceScore(num_classes=2).to(device)
            self.f1_score = BinaryF1Score(threshold=threshold).to(device)
            self.accuracy = BinaryAccuracy(threshold=threshold).to(device)
        else:
            # Multiclass segmentation metrics
            self.iou_score = MeanIoU(num_classes=num_classes).to(device)
            self.dice_score = DiceScore(num_classes=num_classes).to(device)
            self.f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
            self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    
    def update(self, outputs, targets):
        """
        Update metrics with batch of predictions and targets.
        
        Parameters:
        -----------
        outputs : torch.Tensor
            Predicted segmentation maps (B, C, H, W) for logits or (B, H, W) for indices
        targets : torch.Tensor
            Ground truth segmentation maps (B, H, W) with class indices
        """
        with torch.no_grad():  # Ensure no gradients are tracked for metrics
            # Handle binary segmentation
            if self.num_classes == 1:
                # Convert from (B, 1, H, W) logits to (B, H, W) binary predictions
                if outputs.dim() == 4 and outputs.shape[1] == 1:
                    preds = (torch.sigmoid(outputs) > self.threshold).squeeze(1).long()
                else:
                    preds = outputs  # Assume already processed
                
                # Ensure targets are in correct format (B, H, W) with class indices (0 or 1)
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1).long()
            
            # Handle multiclass segmentation
            else:
                # Handle logits (B, C, H, W)
                if outputs.dim() == 4 and outputs.shape[1] == self.num_classes:
                    # For IoU and Dice - need class indices
                    preds_indices = torch.argmax(outputs, dim=1)
                    
                    # For F1 and Accuracy - need softmax probabilities
                    preds_probs = torch.softmax(outputs, dim=1)
                    
                    # Use appropriate prediction format for each metric
                    self.iou_score.update(preds_indices, targets)
                    self.dice_score.update(preds_indices, targets)
                    self.f1_score.update(preds_probs, targets)
                    self.accuracy.update(preds_probs, targets)
                    
                    # Exit early since we've already updated all metrics
                    return
                else:
                    # If outputs are already class indices (B, H, W)
                    preds = outputs
                
                # Ensure targets are in correct format (B, H, W) with class indices
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1).long()
                elif targets.dim() == 4 and targets.shape[1] > 1:
                    # One-hot encoded targets, convert to indices
                    targets = torch.argmax(targets, dim=1)
            
            # Update metrics (only for binary or when outputs are already indices)
            self.iou_score.update(preds, targets)
            self.dice_score.update(preds, targets)
            
            # For binary or when already using indices, convert to appropriate format for F1 and Accuracy
            if self.num_classes == 1:
                # Binary metrics need different input formats
                self.f1_score.update(preds, targets)
                self.accuracy.update(preds, targets)
            else:
                # For multiclass with indices, we need to one-hot encode for F1 and Accuracy
                preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
                self.f1_score.update(preds_one_hot, targets)
                self.accuracy.update(preds_one_hot, targets)
    
    def compute(self):
        """
        Compute and return metrics with error handling.
        
        Returns:
        --------
        dict
            Dictionary with computed metric values
        """
        try:
            iou = self.iou_score.compute().item()
        except Exception as e:
            print(f"Error computing IoU: {e}")
            iou = 0.0
            
        try:
            dice = self.dice_score.compute().item()
        except Exception as e:
            print(f"Error computing Dice: {e}")
            dice = 0.0
            
        try:
            f1 = self.f1_score.compute().item()
        except Exception as e:
            print(f"Error computing F1: {e}")
            f1 = 0.0
            
        try:
            accuracy = self.accuracy.compute().item()
        except Exception as e:
            print(f"Error computing Accuracy: {e}")
            accuracy = 0.0
            
        return {
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "accuracy": accuracy
        }
    
    def reset(self):
        """Reset all metrics."""
        self.iou_score.reset()
        self.dice_score.reset()
        self.f1_score.reset()
        self.accuracy.reset()


def create_metrics(num_classes=1, threshold=0.5, device=None):
    """
    Factory function to create segmentation metrics.
    
    Parameters:
    -----------
    num_classes : int
        Number of classes (1 for binary segmentation, >1 for multiclass)
    threshold : float
        Threshold for binary segmentation
    device : torch.device
        Device to use for metric computation
        
    Returns:
    --------
    SegmentationMetrics
        Metrics object for segmentation
    """
    return SegmentationMetrics(num_classes, threshold, device)