"""
Revised evaluation metrics for semantic segmentation using torchmetrics.

This module provides metrics for semantic segmentation using the torchmetrics library,
with improved handling for multiclass segmentation and edge cases.
"""

import torch
import numpy as np
from torchmetrics.segmentation import MeanIoU, DiceScore
import torchmetrics
from torchmetrics.classification import (
    BinaryF1Score,
    MulticlassF1Score,
    BinaryAccuracy,
    MulticlassAccuracy,
)

import torch.nn as nn
import torch.nn.functional as F

class CorrectDiceMetric(torchmetrics.Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Initialize accumulators for TP, FP, FN per class
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
    
    def update(self, preds, targets):
        """Update metric state with predictions and targets"""
        # Handle logits vs predictions
        if preds.dim() == 4 and preds.size(1) > 1:  # [B, C, H, W] format with logits
            preds = torch.argmax(preds, dim=1)  # Convert to class indices
        
        # Process each class
        for c in range(self.num_classes):
            # Create binary masks
            pred_mask = (preds == c)
            target_mask = (targets == c)
            
            # Skip ignored indices
            if self.ignore_index is not None:
                valid_mask = (targets != self.ignore_index)
                pred_mask = pred_mask & valid_mask
                target_mask = target_mask & valid_mask
            
            # Calculate metrics components
            self.true_positives[c] += (pred_mask & target_mask).sum().float()
            self.false_positives[c] += (pred_mask & ~target_mask).sum().float()
            self.false_negatives[c] += (~pred_mask & target_mask).sum().float()
    
    def compute(self):
        """Compute Dice scores for all classes"""
        # Formula: 2*TP / (2*TP + FP + FN)
        numerator = 2 * self.true_positives
        denominator = 2 * self.true_positives + self.false_positives + self.false_negatives
        
        # Handle division by zero (classes not present in batch)
        dice_per_class = torch.zeros_like(numerator)
        non_zero_indices = denominator > 0
        dice_per_class[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
        
        # For overall dice, only average classes that actually appear in the data
        class_has_data = (self.true_positives + self.false_negatives) > 0
        
        if class_has_data.sum() > 0:
            # Average only over classes that appear in the data
            mean_dice = dice_per_class[class_has_data].mean()
        else:
            # Edge case - no class data
            mean_dice = torch.tensor(0.0, device=dice_per_class.device)
        
        # Print diagnostics to help debug
        print(f"Class dice values: {dice_per_class}")
        print(f"Mean dice: {mean_dice}")
        
        return mean_dice


class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            # Ensure alpha is balanced - limit the maximum weight
            if isinstance(alpha, torch.Tensor):
                # Cap the maximum weight to 10x the minimum
                min_alpha = alpha.min()
                alpha = torch.clamp(alpha, min=min_alpha, max=min_alpha * 10)
                # Renormalize
                alpha = alpha / alpha.sum()
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # Standard focal loss implementation with balanced alpha
        if inputs.dim() > 2:
            # N,C,H,W => N,C,H*W
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            # N,C,H*W => N,H*W,C
            inputs = inputs.transpose(1, 2)
            # N,H*W,C => N*H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))

        targets = targets.view(-1, 1)

        # Apply softmax to get probabilities
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets.view(-1))
            logpt = logpt * at

        # Compute focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def calculate_balanced_alpha(class_counts):
    """Calculate balanced alpha weights for focal loss."""
    class_counts = np.array(class_counts)

    # Method 1: Square root scaling to reduce extreme values
    # This moderates the impact of class imbalance
    scaled_counts = np.sqrt(class_counts)
    inv_freq = 1.0 / scaled_counts

    # Method 2: Logarithmic scaling - even more moderate
    # scaled_counts = np.log(class_counts + 1)
    # inv_freq = 1.0 / scaled_counts

    # Normalize to sum to 1
    alpha = inv_freq / inv_freq.sum()

    # Ensure no class gets more than 30% of total weight
    alpha = np.clip(alpha, 0, 0.3)
    alpha = alpha / alpha.sum()  # Renormalize after clipping

    return alpha


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
            self.dice_score = CorrectDiceMetric(num_classes=num_classes),
            self.f1_score = BinaryF1Score(threshold=threshold).to(device)
            self.accuracy = BinaryAccuracy(threshold=threshold).to(device)
        else:
            # Multiclass segmentation metrics
            self.iou_score = MeanIoU(num_classes=num_classes).to(device)
            self.dice_score = DiceScore(num_classes=num_classes).to(device)
            self.f1_score = MulticlassF1Score(
                num_classes=num_classes, average="macro"
            ).to(device)
            self.accuracy = MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            ).to(device)

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
                preds_one_hot = (
                    torch.nn.functional.one_hot(preds, num_classes=self.num_classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
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
            # For multiclass, compute() returns a tensor with dice for each class
            # We should average them, not sum them
            if self.num_classes > 1:
                dice_tensor = self.dice_score.compute()
                # Check if dice values are in expected range
                if torch.any(dice_tensor > 1.5):
                    print(f"WARNING: Unusual dice values detected: {dice_tensor}")
                    # Apply clipping if values are abnormal
                    dice_tensor = torch.clamp(dice_tensor, 0.0, 1.0)
                dice = torch.mean(dice_tensor).item()  # Average across classes
            else:
                dice = self.dice_score.compute().item()
        except Exception as e:
            print(f"Error computing Dice: {e}")
            dice = 0.0

        try:
            iou = self.iou_score.compute().item()
        except Exception as e:
            print(f"Error computing IoU: {e}")
            iou = 0.0

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

        return {"iou": iou, "dice": dice, "f1": f1, "accuracy": accuracy}

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

class CombinedCEDiceLoss(nn.Module):
    def __init__(self, weights=None, num_classes=9, ce_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        # CE Loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # Dice Loss with per-sample calculation
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Initialize dice scores
        batch_dice = 0
        
        # Calculate dice per sample in batch
        for batch_idx in range(inputs.shape[0]):
            sample_dice = 0
            for class_idx in range(self.num_classes):
                pred = inputs_soft[batch_idx, class_idx]
                target = targets_one_hot[batch_idx, class_idx]
                
                intersection = (pred * target).sum()
                cardinality = pred.sum() + target.sum()
                
                if cardinality > 0:
                    sample_dice += (2. * intersection / (cardinality + 1e-5))
                else:
                    # Both prediction and target are empty for this class
                    sample_dice += 1.0
                    
            batch_dice += sample_dice / self.num_classes
        
        # Average across batch
        dice_coefficient = batch_dice / inputs.shape[0]
        dice_loss = 1 - dice_coefficient
        
        # Combine both losses
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# In your metrics.py or evaluation code:
def calculate_per_class_metrics(predictions, targets, num_classes):
    """Calculate metrics for each class separately"""
    per_class_iou = []
    per_class_dice = []
    
    for cls in range(num_classes):
        # Create binary masks
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        # Calculate IoU
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        iou = intersection / (union + 1e-6)
        
        # Calculate Dice
        dice = 2 * intersection / (pred_mask.sum() + target_mask.sum() + 1e-6)
        
        per_class_iou.append(float(iou))
        per_class_dice.append(float(dice))
    
    return {'per_class_iou': per_class_iou, 'per_class_dice': per_class_dice}

def calculate_sqrt_inverse_weights(class_counts):
    """Calculate sqrt-inverse frequency weights with proper normalization"""
    # Convert class_counts to numpy array if it's not already
    if isinstance(class_counts, dict):
        # If class_counts is a dictionary
        max_class = max(class_counts.keys()) + 1
        counts_array = np.zeros(max_class, dtype=np.float32)
        for cls, count in class_counts.items():
            counts_array[cls] = count
    else:
        # If class_counts is already an array-like object
        counts_array = np.array(class_counts, dtype=np.float32)
    
    # Calculate total pixels
    total_pixels = np.sum(counts_array)
    
    # Calculate sqrt-inverse weights
    num_classes = len(counts_array)
    weights = np.zeros(num_classes, dtype=np.float32)
    
    for cls in range(num_classes):
        # Add a small constant (1.0) to avoid division by zero
        weights[cls] = np.sqrt(total_pixels / (num_classes * max(counts_array[cls], 1.0)))
    
    # Convert to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    # Cap to avoid extreme values (limit to 3x the minimum non-zero weight)
    non_zero_weights = weights_tensor[weights_tensor > 0]
    if len(non_zero_weights) > 0:
        min_non_zero = torch.min(non_zero_weights)
        max_weight = min_non_zero * 10.0  # Allow up to 10x difference between weights
        weights_tensor = torch.clamp(weights_tensor, max=max_weight)
    
    # Normalize to maintain scale (sum to number of classes)
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
    
    return weights_tensor
