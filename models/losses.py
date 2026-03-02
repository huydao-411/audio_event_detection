"""
Custom Loss Functions for Audio Event Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 7):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter
            num_classes: Number of classes
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        p_t = (targets_one_hot * probs).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss to prevent overconfidence
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss
        
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * self.confidence + self.smoothing / self.num_classes
        
        # Compute loss
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights for severe class imbalance
    """
    
    def __init__(self, 
                 alpha: float = 0.25, 
                 gamma: float = 2.0,
                 class_weights: torch.Tensor = None):
        """
        Initialize Weighted Focal Loss
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            class_weights: Per-class weights
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted focal loss
        
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Weighted focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 32
    num_classes = 7
    
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal_loss = FocalLoss(num_classes=num_classes)
    loss_value = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss_value.item():.4f}")
    
    # Test Label Smoothing Loss
    label_smoothing_loss = LabelSmoothingLoss(num_classes=num_classes)
    loss_value = label_smoothing_loss(inputs, targets)
    print(f"Label Smoothing Loss: {loss_value.item():.4f}")
    
    # Test Weighted Focal Loss
    class_weights = torch.ones(num_classes)
    weighted_focal_loss = WeightedFocalLoss(class_weights=class_weights)
    loss_value = weighted_focal_loss(inputs, targets)
    print(f"Weighted Focal Loss: {loss_value.item():.4f}")
    
    print("Loss functions test complete!")


if __name__ == "__main__":
    test_losses()
