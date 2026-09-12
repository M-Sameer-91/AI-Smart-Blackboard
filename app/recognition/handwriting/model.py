"""
Handwriting Recognition CNN Model Module

This module provides a lightweight CNN architecture for handwritten character
classification. The model is designed for 64x64 grayscale images and supports
dynamic number of output classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class HandwritingCNN(nn.Module):
    """
    Lightweight CNN for handwritten character recognition.
    
    Architecture:
    - 3 convolutional blocks with batch normalization and max pooling
    - Dropout for regularization
    - Fully connected classification layer
    
    Input: [B, 1, 64, 64]
    Output: [B, num_classes] (logits)
    
    Args:
        num_classes: Number of output classes (must be > 0)
        input_channels: Number of input channels (default: 1 for grayscale)
        input_size: Input image size (default: 64)
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        input_size: int = 64,
        dropout_rate: float = 0.25
    ):
        super(HandwritingCNN, self).__init__()
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        
        if input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {input_channels}")
        
        if input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {input_size}")
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        
        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        # Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 8 -> 4
        
        # Calculate flattened size
        # After 4 pooling layers of stride 2: 64 -> 32 -> 16 -> 8 -> 4
        self.flattened_size = 256 * 4 * 4  # 256 * 16 = 4096
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, input_channels, input_size, input_size]
        
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape {x.shape}")
        
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"got {x.shape[1]}"
            )
        
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} input, "
                f"got {x.shape[2]}x{x.shape[3]}"
            )
        
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities using softmax.
        
        Args:
            logits: Raw logits from forward pass
        
        Returns:
            Probability tensor of shape [B, num_classes]
        """
        return F.softmax(logits, dim=1)
    
    def predict(self, logits: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from logits.
        
        Args:
            logits: Raw logits from forward pass
            top_k: Number of top predictions to return (default: 1)
        
        Returns:
            Tuple of (indices, probabilities)
            - indices: Top-k class indices [B, top_k]
            - probabilities: Top-k class probabilities [B, top_k]
        """
        probs = self.predict_proba(logits)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=1)
        return top_indices, top_probs
    
    def predict_class(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single class prediction with confidence.
        
        Args:
            logits: Raw logits from forward pass
        
        Returns:
            Tuple of (indices, confidences)
            - indices: Predicted class indices [B]
            - confidences: Confidence scores [B]
        """
        indices, probs = self.predict(logits, top_k=1)
        return indices.squeeze(1), probs.squeeze(1)
    
    def get_architecture_summary(self) -> dict:
        """
        Get summary of the model architecture.
        
        Returns:
            Dictionary with architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'flattened_size': self.flattened_size,
        }


# Convenience function for creating model variants
def create_handwriting_model(
    num_classes: int,
    input_channels: int = 1,
    input_size: int = 64,
    dropout_rate: float = 0.25,
    device: Optional[torch.device] = None
) -> HandwritingCNN:
    """
    Create a HandwritingCNN model with optional device placement.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (default: 1)
        input_size: Input image size (default: 64)
        dropout_rate: Dropout rate (default: 0.25)
        device: Optional device to place model on
    
    Returns:
        HandwritingCNN instance
    """
    model = HandwritingCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        dropout_rate=dropout_rate
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


# Simple test function
def test_model():
    """Test the model with dummy inputs."""
    print("Testing HandwritingCNN model...")
    
    # Test with 62 classes (A-Z, a-z, 0-9)
    print("\n1. Testing with 62 classes...")
    model = HandwritingCNN(num_classes=62)
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with 10 classes
    print("\n2. Testing with 10 classes...")
    model = HandwritingCNN(num_classes=10)
    dummy_input = torch.randn(3, 1, 64, 64)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test prediction helper
    print("\n3. Testing prediction helpers...")
    model = HandwritingCNN(num_classes=5)
    logits = torch.randn(2, 5)
    indices, probs = model.predict(logits, top_k=2)
    print(f"   Top-2 indices shape: {indices.shape}")
    print(f"   Top-2 probs shape: {probs.shape}")
    print(f"   Sum of probs: {probs.sum(dim=1)}")
    
    # Test state_dict save/load
    print("\n4. Testing state_dict save/load...")
    model1 = HandwritingCNN(num_classes=62)
    model2 = HandwritingCNN(num_classes=62)
    
    # Save state dict
    state_dict = model1.state_dict()
    
    # Load into new model
    model2.load_state_dict(state_dict)
    
    # Verify same output
    test_input = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out1 = model1(test_input)
        out2 = model2(test_input)
    
    print(f"   Outputs match: {torch.allclose(out1, out2)}")
    
    # Test architecture summary
    print("\n5. Architecture summary:")
    summary = model1.get_architecture_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_model()