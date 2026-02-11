"""
Model 1: Baseline CNN for CIFAR-10 Classification.

Architecture:
    3 convolutional blocks, each containing:
        - Conv2d → BatchNorm2d → ReLU → MaxPool2d
    Followed by:
        - Flatten → FC → ReLU → Dropout → FC (output)

Design Rationale:
    - Simple 3-block CNN to establish a performance baseline (~65-75%)
    - BatchNorm accelerates training and provides mild regularization
    - Dropout prevents overfitting on the small dataset
    - MaxPooling progressively reduces spatial dimensions
    - ReLU activation for efficient gradient flow

Expected Performance:
    - CIFAR-10 Test Accuracy: 65-75%
    - This is limited by the shallow architecture and small receptive field

Parameter Count: ~190K (intentionally compact)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline 3-layer CNN for CIFAR-10.
    
    Input:  (B, 3, 32, 32) — RGB images
    Output: (B, 10) — class logits
    
    Args:
        num_classes: Number of output classes (default: 10)
        dropout_rate: Dropout probability in FC layers (default: 0.5)
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5, **kwargs):
        super(BaselineCNN, self).__init__()
        
        self.model_name = "BaselineCNN"
        
        # ---- Convolutional Feature Extractor ----
        # Block 1: 3 → 32 channels, 32x32 → 16x16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # (B, 32, 16, 16)
        )
        
        # Block 2: 32 → 64 channels, 16x16 → 8x8
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # (B, 64, 8, 8)
        )
        
        # Block 3: 64 → 128 channels, 8x8 → 4x4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # (B, 128, 4, 4)
        )
        
        # ---- Classifier Head ----
        # Feature size: 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self) -> str:
        return (
            f"{self.model_name} | "
            f"Parameters: {self.get_num_parameters():,} | "
            f"Architecture: 3-block CNN"
        )


if __name__ == "__main__":
    # Quick test
    model = BaselineCNN(num_classes=10)
    print(model)
    
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print("✓ BaselineCNN test passed")
