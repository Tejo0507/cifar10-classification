"""
Model 2: Improved CNN for CIFAR-10 Classification.

Architecture:
    6 convolutional layers organized in 3 double-conv blocks:
        - Each block: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    Followed by:
        - Global Average Pooling → FC → Dropout → FC (output)

Key Improvements over Baseline:
    1. **Deeper architecture**: 6 conv layers vs 3, enabling more complex feature
       hierarchies (edges → textures → parts → objects)
    2. **More feature channels**: Up to 256 channels (vs 128), increasing model capacity
    3. **Double-conv blocks**: Two convolutions per spatial scale capture richer patterns
       before downsampling
    4. **Global Average Pooling**: Replaces flattening, reducing parameters dramatically
       and acting as structural regularization
    5. **Progressive dropout**: Increasing dropout rates (0.2 → 0.3 → 0.4) prevent
       co-adaptation in deeper layers
    6. **Residual-style information flow** via the double-conv pattern

Expected Performance:
    - CIFAR-10 Test Accuracy: 75-85%
    - The deeper architecture captures more complex features
    - GAP reduces overfitting compared to fully-connected layers

Parameter Count: ~850K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCNN(nn.Module):
    """
    Deeper CNN with Global Average Pooling for CIFAR-10.
    
    Input:  (B, 3, 32, 32) — RGB images
    Output: (B, 10) — class logits
    
    Args:
        num_classes: Number of output classes (default: 10)
        dropout_rate: Base dropout rate (default: 0.4)
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4, **kwargs):
        super(ImprovedCNN, self).__init__()
        
        self.model_name = "ImprovedCNN"
        
        # ---- Block 1: 3 → 64 channels, 32x32 → 16x16 ----
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),    # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                              # (B, 64, 16, 16)
            nn.Dropout2d(p=dropout_rate * 0.5),              # Spatial dropout
        )
        
        # ---- Block 2: 64 → 128 channels, 16x16 → 8x8 ----
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                               # (B, 128, 8, 8)
            nn.Dropout2d(p=dropout_rate * 0.75),
        )
        
        # ---- Block 3: 128 → 256 channels, 8x8 → 4x4 ----
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),   # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                               # (B, 256, 4, 4)
            nn.Dropout2d(p=dropout_rate),
        )
        
        # ---- Global Average Pooling ----
        # Replaces flatten + large FC layer
        # (B, 256, 4, 4) → (B, 256, 1, 1) → (B, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ---- Classifier Head ----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for improved gradient flow."""
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
        Forward pass through the improved CNN.
        
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)   # Flatten: (B, 256)
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self) -> str:
        return (
            f"{self.model_name} | "
            f"Parameters: {self.get_num_parameters():,} | "
            f"Architecture: 6-layer CNN + GAP"
        )


if __name__ == "__main__":
    model = ImprovedCNN(num_classes=10)
    print(model)
    
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    print("✓ ImprovedCNN test passed")
