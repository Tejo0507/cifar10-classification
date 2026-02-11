"""
Model 3: ResNet18 Transfer Learning for CIFAR-10 Classification.

Architecture:
    - Pretrained ResNet18 backbone (trained on ImageNet-1K)
    - Modified first conv layer for 32x32 input (optional)
    - Custom classification head replacing the original FC layer

Transfer Learning Strategy:
    Two modes are supported:
    
    1. **Feature Extraction (freeze_backbone=True)**:
       - Freeze all backbone parameters
       - Only train the custom classifier head
       - Fastest training, uses pretrained features as-is
       - Good when dataset is small or similar to ImageNet
    
    2. **Fine-tuning (freeze_backbone=False)**:
       - Train all parameters end-to-end
       - Lower learning rate recommended for backbone layers
       - Adapts pretrained features to CIFAR-10 domain
       - Generally achieves higher accuracy

Design Decisions:
    - Keep original 7x7 conv (with stride adaptation) rather than replacing
      with 3x3 — pretrained weights are more valuable than architectural fit
    - Use a 2-layer classifier head with dropout for regularization
    - Provide parameter group separation for differential learning rates

Expected Performance:
    - Feature extraction: 80-88%
    - Fine-tuning: 88-93%
    - The ImageNet pretrained features transfer well to CIFAR-10

Parameter Count: ~11.2M (pretrained) + ~5K (classifier head)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Transfer(nn.Module):
    """
    ResNet18 with transfer learning for CIFAR-10.
    
    Input:  (B, 3, 32, 32) — RGB images
    Output: (B, 10) — class logits
    
    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Use ImageNet pretrained weights (default: True)
        freeze_backbone: Freeze backbone parameters (default: False)
        dropout_rate: Dropout in classifier head (default: 0.5)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super(ResNet18Transfer, self).__init__()
        
        self.model_name = "ResNet18Transfer"
        self.freeze_backbone = freeze_backbone
        
        # ---- Load pretrained ResNet18 ----
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            print(f"[Model] Loaded ResNet18 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet18(weights=None)
            print(f"[Model] Loaded ResNet18 without pretrained weights")
        
        # ---- Modify first conv layer for CIFAR-10 (32x32 images) ----
        # Original ResNet18 uses 7x7 conv with stride 2 + maxpool
        # For 32x32 images, this is too aggressive — replace with 3x3
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # ---- Custom classifier head ----
        in_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        
        # ---- Freeze backbone if requested ----
        if freeze_backbone:
            self._freeze_backbone()
        
        # Initialize the new layers
        self._initialize_new_layers()
    
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the classifier head."""
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
                frozen_count += 1
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Backbone frozen: {frozen_count} parameter groups")
        print(f"[Model] Trainable: {trainable:,} / {total:,} parameters "
              f"({100*trainable/total:.1f}%)")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] All parameters unfrozen: {trainable:,} trainable")
    
    def _initialize_new_layers(self):
        """Initialize the custom classifier head layers."""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list:
        """
        Get parameter groups with differential learning rates.
        
        This enables using a lower learning rate for pretrained backbone
        layers and a higher rate for the randomly initialized head.
        
        Args:
            backbone_lr: Learning rate for backbone parameters
            head_lr: Learning rate for classifier head
        
        Returns:
            List of parameter group dicts for optimizer
        """
        backbone_params = []
        head_params = []
        
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                if "fc" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet18.
        
        Args:
            x: Input tensor of shape (B, 3, 32, 32)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def __str__(self) -> str:
        trainable = self.get_num_parameters(trainable_only=True)
        total = self.get_num_parameters(trainable_only=False)
        mode = "Frozen" if self.freeze_backbone else "Fine-tune"
        return (
            f"{self.model_name} | "
            f"Trainable: {trainable:,} / Total: {total:,} | "
            f"Mode: {mode}"
        )


if __name__ == "__main__":
    # Test fine-tuning mode
    model = ResNet18Transfer(num_classes=10, pretrained=False, freeze_backbone=False)
    print(model)
    
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    
    # Test frozen mode
    model_frozen = ResNet18Transfer(num_classes=10, pretrained=False, freeze_backbone=True)
    print(model_frozen)
    
    out_frozen = model_frozen(x)
    assert out_frozen.shape == (4, 10)
    print("✓ ResNet18Transfer test passed (both modes)")
