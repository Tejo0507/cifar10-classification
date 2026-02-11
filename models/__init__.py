"""Model architectures for CIFAR-10 classification."""
from models.baseline_cnn import BaselineCNN
from models.improved_cnn import ImprovedCNN
from models.resnet_transfer import ResNet18Transfer


def get_model(name: str, num_classes: int = 10, **kwargs):
    """
    Model factory function.
    
    Args:
        name: Model identifier — "baseline_cnn", "improved_cnn", or "resnet18"
        num_classes: Number of output classes
        **kwargs: Model-specific arguments (dropout_rate, pretrained, etc.)
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model name is not recognized
    """
    models = {
        "baseline_cnn": BaselineCNN,
        "improved_cnn": ImprovedCNN,
        "resnet18": ResNet18Transfer,
    }
    
    if name not in models:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(models.keys())}"
        )
    
    return models[name](num_classes=num_classes, **kwargs)
