"""
Configuration management for CIFAR-10 classification experiments.

Provides a centralized, dataclass-based configuration system that supports
reproducible experimentation with different hyperparameters, model architectures,
and data augmentation strategies.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os


@dataclass
class DataConfig:
    """Configuration for dataset handling and augmentation."""
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    augmentation: str = "baseline"  # "baseline" or "advanced"
    
    # CIFAR-10 computed statistics (will be recalculated if needed)
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)


@dataclass
class ModelConfig:
    """Configuration for model architecture selection."""
    name: str = "baseline_cnn"  # "baseline_cnn", "improved_cnn", "resnet18"
    num_classes: int = 10
    dropout_rate: float = 0.5
    
    # Transfer learning specific
    pretrained: bool = True
    freeze_backbone: bool = False  # If True, only train classifier head


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "./outputs/checkpoints"
    save_best_only: bool = True
    
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = True


@dataclass
class ExperimentConfig:
    """
    Master configuration combining all sub-configs.
    
    This is the single entry point for experiment configuration.
    All hyperparameters, model choices, and data settings are
    encapsulated here for reproducibility and easy modification.
    """
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    output_dir: str = "./outputs"
    log_interval: int = 50  # Log every N batches
    
    # CIFAR-10 class names
    class_names: tuple = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )
    
    def save(self, path: str) -> None:
        """Serialize configuration to JSON for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_dict = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        config = cls()
        config.experiment_name = config_dict.get("experiment_name", "loaded_experiment")
        config.seed = config_dict.get("seed", 42)
        config.device = config_dict.get("device", "auto")
        
        for key, value in config_dict.get("data", {}).items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
        
        for key, value in config_dict.get("model", {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        
        for key, value in config_dict.get("training", {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
        
        return config
    
    def __str__(self) -> str:
        """Readable string representation of the configuration."""
        lines = [
            f"{'='*60}",
            f"  Experiment: {self.experiment_name}",
            f"{'='*60}",
            f"  Seed: {self.seed} | Device: {self.device}",
            f"  Model: {self.model.name} | Dropout: {self.model.dropout_rate}",
            f"  Augmentation: {self.data.augmentation}",
            f"  Batch Size: {self.data.batch_size}",
            f"  LR: {self.training.learning_rate} | Epochs: {self.training.epochs}",
            f"  Optimizer: {self.training.optimizer} | Scheduler: {self.training.scheduler}",
            f"  Weight Decay: {self.training.weight_decay}",
            f"  Early Stopping: {self.training.early_stopping} (patience={self.training.early_stopping_patience})",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def get_config(
    model_name: str = "baseline_cnn",
    augmentation: str = "baseline",
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 128,
    experiment_name: Optional[str] = None,
    **kwargs
) -> ExperimentConfig:
    """
    Factory function to create experiment configurations.
    
    This is the recommended way to create configs for experiments.
    It sets sensible defaults per model type and allows overrides.
    
    Args:
        model_name: One of "baseline_cnn", "improved_cnn", "resnet18"
        augmentation: One of "baseline", "advanced"
        learning_rate: Initial learning rate
        epochs: Maximum training epochs
        batch_size: Batch size for data loaders
        experiment_name: Human-readable experiment identifier
        **kwargs: Additional overrides
    
    Returns:
        ExperimentConfig with all parameters set
    """
    if experiment_name is None:
        experiment_name = f"{model_name}_{augmentation}_lr{learning_rate}"
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        data=DataConfig(
            batch_size=batch_size,
            augmentation=augmentation,
        ),
        model=ModelConfig(name=model_name),
        training=TrainingConfig(
            learning_rate=learning_rate,
            epochs=epochs,
        ),
    )
    
    # Model-specific defaults
    if model_name == "baseline_cnn":
        config.model.dropout_rate = 0.5
        config.training.weight_decay = 1e-4
    elif model_name == "improved_cnn":
        config.model.dropout_rate = 0.4
        config.training.weight_decay = 5e-4
    elif model_name == "resnet18":
        config.model.pretrained = True
        config.model.freeze_backbone = kwargs.get("freeze_backbone", False)
        config.training.weight_decay = 1e-4
        config.training.scheduler = "cosine"
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config
