"""
CIFAR-10 Classification — Training Script

Main entry point for training a single model with specified configuration.
Handles model creation, data loading, training, and saving results.

Usage:
    python train.py                                     # Default baseline CNN
    python train.py --model improved_cnn --aug advanced  # Improved CNN + augmentation
    python train.py --model resnet18 --lr 0.001          # ResNet18 transfer learning

All hyperparameters can be set via command-line arguments.
"""

import argparse
import os
import sys
import json

import torch

from configs.config import get_config
from utils.seed import set_seed
from utils.device import get_device
from utils.data_loader import get_data_loaders, get_raw_samples
from utils.visualization import plot_sample_images, plot_class_distribution, plot_training_curves
from models import get_model
from training.trainer import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Image Classification — Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model baseline_cnn --aug baseline --lr 0.001
  python train.py --model improved_cnn --aug advanced --lr 0.001 --epochs 80
  python train.py --model resnet18 --aug advanced --lr 0.001 --epochs 50
        """,
    )
    
    # Model
    parser.add_argument("--model", type=str, default="baseline_cnn",
                        choices=["baseline_cnn", "improved_cnn", "resnet18"],
                        help="Model architecture to train")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (default: model-specific)")
    
    # Data
    parser.add_argument("--aug", type=str, default="baseline",
                        choices=["baseline", "advanced"],
                        help="Data augmentation strategy")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"],
                        help="Optimizer algorithm")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "plateau"],
                        help="LR scheduler type")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    
    # Transfer learning
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze backbone (for ResNet18)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Don't use pretrained weights")
    
    # General
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Computation device")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--visualize-data", action="store_true",
                        help="Generate dataset visualization plots before training")
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # ================================================================
    # 1. Configuration
    # ================================================================
    config = get_config(
        model_name=args.model,
        augmentation=args.aug,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        experiment_name=args.name,
        freeze_backbone=args.freeze,
    )
    
    # Override with CLI arguments
    config.seed = args.seed
    config.device = args.device
    config.output_dir = args.output_dir
    config.data.data_dir = args.data_dir
    config.data.num_workers = args.workers
    config.training.optimizer = args.optimizer
    config.training.scheduler = args.scheduler
    config.training.weight_decay = args.weight_decay
    config.training.early_stopping = not args.no_early_stop
    config.training.early_stopping_patience = args.patience
    config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    
    if args.dropout is not None:
        config.model.dropout_rate = args.dropout
    if args.no_pretrained:
        config.model.pretrained = False
    
    print(config)
    
    # ================================================================
    # 2. Setup
    # ================================================================
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Save config for reproducibility
    config_path = os.path.join(config.output_dir, "configs", f"{config.experiment_name}.json")
    config.save(config_path)
    
    # ================================================================
    # 3. Data Loading
    # ================================================================
    print("\n[Main] Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader, data_info = get_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        augmentation=config.data.augmentation,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.seed,
        class_names=config.class_names,
    )
    
    # Optional: Generate dataset visualizations
    if args.visualize_data:
        print("\n[Main] Generating dataset visualizations...")
        plots_dir = os.path.join(config.output_dir, "plots")
        
        # Sample images
        images, labels, class_names = get_raw_samples(config.data.data_dir, num_per_class=5)
        plot_sample_images(
            images, labels, class_names, num_per_class=5,
            save_path=os.path.join(plots_dir, "sample_images.png"),
        )
        
        # Class distribution
        plot_class_distribution(
            {
                "Train": data_info["train_distribution"],
                "Validation": data_info["val_distribution"],
                "Test": data_info["test_distribution"],
            },
            list(config.class_names),
            save_path=os.path.join(plots_dir, "class_distribution.png"),
        )
    
    # ================================================================
    # 4. Model Creation
    # ================================================================
    print(f"\n[Main] Creating model: {config.model.name}")
    model = get_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
    )
    print(f"[Main] {model}")
    print(f"[Main] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Main] Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ================================================================
    # 5. Training
    # ================================================================
    trainer = Trainer(model, device, config)
    history = trainer.fit(train_loader, val_loader)
    
    # Save training history
    history_path = os.path.join(config.output_dir, "results", f"{config.experiment_name}_history.json")
    trainer.save_history(history_path)
    
    # Plot training curves
    curves_path = os.path.join(config.output_dir, "plots", f"{config.experiment_name}_curves.png")
    plot_training_curves(history, config.experiment_name, save_path=curves_path)
    
    # ================================================================
    # 6. Summary
    # ================================================================
    best = trainer.get_best_results()
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Experiment:     {config.experiment_name}")
    print(f"  Model:          {config.model.name}")
    print(f"  Augmentation:   {config.data.augmentation}")
    print(f"  Best Epoch:     {best['best_epoch']}")
    print(f"  Best Val Loss:  {best['best_val_loss']:.4f}")
    print(f"  Best Val Acc:   {best['best_val_acc']:.2f}%")
    print(f"  Total Epochs:   {best['total_epochs']}")
    print(f"  Checkpoint:     {config.training.checkpoint_dir}")
    print(f"{'='*70}")
    
    return best


if __name__ == "__main__":
    main()
