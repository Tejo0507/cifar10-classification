"""
CIFAR-10 Classification — Evaluation Script

Evaluates a trained model checkpoint on the test set.
Generates metrics, confusion matrix, misclassified images, and reports.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/baseline_cnn_baseline_lr0.001_best.pth
    python evaluate.py --checkpoint outputs/checkpoints/resnet18_advanced_lr0.001_best.pth --model resnet18
"""

import argparse
import os
import sys
import json

import torch

from configs.config import get_config
from utils.seed import set_seed
from utils.device import get_device
from utils.data_loader import get_data_loaders
from models import get_model
from evaluation.evaluator import Evaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Image Classification — Evaluation"
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument("--model", type=str, default="baseline_cnn",
                        choices=["baseline_cnn", "improved_cnn", "resnet18"],
                        help="Model architecture (must match checkpoint)")
    parser.add_argument("--aug", type=str, default="baseline",
                        choices=["baseline", "advanced"],
                        help="Augmentation used during training (for correct normalization)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/results")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name for output files")
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    # ================================================================
    # 1. Setup
    # ================================================================
    set_seed(args.seed)
    device = get_device(args.device)
    
    experiment_name = args.name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    
    # ================================================================
    # 2. Config (matching training config)
    # ================================================================
    config = get_config(
        model_name=args.model,
        augmentation=args.aug,
        batch_size=args.batch_size,
        experiment_name=experiment_name,
    )
    
    # ================================================================
    # 3. Data Loading
    # ================================================================
    print("\n[Eval] Loading CIFAR-10 test data...")
    _, _, test_loader, data_info = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=args.aug,
        seed=args.seed,
        class_names=config.class_names,
    )
    
    # ================================================================
    # 4. Model Loading
    # ================================================================
    print(f"\n[Eval] Loading model: {args.model}")
    model = get_model(
        name=args.model,
        num_classes=config.model.num_classes,
        pretrained=False,  # Don't need pretrained weights, loading from checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Eval] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"[Eval] Checkpoint val acc: {checkpoint.get('best_val_acc', '?'):.2f}%")
    
    # ================================================================
    # 5. Evaluation
    # ================================================================
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=config.class_names,
        output_dir=args.output_dir,
    )
    
    print("\n[Eval] Running evaluation on test set...")
    metrics = evaluator.evaluate(test_loader)
    
    # Print detailed report
    evaluator.print_report(metrics, experiment_name)
    
    # Print sklearn classification report
    print("\n[Eval] Detailed Classification Report:")
    print(evaluator.generate_classification_report())
    
    # ================================================================
    # 6. Save Results
    # ================================================================
    evaluator.save_metrics(metrics, experiment_name)
    evaluator.plot_results(
        metrics, experiment_name,
        mean=data_info["mean"],
        std=data_info["std"],
    )
    
    print(f"\n[Eval] Evaluation complete!")
    print(f"[Eval] Test Accuracy: {metrics['accuracy']:.2f}%")
    print(f"[Eval] Results saved to: {args.output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
