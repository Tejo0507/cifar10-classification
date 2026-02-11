"""
CIFAR-10 Classification — Full Experiment Pipeline

Runs a comprehensive set of experiments comparing:
    - Baseline CNN vs Improved CNN vs ResNet18
    - Baseline vs Advanced augmentation
    - Different learning rates

Generates a comparison table, training curves, confusion matrices,
and a detailed error analysis report.

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --quick            # Quick mode (fewer epochs)
    python run_experiments.py --models resnet18   # Run only ResNet18 experiments
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict

import torch

from configs.config import get_config, ExperimentConfig
from utils.seed import set_seed
from utils.device import get_device
from utils.data_loader import get_data_loaders, get_raw_samples
from utils.visualization import (
    plot_sample_images,
    plot_class_distribution,
    plot_training_curves,
    plot_experiment_comparison,
)
from models import get_model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator, generate_error_analysis


# ==============================================================================
# Experiment Definitions
# ==============================================================================

def get_experiment_configs(quick: bool = False, models: list = None) -> List[ExperimentConfig]:
    """
    Define all experiments to run.
    
    Args:
        quick: If True, reduce epochs for faster testing
        models: If provided, only include these model types
    
    Returns:
        List of ExperimentConfig objects
    """
    max_epochs = 20 if quick else 100
    
    all_experiments = [
        # ---- Baseline CNN Experiments ----
        get_config(
            model_name="baseline_cnn",
            augmentation="baseline",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="baseline_cnn_baseline_lr1e-3",
        ),
        get_config(
            model_name="baseline_cnn",
            augmentation="advanced",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="baseline_cnn_advanced_lr1e-3",
        ),
        get_config(
            model_name="baseline_cnn",
            augmentation="advanced",
            learning_rate=5e-4,
            epochs=max_epochs,
            experiment_name="baseline_cnn_advanced_lr5e-4",
        ),
        
        # ---- Improved CNN Experiments ----
        get_config(
            model_name="improved_cnn",
            augmentation="baseline",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="improved_cnn_baseline_lr1e-3",
        ),
        get_config(
            model_name="improved_cnn",
            augmentation="advanced",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="improved_cnn_advanced_lr1e-3",
        ),
        get_config(
            model_name="improved_cnn",
            augmentation="advanced",
            learning_rate=5e-4,
            epochs=max_epochs,
            experiment_name="improved_cnn_advanced_lr5e-4",
        ),
        
        # ---- ResNet18 Transfer Learning Experiments ----
        get_config(
            model_name="resnet18",
            augmentation="advanced",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="resnet18_advanced_lr1e-3",
        ),
        get_config(
            model_name="resnet18",
            augmentation="advanced",
            learning_rate=5e-4,
            epochs=max_epochs,
            experiment_name="resnet18_advanced_lr5e-4",
        ),
        # Frozen backbone comparison
        get_config(
            model_name="resnet18",
            augmentation="advanced",
            learning_rate=1e-3,
            epochs=max_epochs,
            experiment_name="resnet18_frozen_advanced_lr1e-3",
            freeze_backbone=True,
        ),
    ]
    
    # Filter by requested models
    if models:
        all_experiments = [
            exp for exp in all_experiments
            if exp.model.name in models
        ]
    
    return all_experiments


# ==============================================================================
# Single Experiment Runner
# ==============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str = "./outputs",
) -> Dict:
    """
    Run a single experiment: train + evaluate.
    
    Args:
        config: Experiment configuration
        device: Computation device
        output_dir: Base output directory
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: {config.experiment_name}")
    print(f"{'#'*70}")
    print(config)
    
    start_time = time.time()
    
    # ---- Data ----
    train_loader, val_loader, test_loader, data_info = get_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        augmentation=config.data.augmentation,
        seed=config.seed,
        class_names=config.class_names,
    )
    
    # ---- Model ----
    model = get_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
    )
    print(f"[Experiment] {model}")
    
    # ---- Train ----
    config.training.checkpoint_dir = os.path.join(output_dir, "checkpoints")
    trainer = Trainer(model, device, config)
    history = trainer.fit(train_loader, val_loader)
    
    # Save training history and curves
    history_path = os.path.join(output_dir, "results", f"{config.experiment_name}_history.json")
    trainer.save_history(history_path)
    
    curves_path = os.path.join(output_dir, "plots", f"{config.experiment_name}_curves.png")
    plot_training_curves(history, config.experiment_name, save_path=curves_path)
    
    # ---- Evaluate on test set ----
    # Load best checkpoint
    best_ckpt_path = os.path.join(
        output_dir, "checkpoints", f"{config.experiment_name}_best.pth"
    )
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Experiment] Loaded best checkpoint (epoch {checkpoint['epoch']})")
    
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=config.class_names,
        output_dir=os.path.join(output_dir, "results"),
    )
    
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_report(test_metrics, config.experiment_name)
    evaluator.save_metrics(test_metrics, config.experiment_name)
    evaluator.plot_results(
        test_metrics, config.experiment_name,
        mean=data_info["mean"],
        std=data_info["std"],
    )
    
    elapsed = time.time() - start_time
    
    # ---- Compile results ----
    best = trainer.get_best_results()
    result = {
        "name": config.experiment_name,
        "model": config.model.name,
        "augmentation": config.data.augmentation,
        "lr": config.training.learning_rate,
        "train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
        "val_acc": best["best_val_acc"],
        "test_acc": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "f1_weighted": test_metrics["f1_weighted"],
        "precision_macro": test_metrics["precision_macro"],
        "recall_macro": test_metrics["recall_macro"],
        "per_class_accuracy": test_metrics["per_class_accuracy"],
        "most_confused_pairs": test_metrics["most_confused_pairs"],
        "epochs_trained": best["total_epochs"],
        "best_epoch": best["best_epoch"],
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    
    return result


# ==============================================================================
# Results Formatting
# ==============================================================================

def print_comparison_table(results: List[Dict]) -> str:
    """
    Print a formatted comparison table of all experiments.
    
    Args:
        results: List of experiment result dictionaries
    
    Returns:
        Formatted table as a string
    """
    # Header
    header = (
        f"{'Model':<20} {'Augmentation':<14} {'LR':<10} "
        f"{'Val Acc':<10} {'Test Acc':<10} {'F1 (M)':<10} "
        f"{'Epochs':<8} {'Time':<8}"
    )
    separator = "-" * len(header)
    
    lines = [
        "\n" + "=" * len(header),
        "  EXPERIMENT COMPARISON TABLE",
        "=" * len(header),
        header,
        separator,
    ]
    
    # Sort by test accuracy descending
    sorted_results = sorted(results, key=lambda x: x["test_acc"], reverse=True)
    
    for r in sorted_results:
        line = (
            f"{r['model']:<20} {r['augmentation']:<14} {r['lr']:<10.0e} "
            f"{r['val_acc']:<10.2f} {r['test_acc']:<10.2f} {r['f1_macro']:<10.2f} "
            f"{r['epochs_trained']:<8} {r['elapsed_minutes']:<8.1f}m"
        )
        lines.append(line)
    
    lines.append(separator)
    lines.append(f"  Best: {sorted_results[0]['name']} -> {sorted_results[0]['test_acc']:.2f}%")
    lines.append("=" * len(header) + "\n")
    
    table = "\n".join(lines)
    print(table)
    return table


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full CIFAR-10 experiment suite"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer epochs (for testing)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=["baseline_cnn", "improved_cnn", "resnet18"],
                        help="Only run experiments for these models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--visualize-data", action="store_true",
                        help="Generate dataset visualization plots")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  CIFAR-10 CLASSIFICATION — FULL EXPERIMENT SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Create output directories
    for subdir in ["checkpoints", "plots", "results", "configs"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # ---- Dataset Visualization ----
    if args.visualize_data:
        print("\n[Main] Generating dataset visualizations...")
        
        images, labels, class_names = get_raw_samples(args.data_dir, num_per_class=5)
        plot_sample_images(
            images, labels, class_names, num_per_class=5,
            save_path=os.path.join(args.output_dir, "plots", "sample_images.png"),
        )
        
        # Quick load for class distribution
        _, _, _, data_info = get_data_loaders(
            data_dir=args.data_dir, batch_size=128, seed=args.seed,
        )
        plot_class_distribution(
            {
                "Train": data_info["train_distribution"],
                "Validation": data_info["val_distribution"],
                "Test": data_info["test_distribution"],
            },
            class_names,
            save_path=os.path.join(args.output_dir, "plots", "class_distribution.png"),
        )
    
    # ---- Define Experiments ----
    experiments = get_experiment_configs(quick=args.quick, models=args.models)
    print(f"\n[Main] Running {len(experiments)} experiments")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp.experiment_name}")
    
    # ---- Run Experiments ----
    all_results = []
    total_start = time.time()
    
    for i, config in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT {i+1}/{len(experiments)}")
        print(f"{'='*70}")
        
        config.data.data_dir = args.data_dir
        
        try:
            result = run_single_experiment(config, device, args.output_dir)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Experiment {config.experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start
    
    # ================================================================
    # Results Summary
    # ================================================================
    
    # Print comparison table
    table_str = print_comparison_table(all_results)
    
    # Save comparison table
    table_path = os.path.join(args.output_dir, "results", "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    
    # Save all results as JSON
    results_path = os.path.join(args.output_dir, "results", "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[Main] All results saved to {results_path}")
    
    # Plot experiment comparison
    plot_experiment_comparison(
        all_results,
        save_path=os.path.join(args.output_dir, "plots", "experiment_comparison.png"),
    )
    
    # ---- Error Analysis ----
    print("\n[Main] Generating error analysis...")
    analysis = generate_error_analysis(
        all_results,
        output_path=os.path.join(args.output_dir, "results", "error_analysis.txt"),
    )
    print(analysis)
    
    # ---- Final Summary ----
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {args.output_dir}/results/")
    print(f"  Plots: {args.output_dir}/plots/")
    print(f"  Checkpoints: {args.output_dir}/checkpoints/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
