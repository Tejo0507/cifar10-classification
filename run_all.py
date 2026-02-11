"""
Streamlined training script: trains all 3 models with optimal settings,
evaluates on test set, generates all plots and comparison tables.
"""
import os
import sys
import json
import time
from datetime import datetime

import torch

from configs.config import get_config
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


OUTPUT_DIR = "./outputs"
DATA_DIR = "./data"
SEED = 42


def run_experiment(config, device, train_loader, val_loader, test_loader, data_info):
    """Train a model and evaluate."""
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: {config.experiment_name}")
    print(f"{'#'*70}")
    print(config)
    
    start = time.time()
    
    # Model
    model = get_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
    )
    print(f"[Exp] {model}")
    
    # Train
    config.training.checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    trainer = Trainer(model, device, config)
    history = trainer.fit(train_loader, val_loader)
    
    # Save history + curves
    history_path = os.path.join(OUTPUT_DIR, "results", f"{config.experiment_name}_history.json")
    trainer.save_history(history_path)
    curves_path = os.path.join(OUTPUT_DIR, "plots", f"{config.experiment_name}_curves.png")
    plot_training_curves(history, config.experiment_name, save_path=curves_path)
    
    # Load best checkpoint and evaluate
    best_ckpt = os.path.join(OUTPUT_DIR, "checkpoints", f"{config.experiment_name}_best.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Exp] Loaded best checkpoint (epoch {ckpt['epoch']})")
    
    evaluator = Evaluator(
        model=model, device=device,
        class_names=config.class_names,
        output_dir=os.path.join(OUTPUT_DIR, "results"),
    )
    test_metrics = evaluator.evaluate(test_loader)
    evaluator.print_report(test_metrics, config.experiment_name)
    evaluator.save_metrics(test_metrics, config.experiment_name)
    evaluator.plot_results(
        test_metrics, config.experiment_name,
        mean=data_info["mean"], std=data_info["std"],
    )
    
    elapsed = time.time() - start
    best = trainer.get_best_results()
    
    return {
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


def main():
    print("=" * 70)
    print("  CIFAR-10 CLASSIFICATION — FULL EXPERIMENT SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    set_seed(SEED)
    device = get_device("auto")
    
    for subdir in ["checkpoints", "plots", "results", "configs"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
    
    # ============ Load data once ============
    # Advanced augmentation for training (reused across experiments)
    print("\n[Main] Loading data with ADVANCED augmentation...")
    train_adv, val_adv, test_adv, info_adv = get_data_loaders(
        data_dir=DATA_DIR, batch_size=128, augmentation="advanced", seed=SEED,
    )
    print("\n[Main] Loading data with BASELINE augmentation...")
    train_base, val_base, test_base, info_base = get_data_loaders(
        data_dir=DATA_DIR, batch_size=128, augmentation="baseline", seed=SEED,
    )
    
    # ============ Dataset visualizations ============
    print("\n[Main] Generating dataset visualizations...")
    images, labels, class_names = get_raw_samples(DATA_DIR, num_per_class=5)
    plot_sample_images(
        images, labels, class_names, num_per_class=5,
        save_path=os.path.join(OUTPUT_DIR, "plots", "sample_images.png"),
    )
    plot_class_distribution(
        {
            "Train": info_adv["train_distribution"],
            "Validation": info_adv["val_distribution"],
            "Test": info_adv["test_distribution"],
        },
        list(class_names),
        save_path=os.path.join(OUTPUT_DIR, "plots", "class_distribution.png"),
    )
    
    # ============ Define experiments ============
    experiments = [
        # Baseline CNN — baseline augmentation
        (
            get_config(
                model_name="baseline_cnn", augmentation="baseline",
                learning_rate=1e-3, epochs=50,
                experiment_name="baseline_cnn_baseline",
            ),
            train_base, val_base, test_base, info_base,
        ),
        # Baseline CNN — advanced augmentation
        (
            get_config(
                model_name="baseline_cnn", augmentation="advanced",
                learning_rate=1e-3, epochs=50,
                experiment_name="baseline_cnn_advanced",
            ),
            train_adv, val_adv, test_adv, info_adv,
        ),
        # Improved CNN — advanced augmentation
        (
            get_config(
                model_name="improved_cnn", augmentation="advanced",
                learning_rate=1e-3, epochs=60,
                experiment_name="improved_cnn_advanced",
            ),
            train_adv, val_adv, test_adv, info_adv,
        ),
        # Improved CNN — advanced augmentation, lower LR
        (
            get_config(
                model_name="improved_cnn", augmentation="advanced",
                learning_rate=5e-4, epochs=60,
                experiment_name="improved_cnn_advanced_lr5e-4",
            ),
            train_adv, val_adv, test_adv, info_adv,
        ),
        # ResNet18 — advanced augmentation
        (
            get_config(
                model_name="resnet18", augmentation="advanced",
                learning_rate=1e-3, epochs=40,
                experiment_name="resnet18_advanced",
            ),
            train_adv, val_adv, test_adv, info_adv,
        ),
        # ResNet18 — advanced augmentation, lower LR
        (
            get_config(
                model_name="resnet18", augmentation="advanced",
                learning_rate=5e-4, epochs=40,
                experiment_name="resnet18_advanced_lr5e-4",
            ),
            train_adv, val_adv, test_adv, info_adv,
        ),
    ]
    
    print(f"\n[Main] Running {len(experiments)} experiments")
    for i, (cfg, *_) in enumerate(experiments):
        print(f"  {i+1}. {cfg.experiment_name}")
    
    # ============ Run experiments ============
    all_results = []
    total_start = time.time()
    
    for i, (config, tl, vl, tel, di) in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT {i+1}/{len(experiments)}")
        print(f"{'='*70}")
        
        try:
            result = run_experiment(config, device, tl, vl, tel, di)
            all_results.append(result)
            print(f"\n  >>> Test Accuracy: {result['test_acc']:.2f}%")
        except Exception as e:
            print(f"[ERROR] {config.experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start
    
    # ============ Results ============
    # Comparison table
    header = (
        f"{'Model':<30} {'Aug':<10} {'LR':<10} "
        f"{'Val Acc':<10} {'Test Acc':<10} {'F1(M)':<10} "
        f"{'Epochs':<8} {'Time':<8}"
    )
    sep = "-" * len(header)
    lines = ["\n" + "=" * len(header), "  EXPERIMENT COMPARISON", "=" * len(header), header, sep]
    
    sorted_results = sorted(all_results, key=lambda x: x["test_acc"], reverse=True)
    for r in sorted_results:
        line = (
            f"{r['name']:<30} {r['augmentation']:<10} {r['lr']:<10.0e} "
            f"{r['val_acc']:<10.2f} {r['test_acc']:<10.2f} {r['f1_macro']:<10.2f} "
            f"{r['epochs_trained']:<8} {r['elapsed_minutes']:<8.1f}m"
        )
        lines.append(line)
    lines.append(sep)
    if sorted_results:
        lines.append(f"  Best: {sorted_results[0]['name']} -> {sorted_results[0]['test_acc']:.2f}%")
    lines.append("=" * len(header) + "\n")
    table = "\n".join(lines)
    print(table)
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, "results", "comparison_table.txt"), "w") as f:
        f.write(table)
    with open(os.path.join(OUTPUT_DIR, "results", "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Comparison plot
    plot_experiment_comparison(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, "plots", "experiment_comparison.png"),
    )
    
    # Error analysis
    print("\n[Main] Generating error analysis...")
    analysis = generate_error_analysis(
        all_results,
        output_path=os.path.join(OUTPUT_DIR, "results", "error_analysis.txt"),
    )
    print(analysis)
    
    # Final
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
