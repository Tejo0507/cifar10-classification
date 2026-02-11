"""
Visualization utilities for CIFAR-10 classification project.

Provides functions for:
    - Sample image grids per class
    - Training curve plots (loss & accuracy)
    - Confusion matrix heatmaps
    - Misclassified image analysis
    - Class distribution bar charts
    - Experiment comparison tables
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch
from sklearn.metrics import confusion_matrix


# Set global matplotlib style for publication-quality figures
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


# ==============================================================================
# Sample Image Visualization
# ==============================================================================

def plot_sample_images(
    images: list,
    labels: list,
    class_names: list,
    num_per_class: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a grid of sample images organized by class.
    
    Creates a 10 x num_per_class grid showing representative samples
    from each CIFAR-10 class, useful for dataset overview.
    
    Args:
        images: List of PIL images
        labels: Corresponding class labels
        class_names: Class name strings
        num_per_class: Columns per class
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(10, num_per_class, figsize=(num_per_class * 2, 20))
    fig.suptitle("CIFAR-10 Sample Images by Class", fontsize=16, fontweight="bold")
    
    for class_idx in range(10):
        class_images = [img for img, lbl in zip(images, labels) if lbl == class_idx]
        for j in range(min(num_per_class, len(class_images))):
            ax = axes[class_idx, j]
            ax.imshow(class_images[j])
            ax.axis("off")
            if j == 0:
                ax.set_title(class_names[class_idx], fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Sample images saved to {save_path}")
    plt.close()


# ==============================================================================
# Class Distribution
# ==============================================================================

def plot_class_distribution(
    distributions: Dict[str, Dict[str, int]],
    class_names: list,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot class distribution across train/val/test splits as grouped bar chart.
    
    Args:
        distributions: Dict of split_name -> {class_name: count}
        class_names: List of class names
        save_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    splits = list(distributions.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    
    for i, split_name in enumerate(splits):
        counts = [distributions[split_name].get(name, 0) for name in class_names]
        ax.bar(x + i * width, counts, width, label=split_name, color=colors[i], alpha=0.85)
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution Across Splits", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Class distribution saved to {save_path}")
    plt.close()


# ==============================================================================
# Training Curves
# ==============================================================================

def plot_training_curves(
    history: Dict[str, list],
    experiment_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Creates a 1x2 subplot figure with:
        - Left: Loss curves (train & validation)
        - Right: Accuracy curves (train & validation)
    
    Highlights the best validation accuracy epoch.
    
    Args:
        history: Dictionary with keys "train_loss", "val_loss",
                 "train_acc", "val_acc"
        experiment_name: Title prefix for the plot
        save_path: Output file path
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {experiment_name}", fontsize=14, fontweight="bold")
    
    # ---- Loss Curves ----
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    
    # Mark minimum validation loss
    min_val_loss_epoch = np.argmin(history["val_loss"]) + 1
    min_val_loss = min(history["val_loss"])
    ax1.axvline(x=min_val_loss_epoch, color="gray", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"Min: {min_val_loss:.4f}\nEpoch {min_val_loss_epoch}",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(min_val_loss_epoch + 2, min_val_loss + 0.1),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ---- Accuracy Curves ----
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Accuracy", linewidth=2)
    
    # Mark maximum validation accuracy
    max_val_acc_epoch = np.argmax(history["val_acc"]) + 1
    max_val_acc = max(history["val_acc"])
    ax2.axvline(x=max_val_acc_epoch, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(
        f"Best: {max_val_acc:.2f}%\nEpoch {max_val_acc_epoch}",
        xy=(max_val_acc_epoch, max_val_acc),
        xytext=(max_val_acc_epoch + 2, max_val_acc - 5),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"),
    )
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Training curves saved to {save_path}")
    plt.close()


# ==============================================================================
# Confusion Matrix
# ==============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    experiment_name: str = "",
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Plot a confusion matrix heatmap with optional normalization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class name strings
        experiment_name: Title prefix
        normalize: If True, normalize rows to percentages
        save_path: Output file path
    
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        fmt = ".1f"
        title_suffix = "(Normalized %)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = "(Counts)"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Percentage (%)" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix {title_suffix} — {experiment_name}",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


# ==============================================================================
# Misclassified Images
# ==============================================================================

def plot_misclassified(
    images: list,
    true_labels: list,
    pred_labels: list,
    class_names: list,
    num_images: int = 25,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2470, 0.2435, 0.2616),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize misclassified images with true and predicted labels.
    
    Displays a grid of incorrectly classified images, showing
    what the model predicted vs. the ground truth.
    
    Args:
        images: List of tensor images (C, H, W)
        true_labels: Ground truth labels
        pred_labels: Model predictions
        class_names: Class name strings
        num_images: Max number of images to display
        mean: Normalization mean (for denormalization)
        std: Normalization std (for denormalization)
        save_path: Output file path
    """
    num_images = min(num_images, len(images))
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    fig.suptitle("Misclassified Images", fontsize=14, fontweight="bold")
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < num_images:
            img = denormalize(images[idx], mean, std)
            ax.imshow(img)
            true_name = class_names[true_labels[idx]]
            pred_name = class_names[pred_labels[idx]]
            ax.set_title(f"T: {true_name}\nP: {pred_name}",
                        fontsize=8, color="red", fontweight="bold")
        ax.axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Misclassified images saved to {save_path}")
    plt.close()


# ==============================================================================
# Experiment Comparison
# ==============================================================================

def plot_experiment_comparison(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Create a bar chart comparing experiment results.
    
    Args:
        results: List of dicts with keys:
            "name", "val_accuracy", "test_accuracy"
        save_path: Output file path
    """
    names = [r["name"] for r in results]
    val_accs = [r["val_accuracy"] for r in results]
    test_accs = [r["test_accuracy"] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, val_accs, width, label="Val Accuracy",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy",
                   color="#4CAF50", alpha=0.85)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Experiment comparison saved to {save_path}")
    plt.close()


# ==============================================================================
# Helper Functions
# ==============================================================================

def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2470, 0.2435, 0.2616),
) -> np.ndarray:
    """
    Reverse normalization for visualization.
    
    Converts a normalized tensor image back to [0, 1] range for display.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        NumPy array in (H, W, C) format, clipped to [0, 1]
    """
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    return img
