"""
Data loading and preprocessing pipeline for CIFAR-10 (Kaggle format).

Handles:
    - Loading PNG images + CSV labels from Kaggle CIFAR-10 dataset
    - Train/Validation/Test splitting (70/15/15)
    - Dataset statistics computation (mean & std)
    - Baseline and advanced augmentation pipelines
    - Class distribution analysis
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, Optional, List
from collections import Counter


# CIFAR-10 class name to index mapping (alphabetical order matches standard CIFAR-10)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
CIFAR10_LABEL_TO_IDX = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}


# ==============================================================================
# Kaggle CIFAR-10 Dataset
# ==============================================================================

class KaggleCIFAR10Dataset(Dataset):
    """
    Custom PyTorch Dataset for Kaggle-format CIFAR-10.
    
    Expects:
        - A CSV file with columns: id, label
        - A directory of PNG images named {id}.png
    
    Args:
        image_dir: Path to directory containing {id}.png images
        labels_csv: Path to CSV file with id,label columns
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_dir = image_dir
        self.transform = transform
        
        # Parse CSV labels
        self.samples = []  # List of (image_path, label_idx)
        self.targets = []  # List of label indices (for stratification/analysis)
        
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["id"]
                label_str = row["label"].strip()
                label_idx = CIFAR10_LABEL_TO_IDX[label_str]
                
                img_path = os.path.join(image_dir, f"{img_id}.png")
                self.samples.append((img_path, label_idx))
                self.targets.append(label_idx)
        
        print(f"[Data] Loaded {len(self.samples)} samples from {labels_csv}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# ==============================================================================
# Dataset Statistics Computation
# ==============================================================================

def compute_dataset_statistics(dataset: Dataset) -> Tuple[tuple, tuple]:
    """
    Compute per-channel mean and standard deviation of the dataset.
    
    Uses the raw pixel values (normalized to [0, 1] by ToTensor) to compute
    statistics that will be used for normalization during training and inference.
    
    Args:
        dataset: Dataset with ToTensor transform applied.
    
    Returns:
        Tuple of (mean, std), each a tuple of 3 floats (per RGB channel).
    """
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)  # (B, C, H*W)
        mean += images.mean(dim=[0, 2]) * batch_size
        std += images.std(dim=[0, 2]) * batch_size
        n_samples += batch_size
    
    mean /= n_samples
    std /= n_samples
    
    mean_tuple = tuple(mean.tolist())
    std_tuple = tuple(std.tolist())
    
    print(f"[Data] Computed mean: {mean_tuple}")
    print(f"[Data] Computed std:  {std_tuple}")
    
    return mean_tuple, std_tuple


# ==============================================================================
# Transform Pipelines
# ==============================================================================

def get_transforms(
    augmentation: str = "baseline",
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple = (0.2470, 0.2435, 0.2616),
    is_train: bool = True,
) -> transforms.Compose:
    """
    Build transformation pipelines for training and evaluation.
    
    Two augmentation strategies are supported:
    
    **Baseline**: Minimal augmentation (random horizontal flip only)
        - Provides a controlled comparison point
        - Tests model capacity without augmentation boost
    
    **Advanced**: Aggressive augmentation for regularization
        - RandomCrop with padding (spatial invariance)
        - RandomHorizontalFlip (orientation invariance)
        - RandomRotation (rotation invariance)
        - ColorJitter (illumination invariance)
        - RandomErasing (occlusion robustness)
    
    Args:
        augmentation: "baseline" or "advanced"
        mean: Per-channel mean for normalization
        std: Per-channel std for normalization
        is_train: If True, apply data augmentation; if False, only normalize
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if not is_train:
        # Evaluation/test: no augmentation, only normalize
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    if augmentation == "baseline":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation == "advanced":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation strategy: {augmentation}")


# ==============================================================================
# Dataset Splitting
# ==============================================================================

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train/validation/test subsets.
    
    Uses random splitting with deterministic seed for reproducibility.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training (default 0.70)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Remainder to test
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"[Data] Split sizes — Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_subset, val_subset, test_subset


# ==============================================================================
# Class Distribution Analysis
# ==============================================================================

def analyze_class_distribution(
    dataset,
    class_names: tuple,
    split_name: str = "Full",
) -> Dict[str, int]:
    """
    Analyze and report class distribution in a dataset split.
    
    Args:
        dataset: Dataset or Subset to analyze
        class_names: Tuple of class name strings
        split_name: Name of the split for logging
    
    Returns:
        Dictionary mapping class names to counts
    """
    if isinstance(dataset, Subset):
        # Get targets from parent dataset via indices
        parent = dataset.dataset
        if hasattr(parent, 'targets'):
            targets = [parent.targets[i] for i in dataset.indices]
        else:
            # Fallback: iterate
            targets = [parent[i][1] for i in dataset.indices]
    elif hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    counter = Counter(targets)
    distribution = {}
    
    print(f"\n[Data] Class distribution for {split_name} split:")
    print(f"  {'Class':<15} {'Count':>6} {'Percentage':>10}")
    print(f"  {'-'*35}")
    
    for class_idx in sorted(counter.keys()):
        name = class_names[class_idx]
        count = counter[class_idx]
        pct = 100.0 * count / len(targets)
        distribution[name] = count
        print(f"  {name:<15} {count:>6} {pct:>9.1f}%")
    
    print(f"  {'-'*35}")
    print(f"  {'Total':<15} {len(targets):>6}")
    
    return distribution


# ==============================================================================
# Data Loader Factory
# ==============================================================================

def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    augmentation: str = "baseline",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    class_names: tuple = CIFAR10_CLASSES,
    compute_stats: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Complete data loading pipeline: detect dataset, split, transform, and wrap in DataLoaders.
    
    This is the main entry point for data preparation. It handles:
    1. Auto-detecting Kaggle CIFAR-10 dataset location
    2. Loading PNG images with CSV labels
    3. Optionally computing dataset statistics
    4. Creating train/val/test splits
    5. Applying appropriate transformations
    6. Wrapping in DataLoaders with proper settings
    
    Args:
        data_dir: Base directory for dataset storage
        batch_size: Batch size for all loaders
        augmentation: "baseline" or "advanced" augmentation strategy
        train_ratio: Training split fraction
        val_ratio: Validation split fraction
        test_ratio: Test split fraction
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        class_names: CIFAR-10 class names
        compute_stats: Whether to recompute mean/std from data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict)
        info_dict contains split sizes, class distributions, and transform info
    """
    # ---- Step 1: Auto-detect dataset location ----
    image_dir, labels_csv = _find_kaggle_dataset(data_dir)
    
    # ---- Step 2: Load full dataset (no transform, for splitting) ----
    full_dataset = KaggleCIFAR10Dataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        transform=None,  # No transform — will apply via TransformedSubset
    )
    
    # ---- Step 3: Compute or use precomputed statistics ----
    if compute_stats:
        stats_dataset = KaggleCIFAR10Dataset(
            image_dir=image_dir,
            labels_csv=labels_csv,
            transform=transforms.ToTensor(),
        )
        mean, std = compute_dataset_statistics(stats_dataset)
        del stats_dataset
    else:
        # Standard CIFAR-10 statistics
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    
    # ---- Step 4: Split into train/val/test ----
    train_subset, val_subset, test_subset = split_dataset(
        full_dataset, train_ratio, val_ratio, test_ratio, seed
    )
    
    # ---- Step 5: Analyze class distributions ----
    train_dist = analyze_class_distribution(train_subset, class_names, "Train")
    val_dist = analyze_class_distribution(val_subset, class_names, "Validation")
    test_dist = analyze_class_distribution(test_subset, class_names, "Test")
    
    # ---- Step 6: Create transform-aware dataset wrappers ----
    train_transform = get_transforms(augmentation, mean, std, is_train=True)
    eval_transform = get_transforms(augmentation, mean, std, is_train=False)
    
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=eval_transform)
    test_dataset = TransformedSubset(test_subset, transform=eval_transform)
    
    # ---- Step 7: Create DataLoaders ----
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    effective_workers = num_workers if os.name != 'nt' else 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=effective_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_workers, pin_memory=pin_memory
    )
    
    info = {
        "mean": mean,
        "std": std,
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_subset),
        "train_distribution": train_dist,
        "val_distribution": val_dist,
        "test_distribution": test_dist,
        "augmentation": augmentation,
        "batch_size": batch_size,
    }
    
    print(f"\n[Data] DataLoaders ready — Augmentation: {augmentation}")
    print(f"[Data] Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, info


# ==============================================================================
# Auto-detect Dataset Path
# ==============================================================================

def _find_kaggle_dataset(data_dir: str) -> Tuple[str, str]:
    """
    Auto-detect the Kaggle CIFAR-10 dataset structure.
    
    Searches for trainLabels.csv and the train/ image directory
    under common Kaggle dataset layouts.
    
    Args:
        data_dir: Base data directory
    
    Returns:
        Tuple of (image_dir, labels_csv_path)
    
    Raises:
        FileNotFoundError: If dataset files cannot be located
    """
    search_paths = [
        data_dir,
        os.path.join(data_dir, "cifar-10"),
        os.path.join(data_dir, "cifar10"),
        os.path.join(data_dir, "cifar-10-python"),
    ]
    
    for base in search_paths:
        csv_path = os.path.join(base, "trainLabels.csv")
        train_dir = os.path.join(base, "train")
        
        if os.path.isfile(csv_path) and os.path.isdir(train_dir):
            sample_files = os.listdir(train_dir)[:5]
            if any(f.endswith(".png") for f in sample_files):
                print(f"[Data] Found Kaggle CIFAR-10 at: {base}")
                return train_dir, csv_path
    
    raise FileNotFoundError(
        f"Could not find Kaggle CIFAR-10 dataset. "
        f"Expected trainLabels.csv and train/ directory under one of: {search_paths}. "
        f"Please ensure the dataset is extracted correctly."
    )


# ==============================================================================
# Transformed Subset Wrapper
# ==============================================================================

class TransformedSubset(Dataset):
    """
    Wraps a Subset with a custom transform.
    
    This is necessary because torch Subsets inherit the transform of the
    parent dataset. We need different transforms for train vs. eval subsets
    derived from the same parent.
    """
    
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image is a PIL Image (parent dataset has no transform)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ==============================================================================
# Raw Image Retrieval (for visualization)
# ==============================================================================

def get_raw_samples(
    data_dir: str = "./data",
    num_per_class: int = 5,
) -> Tuple[list, list, list]:
    """
    Retrieve raw (unnormalized) sample images for visualization.
    
    Args:
        data_dir: Dataset directory
        num_per_class: Number of samples per class to retrieve
    
    Returns:
        Tuple of (images, labels, class_names)
        images is a list of PIL Images
    """
    image_dir, labels_csv = _find_kaggle_dataset(data_dir)
    dataset = KaggleCIFAR10Dataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        transform=None,
    )
    class_names = list(CIFAR10_CLASSES)
    
    # Collect samples per class
    class_images = {i: [] for i in range(10)}
    class_labels = {i: [] for i in range(10)}
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(class_images[label]) < num_per_class:
            class_images[label].append(img)
            class_labels[label].append(label)
        
        # Check if we have enough
        if all(len(v) >= num_per_class for v in class_images.values()):
            break
    
    images = []
    labels = []
    for c in range(10):
        images.extend(class_images[c])
        labels.extend(class_labels[c])
    
    return images, labels, class_names
