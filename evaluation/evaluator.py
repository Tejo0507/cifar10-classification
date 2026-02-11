"""
Comprehensive evaluation pipeline for CIFAR-10 classification.

Provides:
    - Accuracy, Precision, Recall, F1-score (macro + weighted)
    - Per-class accuracy breakdown
    - Confusion matrix computation
    - Misclassified image collection
    - Error analysis and most-confused class pairs
    - Full evaluation reports
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from utils.visualization import (
    plot_confusion_matrix,
    plot_misclassified,
)


class Evaluator:
    """
    Comprehensive model evaluation on test/validation sets.
    
    Computes classification metrics, generates confusion matrices,
    collects misclassified samples, and performs error analysis.
    
    Args:
        model: Trained PyTorch model
        device: Computation device
        class_names: Tuple of class name strings
        output_dir: Directory for saving evaluation outputs
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: tuple,
        output_dir: str = "./outputs/results",
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Will be populated after evaluate()
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.misclassified_images = []
        self.misclassified_true = []
        self.misclassified_pred = []
    
    # ==========================================================================
    # Core Evaluation
    # ==========================================================================
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        collect_misclassified: bool = True,
        max_misclassified: int = 100,
    ) -> Dict:
        """
        Run full evaluation on a data loader.
        
        Performs a complete forward pass over the dataset, collecting
        predictions, computing metrics, and optionally gathering
        misclassified samples for visualization.
        
        Args:
            data_loader: DataLoader for evaluation (test or validation)
            collect_misclassified: Whether to store misclassified images
            max_misclassified: Maximum misclassified images to collect
        
        Returns:
            Dictionary with all computed metrics
        """
        self.model.eval()
        
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.misclassified_images = []
        self.misclassified_true = []
        self.misclassified_pred = []
        
        running_loss = 0.0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)
            
            self.all_preds.extend(predicted.cpu().numpy())
            self.all_labels.extend(targets.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())
            
            # Collect misclassified samples
            if collect_misclassified:
                mask = predicted != targets
                if mask.any() and len(self.misclassified_images) < max_misclassified:
                    wrong_idx = mask.nonzero(as_tuple=True)[0]
                    for idx in wrong_idx:
                        if len(self.misclassified_images) >= max_misclassified:
                            break
                        self.misclassified_images.append(inputs[idx].cpu())
                        self.misclassified_true.append(targets[idx].item())
                        self.misclassified_pred.append(predicted[idx].item())
        
        # Convert to numpy arrays
        self.all_preds = np.array(self.all_preds)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)
        
        # Compute all metrics
        metrics = self._compute_metrics(running_loss / total)
        
        return metrics
    
    # ==========================================================================
    # Metrics Computation
    # ==========================================================================
    
    def _compute_metrics(self, avg_loss: float) -> Dict:
        """
        Compute comprehensive classification metrics.
        
        Args:
            avg_loss: Average cross-entropy loss
        
        Returns:
            Dictionary with all metrics
        """
        y_true = self.all_labels
        y_pred = self.all_preds
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        
        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
        
        # Per-class accuracy
        per_class_acc = self._per_class_accuracy(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Most confused pairs
        confused_pairs = self._most_confused_pairs(cm, top_k=5)
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm.tolist(),
            "most_confused_pairs": confused_pairs,
            "num_misclassified": len(self.misclassified_images),
            "total_samples": len(y_true),
        }
        
        return metrics
    
    def _per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute accuracy for each class individually.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary mapping class names to accuracy percentages
        """
        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).sum() / mask.sum() * 100
                per_class[name] = round(class_acc, 2)
            else:
                per_class[name] = 0.0
        
        return per_class
    
    def _most_confused_pairs(
        self,
        cm: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find the most frequently confused class pairs.
        
        Identifies which classes the model most often mistakes for one another,
        useful for error analysis.
        
        Args:
            cm: Confusion matrix
            top_k: Number of top confused pairs to return
        
        Returns:
            List of dicts with true_class, predicted_class, count
        """
        # Zero out diagonal (correct predictions)
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        
        # Find top-k off-diagonal entries
        pairs = []
        flat_indices = np.argsort(cm_copy.ravel())[::-1][:top_k]
        
        for flat_idx in flat_indices:
            true_idx = flat_idx // cm.shape[1]
            pred_idx = flat_idx % cm.shape[1]
            count = cm_copy[true_idx, pred_idx]
            if count > 0:
                pairs.append({
                    "true_class": self.class_names[true_idx],
                    "predicted_as": self.class_names[pred_idx],
                    "count": int(count),
                    "error_rate": round(count / cm[true_idx].sum() * 100, 1),
                })
        
        return pairs
    
    # ==========================================================================
    # Reporting
    # ==========================================================================
    
    def print_report(self, metrics: Dict, experiment_name: str = "") -> None:
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Metrics dictionary from evaluate()
            experiment_name: Name for the report header
        """
        print(f"\n{'='*70}")
        print(f"  EVALUATION REPORT: {experiment_name}")
        print(f"{'='*70}")
        
        print(f"\n  Overall Metrics:")
        print(f"  {'Accuracy:':<25} {metrics['accuracy']:.2f}%")
        print(f"  {'Loss:':<25} {metrics['loss']:.4f}")
        print(f"  {'Precision (macro):':<25} {metrics['precision_macro']:.2f}%")
        print(f"  {'Recall (macro):':<25} {metrics['recall_macro']:.2f}%")
        print(f"  {'F1-Score (macro):':<25} {metrics['f1_macro']:.2f}%")
        print(f"  {'Precision (weighted):':<25} {metrics['precision_weighted']:.2f}%")
        print(f"  {'Recall (weighted):':<25} {metrics['recall_weighted']:.2f}%")
        print(f"  {'F1-Score (weighted):':<25} {metrics['f1_weighted']:.2f}%")
        
        print(f"\n  Per-Class Accuracy:")
        for name, acc in metrics["per_class_accuracy"].items():
            bar = "#" * int(acc / 2) + "." * (50 - int(acc / 2))
            print(f"  {name:<15} {acc:>6.2f}% |{bar}|")
        
        if metrics["most_confused_pairs"]:
            print(f"\n  Most Confused Pairs:")
            for pair in metrics["most_confused_pairs"]:
                print(f"  {pair['true_class']:>12} -> {pair['predicted_as']:<12} "
                      f"({pair['count']} errors, {pair['error_rate']}% of class)")
        
        print(f"\n{'='*70}")
    
    def generate_classification_report(self) -> str:
        """Generate sklearn's detailed classification report."""
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=list(self.class_names),
            digits=4,
        )
        return report
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    def plot_results(
        self,
        metrics: Dict,
        experiment_name: str = "",
        mean: tuple = (0.4914, 0.4822, 0.4465),
        std: tuple = (0.2470, 0.2435, 0.2616),
    ) -> None:
        """
        Generate all evaluation visualizations.
        
        Creates and saves:
            - Confusion matrix heatmap
            - Misclassified image grid
        
        Args:
            metrics: Metrics from evaluate()
            experiment_name: Identifier for file naming
            mean: Dataset mean for denormalization
            std: Dataset std for denormalization
        """
        # Confusion matrix
        cm_path = os.path.join(self.output_dir, f"{experiment_name}_confusion_matrix.png")
        plot_confusion_matrix(
            self.all_labels, self.all_preds,
            list(self.class_names),
            experiment_name=experiment_name,
            normalize=True,
            save_path=cm_path,
        )
        
        # Misclassified images
        if self.misclassified_images:
            mis_path = os.path.join(
                self.output_dir, f"{experiment_name}_misclassified.png"
            )
            plot_misclassified(
                self.misclassified_images,
                self.misclassified_true,
                self.misclassified_pred,
                list(self.class_names),
                num_images=25,
                mean=mean,
                std=std,
                save_path=mis_path,
            )
    
    def save_metrics(self, metrics: Dict, experiment_name: str = "") -> str:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            experiment_name: Identifier for file naming
        
        Returns:
            Path to saved file
        """
        # Convert numpy types for JSON serialization
        serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, np.floating):
                serializable[key] = float(value)
            elif isinstance(value, np.integer):
                serializable[key] = int(value)
            else:
                serializable[key] = value
        
        path = os.path.join(self.output_dir, f"{experiment_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)
        
        print(f"[Evaluator] Metrics saved to {path}")
        return path


# ==============================================================================
# Error Analysis
# ==============================================================================

def generate_error_analysis(
    experiment_results: List[Dict],
    output_path: str = "./outputs/results/error_analysis.txt",
) -> str:
    """
    Generate a research-style error analysis report.
    
    Analyzes patterns across multiple experiments to identify:
        - Overfitting vs underfitting behavior
        - Impact of data augmentation
        - Transfer learning benefits
        - Most challenging classes
    
    Args:
        experiment_results: List of experiment result dicts, each containing:
            - name, model, augmentation, train_acc, val_acc, test_acc,
              per_class_accuracy, most_confused_pairs
        output_path: Path to save the analysis report
    
    Returns:
        The analysis report as a string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("  ERROR ANALYSIS AND EXPERIMENTAL OBSERVATIONS")
    report_lines.append("=" * 80)
    
    # --- Section 1: Overfitting Analysis ---
    report_lines.append("\n1. OVERFITTING AND UNDERFITTING ANALYSIS")
    report_lines.append("-" * 50)
    
    for result in experiment_results:
        name = result.get("name", "Unknown")
        train_acc = result.get("train_acc", 0)
        val_acc = result.get("val_acc", 0)
        test_acc = result.get("test_acc", 0)
        gap = train_acc - val_acc
        
        if gap > 15:
            diagnosis = "SEVERE OVERFITTING"
        elif gap > 8:
            diagnosis = "MODERATE OVERFITTING"
        elif gap > 3:
            diagnosis = "MILD OVERFITTING"
        elif val_acc < 60:
            diagnosis = "UNDERFITTING"
        else:
            diagnosis = "WELL-FITTED"
        
        report_lines.append(
            f"  {name}:\n"
            f"    Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%\n"
            f"    Train-Val Gap: {gap:.2f}% -- {diagnosis}"
        )
    
    # --- Section 2: Data Augmentation Impact ---
    report_lines.append("\n2. IMPACT OF DATA AUGMENTATION")
    report_lines.append("-" * 50)
    
    baseline_results = [r for r in experiment_results if r.get("augmentation") == "baseline"]
    advanced_results = [r for r in experiment_results if r.get("augmentation") == "advanced"]
    
    if baseline_results and advanced_results:
        for base in baseline_results:
            model_type = base.get("model", "")
            advanced_match = [a for a in advanced_results if a.get("model") == model_type]
            if advanced_match:
                adv = advanced_match[0]
                improvement = adv.get("test_acc", 0) - base.get("test_acc", 0)
                report_lines.append(
                    f"  {model_type}:\n"
                    f"    Baseline: {base.get('test_acc', 0):.2f}% -> "
                    f"Advanced: {adv.get('test_acc', 0):.2f}% "
                    f"(Δ = {improvement:+.2f}%)"
                )
    else:
        report_lines.append("  Insufficient data for augmentation comparison.")
    
    report_lines.append(
        "\n  Observation: Data augmentation typically provides 2-5% improvement\n"
        "  by reducing overfitting through artificial sample diversity. Advanced\n"
        "  augmentations (RandomCrop, ColorJitter, RandomErasing) simulate real-world\n"
        "  variations the model may encounter during inference."
    )
    
    # --- Section 3: Transfer Learning Impact ---
    report_lines.append("\n3. TRANSFER LEARNING IMPACT")
    report_lines.append("-" * 50)
    
    cnn_results = [r for r in experiment_results
                   if r.get("model") in ("baseline_cnn", "improved_cnn")]
    resnet_results = [r for r in experiment_results if r.get("model") == "resnet18"]
    
    if cnn_results and resnet_results:
        best_cnn = max(cnn_results, key=lambda x: x.get("test_acc", 0))
        best_resnet = max(resnet_results, key=lambda x: x.get("test_acc", 0))
        gain = best_resnet.get("test_acc", 0) - best_cnn.get("test_acc", 0)
        
        report_lines.append(
            f"  Best custom CNN ({best_cnn['name']}): {best_cnn.get('test_acc', 0):.2f}%\n"
            f"  Best ResNet18 ({best_resnet['name']}): {best_resnet.get('test_acc', 0):.2f}%\n"
            f"  Transfer learning advantage: {gain:+.2f}%"
        )
    
    report_lines.append(
        "\n  Observation: Transfer learning with ImageNet pretrained weights provides\n"
        "  a significant accuracy boost (typically 5-15%) due to:\n"
        "    a) Rich, general-purpose feature representations from ImageNet\n"
        "    b) Better initialization that avoids poor local minima\n"
        "    c) Features learned from 1.2M images vs 42K CIFAR-10 training images\n"
        "    d) Hierarchical features (edges -> textures -> parts) that transfer well"
    )
    
    # --- Section 4: Most Challenging Classes ---
    report_lines.append("\n4. MOST CHALLENGING CLASS PAIRS")
    report_lines.append("-" * 50)
    report_lines.append(
        "  The following class pairs are consistently confused across models,\n"
        "  indicating visual similarity that challenges learned representations:\n"
    )
    
    all_confused = {}
    for result in experiment_results:
        for pair in result.get("most_confused_pairs", []):
            key = (pair["true_class"], pair["predicted_as"])
            all_confused[key] = all_confused.get(key, 0) + pair["count"]
    
    sorted_confused = sorted(all_confused.items(), key=lambda x: x[1], reverse=True)[:8]
    for (true_cls, pred_cls), count in sorted_confused:
        report_lines.append(f"  {true_cls:>12} <-> {pred_cls:<12}  (total errors: {count})")
    
    report_lines.append(
        "\n  Common confusion patterns:\n"
        "    - cat <-> dog: Similar body shapes and textures\n"
        "    - automobile <-> truck: Shared vehicle features\n"
        "    - deer <-> horse: Similar quadruped body structure\n"
        "    - bird <-> airplane: Shared flight-related visual cues\n"
        "  These confusions reflect genuine visual ambiguity in 32x32 images."
    )
    
    # --- Section 5: Model Architecture Impact ---
    report_lines.append("\n5. MODEL ARCHITECTURE ANALYSIS")
    report_lines.append("-" * 50)
    report_lines.append(
        "  Baseline CNN (3 conv layers, ~190K params):\n"
        "    - Limited receptive field restricts global feature learning\n"
        "    - Adequate for simple textures but struggles with complex objects\n"
        "    - Expected range: 65-75% accuracy\n"
        "\n"
        "  Improved CNN (6 conv layers + GAP, ~850K params):\n"
        "    - Deeper architecture captures richer feature hierarchies\n"
        "    - Global Average Pooling reduces overfitting vs FC layers\n"
        "    - Double-conv blocks enhance feature expressiveness\n"
        "    - Expected range: 75-85% accuracy\n"
        "\n"
        "  ResNet18 (pretrained, ~11.2M params):\n"
        "    - Residual connections enable very deep effective feature learning\n"
        "    - ImageNet pretraining provides powerful general features\n"
        "    - Skip connections solve vanishing gradient problem\n"
        "    - Expected range: 85-93% accuracy"
    )
    
    # --- Section 6: Learning Rate Impact ---
    report_lines.append("\n6. LEARNING RATE ANALYSIS")
    report_lines.append("-" * 50)
    
    for result in experiment_results:
        lr = result.get("lr", "N/A")
        report_lines.append(
            f"  {result.get('name', 'Unknown')}: LR={lr}, "
            f"Test Acc={result.get('test_acc', 0):.2f}%"
        )
    
    report_lines.append(
        "\n  Observation: Learning rate is one of the most critical hyperparameters.\n"
        "  Too high: Unstable training, loss oscillation\n"
        "  Too low: Slow convergence, may get stuck in suboptimal minima\n"
        "  Cosine annealing provides smooth decay, often outperforming step decay."
    )
    
    report_lines.append(f"\n{'='*80}")
    
    report = "\n".join(report_lines)
    
    # Save to file
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[Analysis] Error analysis saved to {output_path}")
    
    return report
