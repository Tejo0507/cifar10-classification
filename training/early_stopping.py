"""
Early Stopping callback for training.

Monitors a validation metric and stops training when the metric
has not improved for a specified number of epochs (patience).
This prevents overfitting and saves compute time.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Tracks the best validation loss and counts epochs without improvement.
    When the patience threshold is exceeded, signals training to stop.
    Optionally saves a checkpoint of the best model.
    
    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        verbose: If True, print messages on improvement or stopping.
        mode: "min" to minimize metric (loss), "max" to maximize (accuracy).
    
    Example:
        >>> early_stop = EarlyStopping(patience=10, min_delta=1e-4)
        >>> for epoch in range(max_epochs):
        ...     val_loss = train_one_epoch(...)
        ...     early_stop(val_loss)
        ...     if early_stop.early_stop:
        ...         print("Stopping early!")
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        elif mode == "max":
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = -np.inf
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
    
    def __call__(self, metric_value: float, epoch: int = 0) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            metric_value: Current epoch's validation metric
            epoch: Current epoch number (for logging)
        
        Returns:
            True if metric improved (model should be saved), False otherwise
        """
        if self.is_better(metric_value, self.best_score):
            # Improvement detected
            if self.verbose:
                improvement = abs(metric_value - self.best_score)
                print(f"  [EarlyStopping] Improved by {improvement:.6f} "
                      f"({self.best_score:.6f} -> {metric_value:.6f})")
            
            self.best_score = metric_value
            self.counter = 0
            self.best_epoch = epoch
            return True  # Signal that model should be saved
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                      f"(best: {self.best_score:.6f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] *** Stopping training at epoch {epoch} ***")
                    print(f"  [EarlyStopping] Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            return False
    
    def reset(self):
        """Reset early stopping state (useful for multi-phase training)."""
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
