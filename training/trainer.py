"""
Reusable training pipeline for CIFAR-10 classification.

Provides a complete training loop with:
    - Configurable optimizer (Adam, SGD, AdamW)
    - Learning rate scheduling (Cosine, Step, ReduceOnPlateau)
    - Early stopping with patience
    - Model checkpointing (best and last)
    - Per-epoch metric tracking and logging
    - GPU/CPU device management
    - Training history for plotting
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np

from training.early_stopping import EarlyStopping


class Trainer:
    """
    Configurable training pipeline for image classification.
    
    Encapsulates the entire training workflow: optimizer setup,
    scheduling, training/validation loops, checkpointing, and
    metric tracking.
    
    Args:
        model: PyTorch model to train
        device: Computation device (torch.device)
        config: ExperimentConfig with all hyperparameters
    
    Example:
        >>> trainer = Trainer(model, device, config)
        >>> history = trainer.fit(train_loader, val_loader)
        >>> trainer.save_history("outputs/history.json")
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                verbose=True,
                mode="min",  # Monitor validation loss
            )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
            "epoch_times": [],
        }
        
        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Checkpointing
        self.checkpoint_dir = config.training.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    # ==========================================================================
    # Optimizer & Scheduler Construction
    # ==========================================================================
    
    def _build_optimizer(self) -> optim.Optimizer:
        """
        Build optimizer based on configuration.
        
        Supports differential learning rates for transfer learning models
        (lower LR for pretrained backbone, higher for classifier head).
        """
        cfg = self.config.training
        
        # Check if model supports parameter groups (transfer learning)
        if hasattr(self.model, "get_parameter_groups") and not self.model.freeze_backbone:
            param_groups = self.model.get_parameter_groups(
                backbone_lr=cfg.learning_rate * 0.1,  # 10x lower for backbone
                head_lr=cfg.learning_rate,
            )
            print(f"[Trainer] Using differential learning rates: "
                  f"backbone={cfg.learning_rate * 0.1:.6f}, head={cfg.learning_rate:.6f}")
        else:
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if cfg.optimizer == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
        
        print(f"[Trainer] Optimizer: {cfg.optimizer.upper()}, LR: {cfg.learning_rate}, "
              f"Weight Decay: {cfg.weight_decay}")
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler based on configuration."""
        cfg = self.config.training
        
        if cfg.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.scheduler_min_lr,
            )
        elif cfg.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.scheduler_step_size,
                gamma=cfg.scheduler_gamma,
            )
        elif cfg.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=cfg.scheduler_gamma,
                patience=cfg.scheduler_patience,
                min_lr=cfg.scheduler_min_lr,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
        
        print(f"[Trainer] Scheduler: {cfg.scheduler}")
        return scheduler
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Execute the full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history dictionary with loss/accuracy per epoch
        """
        print(f"\n{'='*70}")
        print(f"  Starting Training: {self.config.experiment_name}")
        print(f"  Model: {self.config.model.name} | Epochs: {self.config.training.epochs}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
        
        total_start_time = time.time()
        
        for epoch in range(1, self.config.training.epochs + 1):
            epoch_start_time = time.time()
            
            # ---- Train one epoch ----
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            
            # ---- Validate ----
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # ---- Record history ----
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)
            
            # ---- Logging ----
            print(
                f"Epoch [{epoch:3d}/{self.config.training.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # ---- Best model tracking & checkpointing ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            
            # ---- Learning rate scheduling ----
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # ---- Early stopping check ----
            if self.early_stopping is not None:
                improved = self.early_stopping(val_loss, epoch)
                if self.early_stopping.early_stop:
                    print(f"\n[Trainer] Early stopping triggered at epoch {epoch}")
                    break
        
        # ---- Training complete ----
        total_time = time.time() - total_start_time
        print(f"\n{'='*70}")
        print(f"  Training Complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best Epoch: {self.best_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*70}\n")
        
        # Save last checkpoint
        self._save_checkpoint(epoch, is_best=False)
        
        return self.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Periodic logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                batch_acc = 100.0 * correct / total
                print(
                    f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%",
                    end="\r"
                )
        
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on the validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    # ==========================================================================
    # Checkpointing
    # ==========================================================================
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Saves both model state and training state for resumable training.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "config": {
                "model_name": self.config.model.name,
                "experiment_name": self.config.experiment_name,
            },
        }
        
        if is_best:
            path = os.path.join(
                self.checkpoint_dir,
                f"{self.config.experiment_name}_best.pth"
            )
            torch.save(checkpoint, path)
            print(f"  [Checkpoint] Best model saved -> {path}")
        else:
            path = os.path.join(
                self.checkpoint_dir,
                f"{self.config.experiment_name}_last.pth"
            )
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint for resuming training.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        
        epoch = checkpoint["epoch"]
        print(f"[Trainer] Resumed from epoch {epoch} (best val acc: {self.best_val_acc:.2f}%)")
        return epoch
    
    # ==========================================================================
    # History Management
    # ==========================================================================
    
    def save_history(self, path: str) -> None:
        """Save training history to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)
        print(f"[Trainer] History saved to {path}")
    
    def get_best_results(self) -> Dict:
        """Return summary of best results."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "total_epochs": len(self.history["train_loss"]),
        }
