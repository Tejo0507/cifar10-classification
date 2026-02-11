"""
Reproducibility utilities.

Sets global random seeds across all libraries to ensure
deterministic behavior for scientific reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Sets seeds for:
        - Python's built-in random module
        - NumPy's random generator
        - PyTorch CPU and CUDA random generators
        - cuDNN deterministic mode
    
    Args:
        seed: Integer seed value. Default is 42.
    
    Note:
        Setting cuDNN to deterministic mode may slightly reduce performance
        but ensures reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All random seeds set to {seed}")
