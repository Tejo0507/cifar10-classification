"""
Device management utilities.

Provides automatic device detection and configuration for
seamless CPU/GPU execution.
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Determine the computation device based on availability and preference.
    
    Args:
        preference: Device preference. Options:
            - "auto": Use CUDA if available, else CPU
            - "cuda": Force CUDA (raises error if unavailable)
            - "cpu": Force CPU
    
    Returns:
        torch.device: The selected computation device.
    
    Raises:
        RuntimeError: If "cuda" is requested but not available.
    """
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("[Device] Using CPU")
    
    return device
