"""
Seed Utilities for Reproducible Results
Ensures consistent random number generation across all modules
"""

import os
import random
import numpy as np

# Set default seed value
DEFAULT_SEED = 42


def set_reproducible_seeds(seed=DEFAULT_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed (int): Random seed value (default: 42)
    
    Note:
        This function sets seeds for:
        - Python's built-in random module
        - NumPy's random number generator
        - PyTorch (if available)
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    except ImportError:
        pass  # PyTorch not available
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_pytorch_determinism():
    """
    Configure PyTorch for deterministic operations.
    
    Note:
        This may impact performance but ensures reproducibility.
        Some operations may not have deterministic implementations.
    """
    try:
        import torch
        
        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic
        
        # Enable cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Disable CUDA convolution benchmarking
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            
    except ImportError:
        pass  # PyTorch not available
    except Exception as e:
        # Some older PyTorch versions may not support all settings
        import warnings
        warnings.warn(f"Could not configure full PyTorch determinism: {e}")


def get_torch_generator(seed=DEFAULT_SEED):
    """
    Get a PyTorch random generator with a fixed seed.
    Useful for DataLoader with shuffle=True.
    
    Args:
        seed (int): Random seed value
        
    Returns:
        torch.Generator or None if PyTorch not available
    """
    try:
        import torch
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator
    except ImportError:
        return None


def set_seed_all(seed=DEFAULT_SEED, deterministic=True):
    """
    Convenience function to set all seeds and configure determinism.
    
    Args:
        seed (int): Random seed value (default: 42)
        deterministic (bool): Whether to configure deterministic PyTorch (default: True)
    """
    set_reproducible_seeds(seed)
    if deterministic:
        configure_pytorch_determinism()
