import os
import torch
import numpy as np
import random

def set_deterministic_training(seed=42):
    """Ensure completely deterministic training including CUDA operations."""
    # Set CUDA deterministic environment variable
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For DataLoader reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    return torch.Generator().manual_seed(seed)