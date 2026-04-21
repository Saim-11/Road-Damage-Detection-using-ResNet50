"""
Computer Vision Final Project - Road Damage Detection and Classification
Module Package
"""

__version__ = "1.0.0"
__author__ = "Mudassar Nawaz, Muhammad Saim Zahid"
__description__ = "Road Damage Detection and Classification using Computer Vision"

# Import key utilities for easier access
try:
    from .seed_utils import set_seed_all, set_reproducible_seeds, configure_pytorch_determinism
except ImportError:
    pass  # Optional imports

print("Project modules loaded successfully")
