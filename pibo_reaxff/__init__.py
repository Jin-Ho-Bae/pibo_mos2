"""
Physics-Informed Bayesian Optimization for ReaxFF
Main package initialization
"""

__version__ = "2.0.0"
__author__ = "PIBO Development Team"

# Core modules
from .core.blocked_pibo import BlockedPIBO
from .core.improved_gp import ImprovedGaussianProcess, DeepKernelGP
from .core.staged_pibo import PhysicsInformedBO

# Visualization
from .visualization_pibo import PIBOVisualizer, visualize_results

__all__ = [
    'BlockedPIBO',
    'ImprovedGaussianProcess', 
    'DeepKernelGP',
    'PhysicsInformedBO',
    'PIBOVisualizer',
    'visualize_results'
]
