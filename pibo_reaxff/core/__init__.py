"""
Core modules for Physics-Informed Bayesian Optimization
"""

from .blocked_pibo import BlockedPIBO
from .improved_gp import ImprovedGaussianProcess, DeepKernelGP
from .staged_pibo import PhysicsInformedBO

__all__ = [
    'BlockedPIBO',
    'ImprovedGaussianProcess',
    'DeepKernelGP', 
    'PhysicsInformedBO'
]

# Module metadata
__version__ = "2.0.0"
__description__ = "Core PIBO optimization modules with blocked search and staged strategies"
