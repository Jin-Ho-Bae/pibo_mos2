"""Optimizer modules exposed via a uniform interface (see optimizers.base)."""

from .base import BaseOptimizer, OptimizerResult
from .pibo import PIBOOptimizer
from .jax_reaxff import JAXReaxFFOptimizer
from .indeedopt import INDEEDoptOptimizer
from .pso import PSOOptimizer
from .cmaes import CMAESOptimizer

OPTIMIZER_REGISTRY = {
    "pibo": PIBOOptimizer,
    "cmaes": CMAESOptimizer,
    "jax_reaxff": JAXReaxFFOptimizer,
    "indeedopt": INDEEDoptOptimizer,
    "pso": PSOOptimizer,
}


def get_optimizer(name):
    """Factory: return an *instantiated* optimizer class by short name."""
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(
            f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZER_REGISTRY)}."
        )
    return OPTIMIZER_REGISTRY[name]


__all__ = [
    "BaseOptimizer",
    "OptimizerResult",
    "PIBOOptimizer",
    "CMAESOptimizer",
    "JAXReaxFFOptimizer",
    "INDEEDoptOptimizer",
    "PSOOptimizer",
    "OPTIMIZER_REGISTRY",
    "get_optimizer",
]
