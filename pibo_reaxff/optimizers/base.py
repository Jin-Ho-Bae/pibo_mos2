"""
Common optimizer interface used by the benchmark suite.

Every optimizer must implement:
    .optimize(loss_callable, specs, budget, rng) -> OptimizerResult

This lets the BenchmarkSuite swap implementations behind a single API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class OptimizerResult:
    """Uniform result schema for all four optimizers."""
    best_x: np.ndarray
    best_loss: float
    history: List[float] = field(default_factory=list)
    n_evals: int = 0
    wall_clock_s: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer:
    """ABC for optimizer wrappers."""

    name: str = "base"

    def __init__(self, physics_informed: bool = True,
                 penalty=None, config: Optional[Dict[str, Any]] = None):
        self.physics_informed = physics_informed
        self.penalty = penalty
        self.config = config or {}

    # Subclasses override ----------------------------------------------------

    def optimize(self,
                 loss: Callable[[np.ndarray], float],
                 specs: List,
                 budget: int,
                 rng: np.random.Generator) -> OptimizerResult:  # pragma: no cover
        raise NotImplementedError

    # Shared helpers --------------------------------------------------------

    def _wrap_loss(self, loss: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        """Augment the loss with the physics penalty when enabled."""
        if not self.physics_informed or self.penalty is None:
            return loss

        def wrapped(x: np.ndarray) -> float:
            return float(loss(x)) + float(self.penalty(x))
        wrapped.__name__ = f"{getattr(loss, '__name__', 'loss')}_phys"
        return wrapped
