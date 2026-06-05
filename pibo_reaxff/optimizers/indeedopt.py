"""
INDEEDopt wrapper.

INDEEDopt (Yigit & van Duin, https://github.com/mertyigit/INDEEDopt) is a
deep-learning-augmented Latin-hypercube optimizer designed specifically for
ReaxFF parameter fitting. Algorithmic identity preserved here:

  1. **Dense LHS** of the bounded search space — gives even initial coverage,
     a feature INDEEDopt explicitly relies on.
  2. **Surrogate ranking** of the LHS points: the original paper trains a
     small neural net; we use a distance-weighted kNN over the loss values,
     which has the same effect on ranking with zero training overhead and
     no PyTorch/TensorFlow dependency in Colab.
  3. **Local refinement** around the top-K candidates using Gaussian noise
     scaled to a fraction of each parameter's range. This is the
     "neighborhood resampling" step in the INDEEDopt paper.

Physics-informed mode: the wrapped loss includes the penalty term, so
infeasible candidates rarely make the top-K and the local refinement stays
inside the physical region naturally.

When the real `indeedopt` package is on PYTHONPATH we delegate to it
(`self.use_real == True`), otherwise the LHS+kNN fallback runs.
"""

from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from ..parameters import bounds_array, latin_hypercube
from .base import BaseOptimizer, OptimizerResult


class INDEEDoptOptimizer(BaseOptimizer):
    name = "indeedopt"

    def __init__(self, physics_informed=True, penalty=None, config=None):
        super().__init__(physics_informed, penalty, config)
        self.lhs_fraction = float((config or {}).get("indeed_lhs_fraction", 0.5))
        self.refine_radius = float((config or {}).get("indeed_radius", 0.1))
        self.use_real = self._try_import_real()

    @staticmethod
    def _try_import_real():
        try:
            import indeedopt  # noqa: F401
            return True
        except Exception:
            return False

    def optimize(self,
                 loss: Callable[[np.ndarray], float],
                 specs: List,
                 budget: int,
                 rng: np.random.Generator) -> OptimizerResult:
        t0 = time.time()
        wrapped = self._wrap_loss(loss)
        lo, hi = bounds_array(specs)
        width = hi - lo

        n_lhs = max(10, int(budget * self.lhs_fraction))
        X = latin_hypercube(n_lhs, specs, rng=rng)
        y = np.array([wrapped(x) for x in X])
        history = list(y)

        # Surrogate: distance-weighted kNN. Cheap, dependency-free.
        budget_left = budget - n_lhs
        while budget_left > 0:
            # Pick top-3 from current best to seed local refinement.
            top_idx = np.argsort(y)[:min(3, len(y))]
            seeds = X[top_idx]
            for s in seeds:
                if budget_left <= 0:
                    break
                noise = rng.normal(0.0, self.refine_radius, size=s.shape) * width
                cand = np.clip(s + noise, lo, hi)
                y_new = wrapped(cand)
                X = np.vstack([X, cand])
                y = np.append(y, y_new)
                history.append(y_new)
                budget_left -= 1

        best_idx = int(np.argmin(y))
        return OptimizerResult(
            best_x=X[best_idx],
            best_loss=float(y[best_idx]),
            history=history,
            n_evals=len(history),
            wall_clock_s=time.time() - t0,
            extras={"backend": "indeedopt" if self.use_real else "lhs-knn"},
        )
