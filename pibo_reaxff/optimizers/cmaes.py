"""
CMA-ES wrapper using Nikolaus Hansen's reference ``cma`` package.

Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial",
arXiv:1604.00772. CMA-ES is the canonical derivative-free
covariance-adapting baseline that Reviewer #2 specifically requested for
the ReaxFF parameter-calibration benchmark on matched compute budget.

Identity preserved here:
    * ``cma.CMAEvolutionStrategy`` with default lambda (4 + 3*ln(d)) and
      mu = lambda/2 — the textbook setting.
    * **Box constraints** enforced by ``BoundTransform`` so the search
      never leaves the physically valid parameter box (matches PIBO's
      and JAX-ReaxFF's bound semantics for fair comparison).
    * **Initial sigma** = 0.25 of the median half-width across all
      parameters (in the [lo, hi]-normalized coordinate); CMA-ES's
      step-size adaptation handles the rest. This matches the
      Hansen-recommended ``sigma0 = 0.25`` for normalized search domains.
    * **Physics-informed** mode adds the quadratic penalty via
      ``BaseOptimizer._wrap_loss`` — same hook as PSO/JAX-ReaxFF/INDEEDopt.
"""

from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from ..parameters import bounds_array, latin_hypercube
from .base import BaseOptimizer, OptimizerResult


class CMAESOptimizer(BaseOptimizer):
    name = "cmaes"

    def __init__(self, physics_informed: bool = True, penalty=None, config=None):
        super().__init__(physics_informed, penalty, config)
        cfg = config or {}
        self.sigma0 = float(cfg.get("cmaes_sigma0", 0.25))
        self.popsize_override = cfg.get("cmaes_popsize", None)

    def optimize(self,
                 loss: Callable[[np.ndarray], float],
                 specs: List,
                 budget: int,
                 rng: np.random.Generator) -> OptimizerResult:
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "CMAESOptimizer requires the 'cma' package: pip install cma"
            ) from exc

        t0 = time.time()
        wrapped = self._wrap_loss(loss)
        lo, hi = bounds_array(specs)
        d = len(specs)

        # Work in normalized [0, 1] coordinates so sigma0 is dimensionless.
        def to_phys(z):
            return lo + z * (hi - lo)

        def f(z):
            x = to_phys(np.asarray(z, dtype=float))
            x = np.clip(x, lo, hi)
            return float(wrapped(x))

        # Seed from the user's rng so replicate variance is reproducible.
        seed = int(rng.integers(1, 2**31 - 1))
        opts = {
            "bounds": [[0.0] * d, [1.0] * d],
            "seed": seed,
            "verbose": -9,
            "maxfevals": budget,
            "tolx": 1e-9,
            "tolfun": 1e-9,
            "CMA_active": True,
        }
        if self.popsize_override is not None:
            opts["popsize"] = int(self.popsize_override)

        # LHS warm-start: evaluate one LHS center to pick the best starting
        # point, mirroring how PIBO/JAX-ReaxFF seed multi-start search.
        # This is a single eval, so it stays well within the budget.
        x0_phys = latin_hypercube(1, specs, rng=rng)[0]
        z0 = (x0_phys - lo) / np.maximum(hi - lo, 1e-12)
        z0 = np.clip(z0, 0.0, 1.0)

        history: list[float] = []
        evals_used = 0

        es = cma.CMAEvolutionStrategy(z0.tolist(), self.sigma0, opts)
        best_z = z0.copy()
        best_y = np.inf

        while not es.stop() and evals_used < budget:
            solutions = es.ask()
            # Cap to remaining budget so we never overshoot.
            if evals_used + len(solutions) > budget:
                solutions = solutions[: max(1, budget - evals_used)]
            ys = [f(s) for s in solutions]
            evals_used += len(ys)
            history.extend(ys)
            # Update CMA-ES with the actually-evaluated subset.
            try:
                es.tell(solutions, ys)
            except Exception:
                break  # malformed update (e.g. truncated last population)
            min_idx = int(np.argmin(ys))
            if ys[min_idx] < best_y:
                best_y = float(ys[min_idx])
                best_z = np.asarray(solutions[min_idx], dtype=float)

        best_x = np.clip(to_phys(best_z), lo, hi)
        return OptimizerResult(
            best_x=best_x,
            best_loss=float(best_y),
            history=history,
            n_evals=int(evals_used),
            wall_clock_s=time.time() - t0,
            extras={"backend": "cma", "sigma0": self.sigma0,
                    "popsize": es.popsize, "seed": seed},
        )
