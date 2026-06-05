"""
Particle Swarm Optimization for ReaxFF parameters.

Reference: Furman, Wallace, et al., "Reactive Force Field for Liquid
Hydrazoic Acid with Applications to Detonation Chemistry", J. Chem. Theory
Comput. 2018, 14, 3, 1420-1430 — the PSO+local-search hybrid that paper uses
for ReaxFF parameter fitting.

Algorithmic specifics implemented here:
    * **Swarm initialization** via Latin-hypercube sampling so each particle
      starts in a different sub-region of the parameter space.
    * **Linearly-decreasing inertia** w(t) = w_max - (w_max - w_min) * t / T,
      with the conventional values w_max=0.9, w_min=0.4. High inertia early
      encourages exploration, low inertia late encourages exploitation —
      important on the rugged ReaxFF loss landscape.
    * **Reflective bound handling**: particles overshooting [lo, hi] are
      reflected back into the box. Cheap and preserves swarm diversity better
      than naive clipping.
    * **Cognitive (c1) and social (c2) weights** both = 1.5 (standard).
    * **Physics-informed mode** adds the quadratic penalty to the loss via
      `BaseOptimizer._wrap_loss`, so infeasible particles are penalized
      smoothly rather than rejected — this lets PSO still extract gradient-
      like information from the penalty surface.
"""

from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from ..parameters import bounds_array, latin_hypercube
from .base import BaseOptimizer, OptimizerResult


class PSOOptimizer(BaseOptimizer):
    name = "pso"

    def __init__(self, physics_informed=True, penalty=None, config=None):
        super().__init__(physics_informed, penalty, config)
        self.swarm_size = int((config or {}).get("pso_swarm", 20))
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5

    def optimize(self,
                 loss: Callable[[np.ndarray], float],
                 specs: List,
                 budget: int,
                 rng: np.random.Generator) -> OptimizerResult:
        t0 = time.time()
        wrapped = self._wrap_loss(loss)
        lo, hi = bounds_array(specs)
        d = len(specs)
        n = max(5, min(self.swarm_size, budget // 5))
        n_iter = max(1, (budget - n) // n)

        X = latin_hypercube(n, specs, rng=rng)
        V = rng.uniform(-1.0, 1.0, size=X.shape) * (hi - lo) * 0.1
        Y = np.array([wrapped(x) for x in X])
        history = list(Y)
        Pbest = X.copy()
        Pbest_y = Y.copy()
        g_idx = int(np.argmin(Y))
        Gbest = X[g_idx].copy()
        Gbest_y = float(Y[g_idx])

        for it in range(n_iter):
            w = self.w_max - (self.w_max - self.w_min) * (it / max(1, n_iter - 1))
            r1, r2 = rng.random(X.shape), rng.random(X.shape)
            V = (w * V
                 + self.c1 * r1 * (Pbest - X)
                 + self.c2 * r2 * (Gbest - X))
            X = X + V
            # Reflective bounds.
            X = np.where(X < lo, 2 * lo - X, X)
            X = np.where(X > hi, 2 * hi - X, X)
            X = np.clip(X, lo, hi)

            Y = np.array([wrapped(x) for x in X])
            history.extend(Y.tolist())
            improved = Y < Pbest_y
            Pbest[improved] = X[improved]
            Pbest_y[improved] = Y[improved]
            g_idx = int(np.argmin(Pbest_y))
            if Pbest_y[g_idx] < Gbest_y:
                Gbest_y = float(Pbest_y[g_idx])
                Gbest = Pbest[g_idx].copy()

        return OptimizerResult(
            best_x=Gbest,
            best_loss=float(Gbest_y),
            history=history,
            n_evals=len(history),
            wall_clock_s=time.time() - t0,
            extras={"swarm_size": n, "iterations": n_iter},
        )
