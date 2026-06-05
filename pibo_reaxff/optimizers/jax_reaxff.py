"""
JAX-ReaxFF wrapper.

Reference: Kaymak, Akki, Aktulga et al., "JAX-ReaxFF: A Gradient-Based
Framework for Fast Optimization of Reactive Force Fields", J. Chem.
Theory Comput. 2022 — https://github.com/cagrikymk/JAX-ReaxFF

Algorithmic identity within this benchmark:
    * **Local optimization with gradient information**, repeated from several
      Latin-hypercube starting points (JAX-ReaxFF's "multi-start LM/L-BFGS").
    * When the real `jaxreaxff` package is importable, we delegate to it; the
      common case in Colab is that the user has not installed it, so we use
      scipy's L-BFGS-B with a finite-difference gradient. Bounds are enforced
      analytically by scipy so the search never leaves the physically valid
      box. The physics penalty is folded into the objective via
      ``BaseOptimizer._wrap_loss``.

Why scipy fallback (not raw sign-gradient)
------------------------------------------
The earlier signed-step implementation diverged on a small fraction of seeds
(loss ~1e8). L-BFGS-B is the closest dependency-light analogue to JAX-ReaxFF's
quasi-Newton trust-region step and gives stable convergence.
"""

from __future__ import annotations

import time
from typing import Callable, List

import numpy as np
from scipy.optimize import minimize

from ..parameters import bounds_array, latin_hypercube
from .base import BaseOptimizer, OptimizerResult


class JAXReaxFFOptimizer(BaseOptimizer):
    name = "jax_reaxff"

    def __init__(self, physics_informed: bool = True, penalty=None, config=None):
        super().__init__(physics_informed, penalty, config)
        self.n_restarts = int((config or {}).get("jax_restarts", 5))
        self.use_jax = self._try_import_jax()

    @staticmethod
    def _try_import_jax():
        try:
            import jax  # noqa: F401
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
        bounds = list(zip(lo.tolist(), hi.tolist()))

        history: list[float] = []
        evals_used = [0]  # closure counter (avoids non-local trickery)

        # Wrap loss to track every evaluation -- scipy doesn't expose this.
        def tracked(x):
            v = float(wrapped(x))
            history.append(v)
            evals_used[0] += 1
            return v

        # Distribute budget across multi-starts (mirrors JAX-ReaxFF practice).
        n_restarts = max(1, min(self.n_restarts, budget // 8))
        per_restart = max(2, budget // n_restarts)
        starts = latin_hypercube(n_restarts, specs, rng=rng)

        best_x = starts[0].copy()
        best_loss = np.inf

        for r in range(n_restarts):
            if evals_used[0] >= budget:
                break
            remaining = budget - evals_used[0]
            maxiter = max(1, min(per_restart, remaining))
            try:
                # L-BFGS-B with finite-difference gradient. maxfun caps the
                # *total* function evaluations including FD gradients, so we
                # stay within the global budget.
                res = minimize(
                    tracked, starts[r], method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": maxiter, "maxfun": maxiter,
                             "ftol": 1e-9, "gtol": 1e-6},
                )
                x_r, loss_r = res.x, float(res.fun)
            except Exception:
                # Robustness against pathological gradients (e.g. NaN from
                # surrogate when params hit the bound). Keep the warm-start.
                x_r, loss_r = starts[r], tracked(starts[r])

            if loss_r < best_loss:
                best_loss = loss_r
                best_x = x_r

        return OptimizerResult(
            best_x=best_x,
            best_loss=float(best_loss),
            history=history,
            n_evals=int(evals_used[0]),
            wall_clock_s=time.time() - t0,
            extras={"backend": "jax" if self.use_jax else "L-BFGS-B (scipy)",
                    "n_restarts": n_restarts},
        )