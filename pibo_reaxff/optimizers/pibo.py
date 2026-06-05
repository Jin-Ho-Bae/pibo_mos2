"""
PIBO: Physics-Informed Bayesian Optimization (Manuscript-faithful).

Reference: Bae et al. (this work) — physics-informed BO for ReaxFF
calibration with staged acquisition, patience-based early stopping, and
online dataset management.

Algorithmic ingredients
-----------------------

1. **ARD-Matern Sparse GP surrogate** (``gp_surrogate.SparseGP``):
   - Matern-2.5 kernel: less brittle than RBF on rugged ReaxFF surfaces.
   - ARD length-scales: GP automatically down-weights irrelevant params.
   - Log-normal hyperprior + MAP fit: data-efficient on <30 DFT frames.
   - FITC-lite sparse approximation activates beyond ``sparse_threshold``.

2. **Stage-conditioned acquisition** (Manuscript §"BO with adaptive GPs"):
   - 0–30% of budget: **UCB** with decaying β — broad exploration.
   - 30–70% of budget: **EI** with ξ = 0.01 — directed improvement.
   - >70% of budget: **PI** — local exploitation around incumbent.
   - Random **Thompson sampling** injection with probability 0.15 to
     escape local optima (Manuscript §"BO with adaptive GPs").

3. **Constrained acquisition** (``physics_constraints.ConstrainedAcquisition``):
   multiplies the chosen acquisition score by a smooth feasibility term so
   the optimizer rarely *proposes* infeasible parameters.

4. **Patience-based early stopping** (Manuscript §"Convergence monitoring"):
   stop after ``patience`` consecutive evaluations without improvement,
   provided we are past the ``burn_in`` iteration.

5. **Online dataset management** (Manuscript §"Candidate generation"):
   keep at most ``online_window_initial`` early anchors plus the most-recent
   ``online_window_recent`` evaluations to bound GP fit cost as iterations
   accumulate.

6. **Latin-hypercube warm-start** (``init_lhs_points`` points) so the
   initial GP fit sees a well-spread sample.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, List, Optional

import numpy as np

from ..gp_surrogate import SparseGP
from ..parameters import bounds_array, latin_hypercube, normalize, denormalize
from ..physics_constraints import ConstrainedAcquisition
from .base import BaseOptimizer, OptimizerResult


class PIBOOptimizer(BaseOptimizer):
    name = "pibo"

    def __init__(self, physics_informed: bool = True, penalty=None,
                 config=None):
        super().__init__(physics_informed=physics_informed,
                         penalty=penalty, config=config)
        cfg = config or {}
        gpc = cfg.get("gp", {})
        self.gp = SparseGP(
            nu=gpc.get("nu", 2.5),
            ard=gpc.get("ard", True),
            sparse_threshold=gpc.get("sparse_threshold", 100),
            n_inducing=gpc.get("n_inducing", 50),
        )
        self.n_init = int(cfg.get("init_lhs_points", 20))
        self.n_acq_candidates = int(cfg.get("n_acq_candidates", 1024))

        # Stage-conditioned acquisition (Manuscript §"BO with adaptive GPs").
        acq = cfg.get("acquisition", {})
        self.stage_ucb_until = float(acq.get("stage_ucb_until", 0.30))
        self.stage_ei_until  = float(acq.get("stage_ei_until",  0.70))
        self.ucb_beta_init   = float(acq.get("ucb_beta_init",   2.0))
        self.ucb_beta_decay  = float(acq.get("ucb_beta_decay",  0.995))
        self.ei_xi           = float(acq.get("ei_xi",           0.01))
        self.thompson_prob   = float(acq.get("thompson_prob",   0.15))

        # Patience-based early stopping (Manuscript §"Convergence monitoring").
        self.patience = int(cfg.get("patience", 0))   # 0 disables
        self.burn_in  = int(cfg.get("burn_in",  0))

        # Online dataset trimming (Manuscript §"Candidate generation").
        self.window_recent  = int(cfg.get("online_window_recent",  0))
        self.window_initial = int(cfg.get("online_window_initial", 0))

        # Physical-validation gate (PIBO-only; other optimizers skip this).
        # validator must be a PhysicalValidator (or None to disable).
        # validate_every_k = 0 also disables the gate.
        self.validator         = cfg.get("validator", None)
        self.validate_every_k  = int(cfg.get("validate_every_k", 0))
        self.val_boost         = float(cfg.get("val_boost",   2.0))
        self.val_shrink        = float(cfg.get("val_shrink",  0.7))
        self.val_early_stop    = bool(cfg.get("val_early_stop", False))
        self.validation_log: List[dict] = []  # populated during optimize()

    # ----- staged acquisition selector ----------------------------------------

    def _stage(self, it: int, total: int) -> str:
        """Return ``"ucb" | "ei" | "pi"`` based on iteration progress."""
        frac = (it + 1) / max(1, total)
        if frac <= self.stage_ucb_until:
            return "ucb"
        if frac <= self.stage_ei_until:
            return "ei"
        return "pi"

    def _score_candidates(self, cand_u: np.ndarray, y_best: float,
                          stage: str, beta: float,
                          rng: np.random.Generator) -> np.ndarray:
        """Return a "higher is better" acquisition score per candidate."""
        # Thompson injection (Manuscript: 0.15 probability per iteration).
        if rng.random() < self.thompson_prob:
            return self.gp.thompson_sample(cand_u, rng=rng)
        if stage == "ucb":
            return self.gp.upper_confidence_bound(cand_u, beta=beta)
        if stage == "pi":
            return self.gp.probability_of_improvement(cand_u, y_best=y_best)
        return self.gp.expected_improvement(cand_u, y_best=y_best,
                                            xi=self.ei_xi)

    # ----- online dataset management ------------------------------------------

    def _trim_window(self, X: np.ndarray, y: np.ndarray, n_init: int
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Keep first-N initial anchors + most-recent-K evaluations.

        Disabled when both windows are 0 (default for the demo profile).
        """
        if self.window_recent <= 0 and self.window_initial <= 0:
            return X, y
        keep_init = min(self.window_initial or n_init, len(X))
        keep_recent = self.window_recent or 0
        if len(X) <= keep_init + keep_recent:
            return X, y
        head = np.arange(keep_init)
        tail = np.arange(len(X) - keep_recent, len(X))
        idx = np.concatenate([head, tail])
        return X[idx], y[idx]

    # ----- physical-validation gate (PIBO-only) ------------------------------

    def _params_dict(self, specs, x):
        return {s.name: float(v) for s, v in zip(specs, x)}

    def _apply_loss_weight_multipliers(self, loss, mults):
        """Mutate ``loss.weights.category_weights`` in place by the given
        multipliers. ``LossWeights`` is a mutable dataclass — safe to edit."""
        if not mults or not hasattr(loss, "weights"):
            return
        cw = getattr(loss.weights, "category_weights", None)
        if cw is None:
            return
        for key, m in mults.items():
            if key in cw:
                cw[key] = float(cw[key]) * float(m)
            else:
                cw[key] = float(m)

    def _shrink_specs(self, specs, best_x, factor, group_prefixes):
        """Return a new specs list with matching params' bounds tightened
        around ``best_x[i]``. ``ParameterSpec`` is frozen — we use replace().
        """
        if factor >= 1.0 or not group_prefixes:
            return specs
        new_specs = []
        for s, x in zip(specs, best_x):
            match = any(s.name.startswith(p) for p in group_prefixes)
            if match:
                half = (float(s.hi) - float(s.lo)) * 0.5 * float(factor)
                if half > 1e-12:
                    new_lo = max(float(s.lo), float(x) - half)
                    new_hi = min(float(s.hi), float(x) + half)
                    if new_hi > new_lo:
                        s = dataclasses.replace(s, lo=new_lo, hi=new_hi)
            new_specs.append(s)
        return new_specs

    def _run_validation_gate(self, loss, specs, best_x, it):
        """Run PhysicalValidator on best_x, apply AdaptiveAdjustment on fail.

        Returns (new_specs, passed). ``new_specs`` differs from ``specs`` only
        when bounds were shrunk. ``passed`` triggers optional early stop.
        """
        from ..physical_validation import derive_adjustment
        params = self._params_dict(specs, best_x)
        try:
            report = self.validator.validate(params)
        except Exception as exc:
            print(f"  [validation it={it}] validator raised: {exc}")
            self.validation_log.append(
                {"iter": it, "passed": False, "error": str(exc)})
            return specs, False
        print(f"  [validation it={it}] pass={report.pass_}  "
              f"category_worst={ {k: f'{v*100:.1f}%' for k,v in report.category_worst.items()} }")
        # Capture per-row reasons so the post-mortem JSON shows *why* each
        # row failed (LAMMPS timeout, no comparable observable, tol blown,
        # etc.) rather than just the aggregate worst-error.
        reason_hist: dict = {}
        for r in report.rows:
            if not r.passed:
                key = r.reason or "unknown"
                # Group by leading prefix so "LAMMPS timeout: ..." aggregates.
                short = key.split(":", 1)[0] if ":" in key else key
                reason_hist[short] = reason_hist.get(short, 0) + 1
        log_entry = {
            "iter": it,
            "passed": bool(report.pass_),
            "category_worst": dict(report.category_worst),
            "n_rows": len(report.rows),
            "n_failed": sum(1 for r in report.rows if not r.passed),
            "failure_reasons": reason_hist,
        }
        if report.pass_:
            self.validation_log.append(log_entry)
            return specs, True
        adj = derive_adjustment(report,
                                boost_factor=self.val_boost,
                                shrink_factor=self.val_shrink)
        print(f"  [validation it={it}] {adj.reason}")
        self._apply_loss_weight_multipliers(loss, adj.loss_weight_multipliers)
        new_specs = self._shrink_specs(specs, best_x,
                                       adj.bound_shrink_factor,
                                       adj.bound_shrink_groups)
        log_entry["adjustment"] = adj.reason
        self.validation_log.append(log_entry)
        return new_specs, False

    # ----- main loop ---------------------------------------------------------

    def optimize(self,
                 loss: Callable[[np.ndarray], float],
                 specs: List,
                 budget: int,
                 rng: np.random.Generator) -> OptimizerResult:
        t0 = time.time()
        wrapped = self._wrap_loss(loss)
        # `live_specs` is the working spec list; the validation gate may shrink
        # bounds around the current best mid-run, and the next LHS / candidate
        # sampling respects the tightened box.
        live_specs = list(specs)
        lo, hi = bounds_array(live_specs)
        d = len(live_specs)

        # Warm start with LHS (Manuscript: 50 init points; budget-scaled here).
        n_init = min(self.n_init, max(5, budget // 5))
        X = latin_hypercube(n_init, live_specs, rng=rng)
        y = np.array([wrapped(x) for x in X])
        history = list(y)

        gp_var_log: List[float] = []
        stage_log: List[str] = ["init"] * n_init
        best_so_far = float(np.min(y))
        unimproved = 0
        beta = self.ucb_beta_init

        remaining = budget - n_init
        # tqdm progress bar over the post-LHS BO iterations. Postfix shows
        # the current best loss + stage so PyCharm console viewers can see
        # the BO actually descending. leave=True so the final position
        # stays in the log after the rep completes.
        try:
            from tqdm.auto import tqdm as _tqdm
            it_iter = _tqdm(range(remaining), total=remaining,
                            desc=f"PIBO BO (phys={self.physics_informed})",
                            unit="it", dynamic_ncols=True)
        except Exception:
            it_iter = range(remaining)
        for it in it_iter:
            # Online window trimming before GP fit (manuscript-style).
            X_fit, y_fit = self._trim_window(X, y, n_init)
            Xn = normalize(X_fit, live_specs)
            self.gp.fit(Xn, y_fit)

            stage = self._stage(it, remaining)
            cand_u = rng.random((self.n_acq_candidates, d))
            cand_x = denormalize(cand_u, live_specs)

            score = self._score_candidates(cand_u, best_so_far, stage,
                                           beta=beta, rng=rng)

            if self.physics_informed and self.penalty is not None:
                acq = ConstrainedAcquisition(lambda U, s=score: s,
                                             self.penalty)
                score = acq(cand_u)

            x_next = cand_x[int(np.argmax(score))]
            y_next = wrapped(x_next)

            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            history.append(y_next)
            stage_log.append(stage)

            # Track GP predictive variance for the visualization.
            _, std = self.gp.predict(cand_u[:64])
            gp_var_log.append(float(np.mean(std ** 2)))

            # UCB β decay (Manuscript: β decays during exploration).
            beta = max(0.5, beta * self.ucb_beta_decay)

            # Patience-based early stopping (Manuscript: 50-iter patience
            # after a 200-iter burn-in).
            if y_next < best_so_far - 1e-9:
                best_so_far = float(y_next)
                unimproved = 0
            else:
                unimproved += 1
            # Live status on the tqdm bar (no-op when tqdm wasn't loaded).
            if hasattr(it_iter, "set_postfix"):
                it_iter.set_postfix(best=f"{best_so_far:.4f}",
                                    stage=stage, beta=f"{beta:.2f}",
                                    refresh=False)
            if (self.patience > 0
                    and (n_init + it) >= self.burn_in
                    and unimproved >= self.patience):
                break

            # Physical-validation gate every K iters (PIBO-only).
            if (self.validator is not None
                    and self.validate_every_k > 0
                    and ((it + 1) % self.validate_every_k == 0)):
                best_idx_now = int(np.argmin(y))
                live_specs, passed = self._run_validation_gate(
                    loss, live_specs, X[best_idx_now], n_init + it)
                if passed and self.val_early_stop:
                    print(f"  [validation it={n_init+it}] all rows within "
                          f"tol — early stopping PIBO.")
                    break

        best_idx = int(np.argmin(y))
        return OptimizerResult(
            best_x=X[best_idx],
            best_loss=float(y[best_idx]),
            history=history,
            n_evals=len(history),
            wall_clock_s=time.time() - t0,
            extras={
                "gp_predictive_var": gp_var_log,
                "stage_log": stage_log,
                "all_X": X,
                "all_y": y,
                "validation_log": list(self.validation_log),
            },
        )
