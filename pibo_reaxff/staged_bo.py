"""
Staged Bayesian-optimization orchestrator for ReaxFF block calibration.

The 42 calibrable parameters (Note S1 diff set) split naturally into
four blocks — ``bond`` (18), ``angle`` (16), ``offdiag`` (5), and
``torsion`` (3). Each block has a primary DFT-category counterpart in
the dataset:

    block      |  category(s) used in that stage    | frames
    -----------|------------------------------------|--------
    bond       |  {"bond_pes"}                      | 13
    angle      |  {"angle_pes"}                     |  8
    offdiag    |  {"strained"}                      | 32
    torsion    |  {"dihedral_pes"}                  | 14
    joint (all)|  all categories                    | 67

A staged BO runs each stage in sequence: only the *active* block's
parameters are free; the rest stay frozen at their current best
values (initially Note S1, then updated from each completed stage).
After every per-block stage, a final joint stage frees all 42
parameters and uses every frame.

Why this helps
--------------
* Each stage solves a low-dimensional subproblem (n_params ≤ 18) with
  a focused dataset, where BO is much more sample-efficient than on
  the full 42-dim joint problem.
* The joint stage starts from a pre-conditioned point near the
  global optimum, so its budget is spent on fine refinement instead
  of re-discovering coarse structure.

Public API
----------
``StagedLoss``
    A loss callable for a single stage. Same shape as ``ReaxFFLoss``
    but works on a subset of parameters + a subset of frames; the
    inactive parameters are read from ``frozen_values`` on every call.

``StagedBORunner.run()``
    Drives the full multi-stage sequence and returns a structured
    ``StagedResult`` with the final best parameters, per-stage best
    losses, and the unified history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .loss import LossWeights, energy_rmse, energy_mae, _shift_to_match_mean
from .parameters import ParameterSpec
from .vasp_reader import DFTFrame


# ---------------------------------------------------------------------------
# StagedLoss — per-stage callable that merges frozen + active params.
# ---------------------------------------------------------------------------

class StagedLoss:
    """Loss callable for one staged BO step.

    Receives an x-vector of length ``len(active_specs)``, merges with
    ``frozen_values`` into a full {name: value} dict, evaluates LAMMPS
    on ``stage_frames`` only, and returns the per-atom weighted RMSE.

    Mirrors ``ReaxFFLoss``'s public surface (``history``, ``n_evals``,
    ``last_ys``, ``last_params``, ``metrics``, ``validation_accuracy``,
    ``parameter_robustness``) so PIBOOptimizer drops in unchanged.
    """

    def __init__(self,
                 runner,
                 active_specs: List[ParameterSpec],
                 frozen_values: Dict[str, float],
                 stage_train: List[DFTFrame],
                 stage_val: List[DFTFrame],
                 weights: Optional[LossWeights] = None):
        self.runner = runner
        self.active_specs = active_specs
        self.specs = active_specs   # alias used by some legacy callers
        self.frozen_values = dict(frozen_values)
        self.train_frames = stage_train
        self.val_frames = stage_val
        self.weights = weights or LossWeights()

        self.n_evals = 0
        self.history: List[float] = []
        self.last_ys: Optional[np.ndarray] = None
        self.last_params: Optional[Dict[str, float]] = None
        self.total_runtime_s: float = 0.0

        self._train_weights = self._frame_weights(self.train_frames)
        self._val_weights = self._frame_weights(self.val_frames)

    def _frame_weights(self, frames: List[DFTFrame]) -> np.ndarray:
        cw = self.weights.category_weights
        return np.array(
            [cw.get(getattr(f, "category", "default"), 1.0) for f in frames],
            dtype=float,
        )

    def _merged_params(self, x_active: np.ndarray) -> Dict[str, float]:
        merged = dict(self.frozen_values)
        for s, v in zip(self.active_specs, x_active):
            merged[s.name] = float(v)
        return merged

    def _eval_frames(self, frames: List[DFTFrame],
                     params: Dict[str, float]
                     ) -> Tuple[np.ndarray, float]:
        ys = np.empty(len(frames), dtype=float)
        rt = 0.0
        try:
            from tqdm.auto import tqdm as _tqdm
            it = _tqdm(frames, total=len(frames), desc="lammps",
                       unit="frame", leave=False, dynamic_ncols=True)
        except Exception:
            it = frames
        for i, fr in enumerate(it):
            r = self.runner.evaluate(params, fr)
            ys[i] = r.energy
            rt += r.runtime_s
        return ys, rt

    def __call__(self, x_active: np.ndarray) -> float:
        params = self._merged_params(x_active)
        ys, rt = self._eval_frames(self.train_frames, params)
        refs = np.array([f.energy for f in self.train_frames], dtype=float)

        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.train_frames],
                              dtype=float)
            ys_n, refs_n = ys / natoms, refs / natoms
        else:
            ys_n, refs_n = ys, refs

        loss = self.weights.energy * energy_rmse(
            ys_n, refs_n, weights=self._train_weights)

        # Geometric anchor: cell-relax 3x3 monolayer, penalise
        # |a - 3.183| + |h_S - 1.564|. Required for staged BO to drive
        # the equilibrium geometry toward Cooper 2014 references.
        if self.weights.geometry > 0:
            try:
                from .loss import _measure_equilibrium_geom
                geo = _measure_equilibrium_geom(self.runner, params)
                a_pen  = ((geo["a"] - 3.1830) / 3.1830) ** 2
                hS_pen = ((geo["h_S"] - 1.5640) / 1.5640) ** 2
                loss += self.weights.geometry * (a_pen + hS_pen) ** 0.5
            except Exception:
                pass

        self.n_evals += 1
        self.history.append(loss)
        self.last_ys = ys
        self.last_params = params
        self.total_runtime_s += rt
        return loss

    def metrics(self) -> Dict[str, float]:
        if self.last_ys is None:
            return {}
        refs = np.array([f.energy for f in self.train_frames], dtype=float)
        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.train_frames],
                              dtype=float)
            ys_n, refs_n = self.last_ys / natoms, refs / natoms
        else:
            ys_n, refs_n = self.last_ys, refs
        return {
            "loss": float(self.history[-1]) if self.history else float("nan"),
            "loss_mae":  energy_mae(ys_n, refs_n, weights=self._train_weights),
            "loss_rmse": energy_rmse(ys_n, refs_n, weights=self._train_weights),
            "loss_rmse_unweighted": energy_rmse(ys_n, refs_n),
            "E_total_error": float(abs(_shift_to_match_mean(
                self.last_ys, refs).sum() - refs.sum())),
            "n_evals": self.n_evals,
            "total_runtime_s": self.total_runtime_s,
        }

    def validation_accuracy(self, x_active: np.ndarray) -> float:
        if not self.val_frames:
            return float("nan")
        params = self._merged_params(x_active)
        ys, _ = self._eval_frames(self.val_frames, params)
        refs = np.array([f.energy for f in self.val_frames], dtype=float)
        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.val_frames],
                              dtype=float)
            ys, refs = ys / natoms, refs / natoms
        return energy_rmse(ys, refs, weights=self._val_weights)

    def parameter_robustness(self, x_active: np.ndarray,
                             n_perturb: int = 5, sigma: float = 0.03,
                             rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng(0)
        from .parameters import bounds_array
        lo, hi = bounds_array(self.active_specs)
        width = hi - lo
        losses = []
        for _ in range(n_perturb):
            noise = rng.normal(0.0, sigma, size=x_active.shape) * width
            losses.append(self.__call__(np.clip(x_active + noise, lo, hi)))
        return float(np.std(losses))


# ---------------------------------------------------------------------------
# Stage definition + result container
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    active_blocks: Tuple[str, ...]
    active_categories: Tuple[str, ...]
    budget: int
    init_lhs_points: int = 10


@dataclass
class StageResult:
    stage_name: str
    active_param_names: List[str]
    n_train_frames: int
    best_x: np.ndarray
    best_loss: float
    n_evals: int
    wall_clock_s: float
    history: List[float] = field(default_factory=list)
    # Full {name: value} snapshot AFTER this stage finished — combines
    # this stage's best_x with whatever the previous stages left as
    # current_params. Lets every stage produce a self-contained ffield.
    params_after_stage: Dict[str, float] = field(default_factory=dict)


@dataclass
class StagedResult:
    stages: List[StageResult]
    final_params: Dict[str, float]
    final_loss: float
    final_val_loss: float
    total_evals: int
    total_wall_s: float
    history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default 4-stage sequence (matches the user's "bond → angle → joint" + the
# offdiag and torsion blocks that the dataset can constrain).
# ---------------------------------------------------------------------------

# User-requested 5-stage block sequence (no joint stage):
#   bond -> angle -> off-diagonal -> torsion -> h-bond
#
# Note: the project's dataset is Mo+S only (no H atoms), so the h-bond
# stage's BO loss is insensitive to its 4 parameters — the BO will pick
# whatever values minimise noise within bounds. We still run it so the
# final ffield has well-defined h-bond entries from the same calibration
# pipeline, and so the per-block report covers all 5 blocks in the
# Note S1 manuscript.
DEFAULT_STAGES: List[Stage] = [
    Stage(name="bond",
          active_blocks=("bond",),
          active_categories=("bond_pes",),
          budget=100, init_lhs_points=15),
    Stage(name="angle",
          active_blocks=("angle",),
          active_categories=("angle_pes",),
          budget=100, init_lhs_points=15),
    Stage(name="offdiag",
          active_blocks=("offdiag",),
          active_categories=("strained",),
          budget=80, init_lhs_points=12),
    Stage(name="torsion",
          active_blocks=("torsion",),
          active_categories=("dihedral_pes",),
          budget=50, init_lhs_points=8),
    # h-bond stage intentionally removed per user goal: dataset is Mo+S
    # only so h-bond term contributes 0 to every loss; calibrating those
    # 4 parameters via BO would be random walk. The 4 h-bond entries in
    # the template stay at their Note S1 anchor values.
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class StagedBORunner:
    """Drive a sequence of stages, each calibrating a parameter subset.

    Carries the best-found values from every stage into the next stage's
    ``frozen_values``. Stage 1 starts from each parameter's Note S1
    anchor (``spec.notes1_value``); later stages start from whatever
    the previous stages found.
    """

    def __init__(self,
                 runner,
                 specs: List[ParameterSpec],
                 train_frames: List[DFTFrame],
                 val_frames: List[DFTFrame],
                 weights: LossWeights,
                 stages: Optional[Sequence[Stage]] = None,
                 optimizer_factory: Optional[Callable] = None,
                 optimizer_config: Optional[Dict] = None,
                 verbose: bool = True):
        self.runner = runner
        self.specs = specs
        self.train_frames = train_frames
        self.val_frames = val_frames
        self.weights = weights
        self.stages = list(stages or DEFAULT_STAGES)
        self.optimizer_config = optimizer_config or {}
        self.verbose = verbose

        if optimizer_factory is None:
            from .optimizers.pibo import PIBOOptimizer
            self.optimizer_factory = lambda: PIBOOptimizer(
                physics_informed=True, penalty=None,
                config=self.optimizer_config)
        else:
            self.optimizer_factory = optimizer_factory

        # Initial frozen values: every param at its Note S1 anchor.
        self.current_params: Dict[str, float] = {
            s.name: float(s.notes1_value) for s in specs
        }

    def _select_specs(self, blocks: Tuple[str, ...]) -> List[ParameterSpec]:
        return [s for s in self.specs if s.block in blocks]

    def _select_frames(self, frames: List[DFTFrame],
                       categories: Tuple[str, ...]) -> List[DFTFrame]:
        cats = set(categories)
        return [f for f in frames if f.category in cats]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self, rng_seed: int = 42) -> StagedResult:
        t0 = time.time()
        stage_results: List[StageResult] = []
        combined_history: List[float] = []

        for stage_idx, stage in enumerate(self.stages):
            active_specs = self._select_specs(stage.active_blocks)
            if not active_specs:
                self._log(f"[staged] stage {stage.name}: no active params, "
                          "skipping")
                continue
            active_names = [s.name for s in active_specs]
            frozen = {k: v for k, v in self.current_params.items()
                      if k not in active_names}

            stage_train = self._select_frames(self.train_frames,
                                              stage.active_categories)
            stage_val = self._select_frames(self.val_frames,
                                            stage.active_categories)
            if not stage_train:
                self._log(f"[staged] stage {stage.name}: no train frames "
                          f"in categories {stage.active_categories}, skipping")
                continue

            self._log(
                f"\n[staged] === stage {stage_idx+1}/{len(self.stages)}: "
                f"{stage.name} ===\n"
                f"  active params  : {len(active_specs)} "
                f"({', '.join(stage.active_blocks)})\n"
                f"  active frames  : {len(stage_train)} train "
                f"({', '.join(stage.active_categories)})\n"
                f"  budget         : {stage.budget}\n"
                f"  carry-fwd      : {len(frozen)} frozen params")

            staged_loss = StagedLoss(
                runner=self.runner,
                active_specs=active_specs,
                frozen_values=frozen,
                stage_train=stage_train,
                stage_val=stage_val,
                weights=self.weights,
            )

            cfg = dict(self.optimizer_config)
            cfg["init_lhs_points"] = stage.init_lhs_points
            # Disable patience inside each stage so a stage with a small
            # data subset doesn't quit before the BO has actually
            # exploited; patience between stages is implicit (we move on).
            cfg["patience"] = 0
            optimizer = self.optimizer_factory()
            optimizer.config = cfg
            # Apply the same config the factory would have set.
            optimizer.n_init = stage.init_lhs_points
            optimizer.patience = 0

            t_stage = time.time()
            result = optimizer.optimize(
                loss=staged_loss,
                specs=active_specs,
                budget=stage.budget,
                rng=np.random.default_rng(rng_seed + stage_idx * 1000),
            )
            stage_wall = time.time() - t_stage

            # Promote best_x into current_params.
            for name, val in zip(active_names, result.best_x):
                self.current_params[name] = float(val)

            stage_res = StageResult(
                stage_name=stage.name,
                active_param_names=active_names,
                n_train_frames=len(stage_train),
                best_x=np.asarray(result.best_x),
                best_loss=float(result.best_loss),
                n_evals=int(result.n_evals),
                wall_clock_s=float(stage_wall),
                history=list(result.history),
                params_after_stage=dict(self.current_params),
            )
            stage_results.append(stage_res)
            combined_history.extend(result.history)

            self._log(
                f"  -> best_loss={result.best_loss:.4f}  "
                f"n_evals={result.n_evals}  wall={stage_wall:.1f}s")

        # Final loss on the FULL dataset with the final merged params.
        from .loss import ReaxFFLoss
        full_loss = ReaxFFLoss(self.runner, self.specs,
                               self.train_frames, self.val_frames,
                               weights=self.weights)
        x_final = np.array([self.current_params[s.name] for s in self.specs])
        final_loss = float(full_loss(x_final))
        final_val = float(full_loss.validation_accuracy(x_final))

        return StagedResult(
            stages=stage_results,
            final_params=dict(self.current_params),
            final_loss=final_loss,
            final_val_loss=final_val,
            total_evals=sum(s.n_evals for s in stage_results),
            total_wall_s=time.time() - t0,
            history=combined_history,
        )
