"""
BenchmarkSuite: switch between PIBO / JAX-ReaxFF / INDEEDopt / PSO and
report a unified pandas DataFrame plus the four required plots.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .lammps_runner import LAMMPSRunner
from .loss import ReaxFFLoss, LossWeights
from .memory import free_memory, memory_usage_mb
from .parameters import ParameterSpec, parameters_for_blocks
from .physics_constraints import PhysicsPenalty
from .vasp_reader import load_dataset, train_validation_split
from .optimizers import OPTIMIZER_REGISTRY, OptimizerResult


@dataclass
class RunRecord:
    """One (optimizer, replicate) result with all metrics needed for the table."""
    optimizer: str
    replicate: int
    best_loss: float
    loss_mae: float
    loss_rmse: float
    E_total_error: float
    Ys_mean: float
    Ys_std: float
    n_evals: int
    fevals_to_convergence: int
    param_robustness: float
    val_rmse: float
    wall_clock_s: float
    cpu_hours: float
    history: List[float] = field(default_factory=list)
    best_x: np.ndarray = field(default_factory=lambda: np.array([]))
    extras: Dict = field(default_factory=dict)
    physics_informed: bool = True


class BenchmarkSuite:
    """End-to-end benchmark runner."""

    def __init__(self,
                 system: str = "MoSH",
                 config: str | Dict | None = "configs/default_mosh.json",
                 dft_root: Optional[str] = None,
                 lammps_mode: str = "lammps",
                 profile: Optional[str] = None,
                 validator=None,
                 validate_every_k: int = 0,
                 val_boost: float = 2.0,
                 val_shrink: float = 0.7,
                 val_early_stop: bool = False):
        """
        Parameters
        ----------
        profile : "demo" | "manuscript" | None
            Pull optimization settings from ``config['optimization'][profile]``
            when set. ``None`` falls back to the config's default profile (or
            the legacy flat layout for old configs).
        """
        self.system = system
        self.config = self._load_config(config)
        self.profile = profile or self.config["optimization"].get(
            "profile", "demo")
        self.opt_cfg = self._resolve_profile(self.config, self.profile)

        # Locate DFT data: try dft_root arg, then config primary, then alternatives.
        self.dft_root = self._locate_dft_root(dft_root)
        self.blocks = self.config.get(
            "blocks", ["bond", "angle", "torsion", "coulomb", "vdw"])

        self.specs: List[ParameterSpec] = parameters_for_blocks(self.blocks)

        # Pull DFT frames (flat list across blocks for the unified objective).
        # max_dE_per_atom_eV (default 3.0) drops frames outside ReaxFF's
        # bond-order saturation regime — see vasp_reader.load_dataset docstring.
        max_dE = float(self.config.get("dataset", {}).get(
            "max_dE_per_atom_eV", 3.0))
        dataset = (load_dataset(self.dft_root, self.blocks,
                                max_dE_per_atom_eV=max_dE)
                   if self.dft_root else {})
        flat = [fr for frames in dataset.values() for fr in frames]
        if not flat:
            print(f"[BenchmarkSuite] No DFT frames found at "
                  f"{self.dft_root!r}; using synthetic dummies "
                  f"(categories spread across Manuscript Table 1).")
            flat = self._synthetic_frames()
        val_ratio = self.opt_cfg.get("validation_split", 0.2)
        self.train_frames, self.val_frames = train_validation_split(
            flat, val_ratio=val_ratio, seed=0)

        # Wire ffield template into the runner so mode='lammps' has a real
        # Manuscript-anchored ffield to render for the LAMMPS evaluator.
        ffield_tmpl = self.config.get("lammps", {}).get("ffield_template")
        if ffield_tmpl and not os.path.isabs(ffield_tmpl):
            ffield_tmpl = os.path.join(os.path.dirname(__file__), "..",
                                       ffield_tmpl)
            ffield_tmpl = os.path.normpath(ffield_tmpl)
            if not os.path.exists(ffield_tmpl):
                ffield_tmpl = None
        self.runner = LAMMPSRunner(
            mode=lammps_mode,
            elements=tuple(self.config.get("elements", ["Mo", "S", "H"])),
            base_ffield=ffield_tmpl,
        )

        # Build LossWeights — accepts both new (with category_weights /
        # per_atom_norm) and legacy (energy/forces/geometry only) blocks.
        lw = dict(self.config.get("loss_weights",
                                  self.opt_cfg.get("loss_weights", {})))
        lw.pop("_about_categories", None)
        # Strip any keys LossWeights doesn't understand (forward-compat).
        valid = {"energy", "forces", "geometry", "per_atom_norm",
                 "category_weights"}
        lw = {k: v for k, v in lw.items() if k in valid}
        self.weights = LossWeights(**lw)
        self.results: List[RunRecord] = []

        # Physical-validation gate (PIBO-only). validator may be None.
        self.validator         = validator
        self.validate_every_k  = int(validate_every_k)
        self.val_boost         = float(val_boost)
        self.val_shrink        = float(val_shrink)
        self.val_early_stop    = bool(val_early_stop)

    # ----- config -----------------------------------------------------------

    def _load_config(self, config) -> Dict:
        if isinstance(config, dict):
            return config
        if isinstance(config, str) and os.path.exists(config):
            with open(config, "r", encoding="utf-8") as f:
                return json.load(f)
        # Reasonable defaults so the suite runs even without a config file.
        return {
            "system": self.system,
            "elements": ["Mo", "S", "H"],
            "dft_root": "../CODE/data/vasp_calculations",
            "blocks": ["bond", "angle", "torsion", "coulomb", "vdw"],
            "loss_weights": {"energy": 1.0, "forces": 0.0, "geometry": 0.0},
            "optimization": {
                "profile": "demo",
                "demo": {
                    "budget": 100, "replicates": 3, "init_lhs_points": 15,
                    "physics_informed": True, "validation_split": 0.2,
                },
            },
            "acquisition": {
                "stage_ucb_until": 0.30, "stage_ei_until": 0.70,
                "stage_pi_after": 0.70, "ucb_beta_init": 2.0,
                "ucb_beta_decay": 0.995, "ei_xi": 0.01,
                "thompson_prob": 0.15,
            },
            "physics_constraints": {"penalty_lambda": 1.0,
                                    "enforce_acquisition_constraint": True},
            "gp": {"kernel": "matern", "nu": 2.5, "ard": True,
                   "sparse_threshold": 100, "n_inducing": 50},
        }

    @staticmethod
    def _resolve_profile(cfg: Dict, profile: str) -> Dict:
        """Return the resolved optimization sub-config for ``profile``.

        Accepts both the new layout (``optimization.{demo,manuscript}``) and
        the legacy flat layout (``optimization`` with budget/replicates inline).
        """
        opt = cfg.get("optimization", {})
        if profile in opt and isinstance(opt[profile], dict):
            return opt[profile]
        # legacy flat layout
        return {k: v for k, v in opt.items()
                if k not in ("profile", "demo", "manuscript")
                and not k.startswith("_")}

    def _locate_dft_root(self, override: Optional[str]) -> Optional[str]:
        """Try the override, then config primary, then alternatives, then env."""
        candidates: List[Optional[str]] = []
        if override:
            candidates.append(override)
        candidates.append(self.config.get("dft_root"))
        candidates.extend(self.config.get("dft_root_alternatives", []))
        # ${PIBO_DFT_ROOT} expansion
        for c in list(candidates):
            if isinstance(c, str) and "${" in c:
                candidates.append(os.path.expandvars(c))
        for c in candidates:
            if not c:
                continue
            if "${" in c:
                continue
            p = os.path.abspath(c)
            if os.path.isdir(p):
                print(f"[BenchmarkSuite] dft_root resolved -> {p}")
                return p
        print("[BenchmarkSuite] no dft_root candidate exists; "
              "falling back to synthetic frames.")
        return None

    def _synthetic_frames(self):
        """Synthetic frames for the smoke-test path. Spreads frames across
        Manuscript Table 1 categories so the per-category weights are exercised.
        """
        from .vasp_reader import DFTFrame
        cats = ["equilibrium", "bond_pes", "angle_pes",
                "dihedral_pes", "strained"]
        rng = np.random.default_rng(0)
        out = []
        for i in range(10):
            n = 3 + (i % 3)
            cell = np.eye(3) * 10.0
            pos = rng.uniform(0, 5, size=(n, 3))
            sp = ["Mo", "S", "H"] * 3
            out.append(DFTFrame(tag=f"syn_{i}", energy=-50.0 - i * 0.5,
                                positions=pos, cell=cell, species=sp[:n],
                                category=cats[i % len(cats)]))
        return out

    # ----- core runner ------------------------------------------------------

    def run_all(self, replicates: int | None = None, budget: int | None = None,
                optimizers: List[str] | None = None,
                physics_informed: bool | None = None,
                ablation_physics_off: bool = True,
                rng_seed: int = 42) -> pd.DataFrame:
        """Run each optimizer for `replicates` independent seeds.

        When `ablation_physics_off=True`, also run a physics-off pass for the
        same optimizers so the physics-informed improvement plot is meaningful.
        """
        opt_cfg = self.opt_cfg
        replicates = replicates or opt_cfg.get("replicates", 3)
        budget = budget or opt_cfg.get("budget", 100)
        optimizers = optimizers or list(OPTIMIZER_REGISTRY.keys())
        physics_informed = (physics_informed if physics_informed is not None
                            else opt_cfg.get("physics_informed", True))

        penalty = PhysicsPenalty(
            specs=self.specs,
            lambda_=self.config["physics_constraints"]["penalty_lambda"],
        )

        runs = []
        for phys in [True, False] if ablation_physics_off else [physics_informed]:
            for opt_name in optimizers:
                for rep in range(replicates):
                    seed = rng_seed + rep * 100
                    rng = np.random.default_rng(seed)
                    loss = ReaxFFLoss(self.runner, self.specs,
                                      self.train_frames, self.val_frames,
                                      weights=self.weights)
                    OptCls = OPTIMIZER_REGISTRY[opt_name]
                    opt_config = {
                        "gp":          self.config.get("gp", {}),
                        "acquisition": self.config.get("acquisition", {}),
                        "init_lhs_points":
                            opt_cfg.get("init_lhs_points", 15),
                        "patience":    opt_cfg.get("patience", 0),
                        "burn_in":     opt_cfg.get("burn_in", 0),
                        "online_window_recent":
                            opt_cfg.get("online_window_recent", 0),
                        "online_window_initial":
                            opt_cfg.get("online_window_initial", 0),
                    }
                    # PIBO-only: attach validator + knobs. Other optimizers
                    # never see these keys so the gate is genuinely PIBO-only.
                    if opt_name == "pibo" and self.validator is not None:
                        opt_config.update({
                            "validator":         self.validator,
                            "validate_every_k":  self.validate_every_k,
                            "val_boost":         self.val_boost,
                            "val_shrink":        self.val_shrink,
                            "val_early_stop":    self.val_early_stop,
                        })
                    optimizer = OptCls(
                        physics_informed=phys, penalty=penalty,
                        config=opt_config)
                    t0 = time.time()
                    result: OptimizerResult = optimizer.optimize(
                        loss=loss, specs=self.specs, budget=budget, rng=rng)
                    cpu_hours = (time.time() - t0) / 3600.0

                    record = self._summarize(
                        opt_name=opt_name, rep=rep, phys=phys, result=result,
                        loss_obj=loss, cpu_hours=cpu_hours,
                    )
                    runs.append(record)
                    self.results.append(record)
                    free_memory()
                    print(f"  [{opt_name} rep={rep} phys={phys}] "
                          f"loss={record.best_loss:.4f}  "
                          f"val={record.val_rmse:.4f}  "
                          f"cpu_h={record.cpu_hours:.3f}  "
                          f"mem={memory_usage_mb():.0f}MB")

        return self.report()

    def _summarize(self, opt_name, rep, phys, result, loss_obj, cpu_hours
                   ) -> RunRecord:
        # First-iteration-below-95%-of-final-improvement = fevals_to_conv.
        h = np.array(result.history)
        if len(h) > 1:
            cmin = np.minimum.accumulate(h)
            target = cmin[-1] + 0.05 * (cmin[0] - cmin[-1])
            fevals_to_conv = int(np.argmax(cmin <= target)) + 1
        else:
            fevals_to_conv = result.n_evals

        # IMPORTANT ordering: validation_accuracy() and parameter_robustness()
        # *both* call loss(perturbed_params), which overwrites loss_obj.last_ys
        # and last_params. We therefore (a) take val/robustness first, then
        # (b) re-evaluate loss at the *best* point so metrics() reflects best_x
        # rather than the final perturbation.
        val_rmse = loss_obj.validation_accuracy(result.best_x)
        robustness = loss_obj.parameter_robustness(result.best_x,
                                                   n_perturb=5, sigma=0.03)
        _ = loss_obj(result.best_x)  # refresh last_ys / last_params on best_x
        metrics = loss_obj.metrics()
        ys = loss_obj.last_ys if loss_obj.last_ys is not None else np.array([np.nan])

        return RunRecord(
            optimizer=opt_name,
            replicate=rep,
            best_loss=float(result.best_loss),
            loss_mae=float(metrics.get("loss_mae", np.nan)),
            loss_rmse=float(metrics.get("loss_rmse", np.nan)),
            E_total_error=float(metrics.get("E_total_error", np.nan)),
            Ys_mean=float(np.nanmean(ys)),
            Ys_std=float(np.nanstd(ys)),
            n_evals=int(result.n_evals),
            fevals_to_convergence=fevals_to_conv,
            param_robustness=float(robustness),
            val_rmse=float(val_rmse),
            wall_clock_s=float(result.wall_clock_s),
            cpu_hours=float(cpu_hours),
            history=h.tolist(),
            best_x=np.asarray(result.best_x),
            extras=dict(result.extras),
            physics_informed=bool(phys),
        )

    # ----- reporting --------------------------------------------------------

    def report(self) -> pd.DataFrame:
        """Build the headline benchmark table."""
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                "optimizer": r.optimizer,
                "physics_informed": r.physics_informed,
                "replicate": r.replicate,
                "Loss(RMSE)": r.loss_rmse,
                "Loss(MAE)": r.loss_mae,
                "Ys_mean": r.Ys_mean,
                "Ys_std": r.Ys_std,
                "E_total_error": r.E_total_error,
                "CPU_hours": r.cpu_hours,
                "wall_s": r.wall_clock_s,
                "n_evals": r.n_evals,
                "fevals_to_conv": r.fevals_to_convergence,
                "param_robustness": r.param_robustness,
                "val_RMSE": r.val_rmse,
            })
        df = pd.DataFrame(rows)
        return df

    def report_aggregate(self) -> pd.DataFrame:
        """Mean ± std table grouped by (optimizer, physics_informed)."""
        df = self.report()
        if df.empty:
            return df
        agg = df.groupby(["optimizer", "physics_informed"]).agg(
            loss_mean=("Loss(RMSE)", "mean"),
            loss_std=("Loss(RMSE)", "std"),
            E_err=("E_total_error", "mean"),
            cpu_h=("CPU_hours", "mean"),
            n_evals=("n_evals", "mean"),
            conv_fevals=("fevals_to_conv", "mean"),
            robustness=("param_robustness", "mean"),
            val_rmse=("val_RMSE", "mean"),
        ).reset_index()
        return agg

    def to_csv(self, path: str) -> None:
        self.report().to_csv(path, index=False)

    # ----- plotting passthrough --------------------------------------------

    def plot_all(self, save_dir: str | None = None):
        from . import visualization as viz
        figs = {}
        figs["replication_variance"] = viz.plot_optimizer_replication_variance(
            self.results, save_dir=save_dir)
        figs["gp_predictive_variance"] = viz.plot_gp_predictive_variance(
            self.results, save_dir=save_dir)
        figs["parameter_posterior"] = viz.plot_parameter_posterior(
            self.results, self.specs, save_dir=save_dir)
        figs["physics_informed_improvement"] = viz.plot_physics_improvement(
            self.results, save_dir=save_dir)
        return figs
