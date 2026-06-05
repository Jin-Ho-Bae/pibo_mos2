from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Make sibling `pibo_reaxff/` importable when run.py is the entry point.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pibo_reaxff.local_env import ensure_lammps, repo_root, sanity_check  # noqa: E402
from pibo_reaxff import ezff_io, ezff_error  # noqa: E402  (format-only EZFF adapter)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    # Deferred import so `--help` / `--check-only` work before pip install.
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_base_json(path: Path) -> Dict[str, Any]:
    """Load configs/default_mosh.json which BenchmarkSuite expects."""
    # encoding=utf-8 already; nothing more needed.
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _merge_yaml_into_base(base: Dict[str, Any],
                          yc: Dict[str, Any],
                          profile: str) -> Dict[str, Any]:
    """Overlay the run-time YAML knobs onto the default JSON config.

    The JSON file has the shape BenchmarkSuite already understands; we just
    write the YAML values into the right slots.
    """
    cfg = deepcopy(base)

    # Profile and its BO knobs.
    cfg["optimization"]["profile"] = profile
    prof = cfg["optimization"].setdefault(profile, {})
    bo = yc.get("bo", {})
    for key in ("budget", "replicates", "init_lhs_points",
                "patience", "burn_in", "physics_informed"):
        if key in bo:
            prof[key] = bo[key]

    # Loss weights.
    lw = yc.get("loss_weights", {})
    cfg.setdefault("loss_weights", {})
    for key in ("energy", "forces", "geometry", "per_atom_norm"):
        if key in lw:
            cfg["loss_weights"][key] = lw[key]
    if "category_weights" in lw:
        cfg["loss_weights"].setdefault("category_weights", {}).update(
            lw["category_weights"])

    # Dataset-filter knobs.
    ds = yc.get("dataset", {})
    if ds:
        cfg.setdefault("dataset", {}).update(ds)

    # Acquisition.
    aq = yc.get("acquisition", {})
    if aq:
        cfg.setdefault("acquisition", {}).update(aq)
        # PIBO reads stage_pi_after; keep it in sync with stage_ei_until.
        if "stage_ei_until" in aq:
            cfg["acquisition"]["stage_pi_after"] = aq["stage_ei_until"]

    # Physics constraints.
    pc = yc.get("physics_constraints", {})
    if pc:
        cfg.setdefault("physics_constraints", {}).update(pc)

    return cfg


# ---------------------------------------------------------------------------
# Validator construction
# ---------------------------------------------------------------------------

def _build_validator(val_cfg: Dict[str, Any],
                     log: logging.Logger):
    """Construct a PhysicalValidator from the YAML 'validation' block.

    Returns None if validation is disabled or the CSV is missing.
    """
    if not val_cfg.get("enabled", False):
        log.info("validation: disabled")
        return None

    csv_path = repo_root() / val_cfg.get("csv_path",
                                         "MoS2_physical_validation.csv")
    if not csv_path.exists():
        log.warning("validation: CSV not found at %s — disabling gate", csv_path)
        return None

    from pibo_reaxff.lammps_runner import LAMMPSRunner
    from pibo_reaxff.physical_validation import PhysicalValidator

    template = (repo_root() / "lammps_templates" /
                "ffield.reax.MoSH.template")
    runner = LAMMPSRunner(
        elements=("Mo", "S", "H"),
        base_ffield=str(template),
    )
    validator = PhysicalValidator(
        csv_path=str(csv_path),
        runner=runner,
        tol=float(val_cfg.get("tol", 0.05)),
        subsample=val_cfg.get("subsample", "default"),
        supercell_mono=tuple(val_cfg.get("supercell_mono", (4, 4))),
        supercell_bulk=tuple(val_cfg.get("supercell_bulk", (3, 3, 1))),
        npt_ps=float(val_cfg.get("npt_ps", 5.0)),
        observables=tuple(val_cfg.get("observables", ("a", "stress"))),
    )
    log.info("validation: %d rows selected (tol=%.1f%%, every_k=%d)",
             len(validator.rows),
             validator.tol * 100,
             int(val_cfg.get("every_k", 0)))
    return validator


def _ezff_default_bounds_path() -> Path:
    return _HERE / "configs" / "ezff_variable_bounds.txt"


def _ezff_default_template_path() -> Path:
    return _HERE / "lammps_templates" / "ffield.reax.MoSH.template"


def _ezff_resolve_params(mode: str,
                         bounds: Dict[str, List[float]]) -> Dict[str, float]:
    """Resolve ``{name: value}`` from one of three modes.

    Modes
    -----
    ``midpoint``
        ``(lo + hi) / 2`` for every key. Cheap default.

    ``manuscript``
        Use ``REAXFF_PARAMETERS.manuscript_mean`` where the spec carries
        a Table-2-anchored posterior mean (25 of 53); fall back to
        midpoint for the rest. The number of anchored vs midpoint
        parameters is logged so the operator can spot bound drift.

    *path-to-JSON*
        Load ``{name: value}`` from a JSON file. Must cover every key
        in ``bounds`` (an explicit subset would be ambiguous against
        the template's column-sensitive layout).
    """
    if mode == "midpoint":
        return {k: 0.5 * (v[0] + v[1]) for k, v in bounds.items()}

    if mode == "manuscript":
        from pibo_reaxff.parameters import REAXFF_PARAMETERS
        by_name = {p.name: p for p in REAXFF_PARAMETERS}
        out: Dict[str, float] = {}
        n_anchor = 0
        for name, (lo, hi) in bounds.items():
            spec = by_name.get(name)
            if spec is not None and spec.manuscript_mean is not None:
                out[name] = float(spec.manuscript_mean)
                n_anchor += 1
            else:
                out[name] = 0.5 * (float(lo) + float(hi))
        logging.getLogger("run").info(
            "ezff-params=manuscript: %d/%d anchored, %d via midpoint",
            n_anchor, len(bounds), len(bounds) - n_anchor)
        return out

    p = Path(mode)
    if not p.exists():
        raise FileNotFoundError(
            f"--ezff-params expects 'midpoint' | 'manuscript' | <path.json>; "
            f"got {mode!r} (not a known mode and not an existing file).")
    with p.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    missing = set(bounds.keys()) - set(loaded.keys())
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"--ezff-params JSON is missing {len(missing)} bound keys; "
            f"first few: {sample}. Every variable_bounds key must be set.")
    return {k: float(loaded[k]) for k in bounds.keys()}


def _fmt_err(v: float) -> str:
    """NaN-safe formatter for the [E, F, geom] vector."""
    return "nan" if (v != v) else f"{v:.6f}"  # NaN-check via inequality


def _run_ezff_render(args: argparse.Namespace,
                     log: logging.Logger) -> int:
    bounds = ezff_io.read_variable_bounds(str(args.ezff_bounds))
    template = ezff_io.read_forcefield_template(str(args.ezff_template))
    params = _ezff_resolve_params(args.ezff_params, bounds)
    out_path = Path(args.ezff_render).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ezff_io.generate_forcefield(
        template, params, FFtype="REAXFF", outfile=str(out_path), MD="LAMMPS")
    log.info("ezff-render: wrote %s (%d substituted parameters)",
             out_path, len(params))
    return 0


def _run_ezff_eval(args: argparse.Namespace,
                   log: logging.Logger) -> int:
    bounds = ezff_io.read_variable_bounds(str(args.ezff_bounds))
    template = ezff_io.read_forcefield_template(str(args.ezff_template))
    params = _ezff_resolve_params(args.ezff_params, bounds)

    dataset_root = (Path(args.ezff_dataset).resolve()
                    if args.ezff_dataset
                    else repo_root() / "vasp_calculations")
    if not dataset_root.is_dir():
        log.error("ezff-eval: dataset_root %s does not exist", dataset_root)
        return 2

    # ezff_error.error_function needs LAMMPS available.
    ensure_lammps(auto_install=args.install_lammps,
                  channel=args.conda_channel)

    errs = ezff_error.error_function(
        params,
        template_string=template,
        dataset_root=str(dataset_root),
        subset=int(args.ezff_subset or 0),
        verbose=bool(args.verbose),
    )
    e_err, f_err, geom_err = errs

    n_frames = ("all" if not args.ezff_subset
                else f"first {args.ezff_subset}")
    log.info("ezff-eval (dataset=%s, frames=%s):", dataset_root, n_frames)
    log.info("  E_RMSE  (weighted, eV)   = %s", _fmt_err(e_err))
    log.info("  F_RMSE  (eV/A)           = %s", _fmt_err(f_err))
    log.info("  geom_RMSE (lattice, A)   = %s", _fmt_err(geom_err))
    # Machine-readable line on stdout for piping/scripting.
    print(json.dumps({"E_err": e_err, "F_err": f_err, "geom_err": geom_err}))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="PIBO-ReaxFF  -  Physics-Informed Bayesian Optimization "
                    "of ReaxFF parameters, with optional MoS2 physical "
                    "validation gate.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=_HERE / "configs" / "run_default.yaml",
        help="Path to the YAML run configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--base-config", type=Path,
        default=_HERE / "configs" / "default_mosh.json",
        help="Path to the BenchmarkSuite base JSON config "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--profile", choices=("demo", "manuscript"),
        default=None,
        help="Override the YAML's optimization profile.",
    )
    parser.add_argument("--budget", "--iterations", dest="budget",
                        type=int, default=None,
                        help="Override bo.budget (total BO iterations per "
                             "optimizer/replicate). Aliased as --iterations.")
    parser.add_argument("--replicates", type=int, default=None,
                        help="Override bo.replicates.")
    parser.add_argument("--init-lhs", type=int, default=None,
                        help="Override bo.init_lhs_points (LHS warm-start size).")
    parser.add_argument(
        "--optimizers", nargs="+", default=None,
        help="Override bo.optimizers (e.g. --optimizers pibo pso).",
    )

    # Loss-weight overrides (composite loss). 'cw-' = per-category weight.
    parser.add_argument("--w-energy",   type=float, default=None,
                        help="Override loss_weights.energy.")
    parser.add_argument("--w-forces",   type=float, default=None,
                        help="Override loss_weights.forces.")
    parser.add_argument("--w-geometry", type=float, default=None,
                        help="Override loss_weights.geometry.")
    parser.add_argument("--cw-equilibrium",  type=float, default=None,
                        help="Override category_weights.equilibrium.")
    parser.add_argument("--cw-bond-pes",     type=float, default=None,
                        help="Override category_weights.bond_pes.")
    parser.add_argument("--cw-angle-pes",    type=float, default=None,
                        help="Override category_weights.angle_pes.")
    parser.add_argument("--cw-dihedral-pes", type=float, default=None,
                        help="Override category_weights.dihedral_pes.")
    parser.add_argument("--cw-strained",     type=float, default=None,
                        help="Override category_weights.strained.")
    parser.add_argument("--cw-default",      type=float, default=None,
                        help="Override category_weights.default.")

    # Staged-acquisition range overrides (fractions of the BO budget).
    parser.add_argument("--stage-ucb-until", type=float, default=None,
                        help="Fraction of budget run with UCB acquisition "
                             "(e.g. 0.30). Sets acquisition.stage_ucb_until.")
    parser.add_argument("--stage-ei-until",  type=float, default=None,
                        help="Fraction of budget run with EI before switching "
                             "to PI (e.g. 0.70). Sets acquisition.stage_ei_until "
                             "and stage_pi_after.")
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Disable the physical-validation gate.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output.results_dir.",
    )
    # ---- EZFF-format adapter flags (short-circuit BO; format-only port) --
    parser.add_argument(
        "--ezff-eval", action="store_true",
        help="EZFF-shape sanity check: load --ezff-bounds and --ezff-template, "
             "compute [E_err, F_err, geom_err] over the VASP frames at "
             "--ezff-dataset (defaults to ./vasp_calculations), print the "
             "vector as JSON on stdout and exit. Requires LAMMPS.",
    )
    parser.add_argument(
        "--ezff-render", type=Path, default=None, metavar="PATH",
        help="Render a ffield to PATH using --ezff-bounds + --ezff-template + "
             "--ezff-params (default 'midpoint'), then exit. Does NOT need LAMMPS.",
    )
    parser.add_argument(
        "--ezff-bounds", type=Path,
        default=_ezff_default_bounds_path(),
        help="EZFF variable_bounds text file (default: %(default)s).",
    )
    parser.add_argument(
        "--ezff-template", type=Path,
        default=_ezff_default_template_path(),
        help="EZFF ffield template with <<NAME>> placeholders "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--ezff-params", default="midpoint",
        help="How --ezff-eval / --ezff-render resolve parameter values: "
             "'midpoint' (default), 'manuscript' (Table-2 posterior means "
             "where available, midpoint elsewhere), or a path to a JSON "
             "file {name: value} covering every bound.",
    )
    parser.add_argument(
        "--ezff-dataset", type=Path, default=None, metavar="DIR",
        help="VASP dataset root for --ezff-eval (default: ./vasp_calculations).",
    )
    parser.add_argument(
        "--ezff-subset", type=int, default=0, metavar="N",
        help="If > 0, --ezff-eval uses only the first N frames "
             "(round-robin across blocks). Useful for fast iteration.",
    )

    # ---- Staged BO (block-wise calibration) ----------------------------
    parser.add_argument(
        "--staged", action="store_true",
        help="Run staged BO instead of the unified BenchmarkSuite loop: "
             "bond block on bond_pes frames -> angle block on angle_pes -> "
             "offdiag on strained -> torsion on dihedral_pes -> joint "
             "(all 42 params on all frames). Each stage carries best "
             "values forward. Output: results/staged_run.json + "
             "results/posterior_parameters_staged.csv.",
    )

    # ---- Best-params export (post-BO) -----------------------------------
    parser.add_argument(
        "--export-best-ffield", type=Path, default=None, metavar="PATH",
        help="After BO completes, write the lowest-loss parameter set as a "
             "LAMMPS-format ffield to PATH (uses the same width-preserving "
             "substitution as the runner; byte-identical to in-BO ffields). "
             "A JSON sidecar with metrics + params lands next to PATH.",
    )
    parser.add_argument(
        "--export-best-only-pibo", action="store_true",
        help="When --export-best-ffield is set, restrict the best-record "
             "search to optimizer='pibo' (default: any optimizer wins).",
    )

    # ---- Convergence-on-divergence retry --------------------------------
    parser.add_argument(
        "--convergence-retries", type=int, default=0, metavar="N",
        help="If the BO run reports any non-finite metric (loss inf/NaN or "
             "validator worst==inf), re-run with the budget multiplied by "
             "--convergence-budget-multiplier, up to N additional attempts. "
             "Default 0 disables retries (single-pass behavior, identical "
             "to legacy pibo_reaxff).",
    )
    parser.add_argument(
        "--convergence-budget-multiplier", type=float, default=2.0,
        metavar="M",
        help="Budget scale factor per retry attempt (default 2.0).",
    )

    parser.add_argument(
        "--check-only", action="store_true",
        help="Run local_env.sanity_check and exit.",
    )
    parser.add_argument(
        "--install-lammps", action="store_true",
        help="If LAMMPS is missing or REAXFF-less, attempt to install it "
             "via `conda install -c conda-forge lammps` into the active env, "
             "then continue.",
    )
    parser.add_argument(
        "--conda-channel", default="conda-forge",
        help="Channel for --install-lammps (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Set log level to DEBUG.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.check_only:
        ok = sanity_check()
        if not ok and args.install_lammps:
            print("[run] --install-lammps set; attempting conda install ...")
            try:
                from pibo_reaxff.local_env import install_lammps_conda
                install_lammps_conda(channel=args.conda_channel)
                ok = sanity_check()
            except Exception as exc:
                print(f"[run] auto-install failed: {exc}", file=sys.stderr)
                return 1
        return 0 if ok else 1

    # EZFF-format short-circuits — these run independently of the BO config
    # and exit before BenchmarkSuite is constructed. Logging is bootstrapped
    # here because the regular path normally defers it until after the
    # YAML config has been read.
    if args.ezff_eval or args.ezff_render is not None:
        os.chdir(repo_root())
        _setup_logging("debug" if args.verbose else "info")
        log = logging.getLogger("run")
        if args.ezff_render is not None:
            return _run_ezff_render(args, log)
        return _run_ezff_eval(args, log)

    # Pin cwd to the repo root so relative paths inside the JSON config
    # (e.g. "../CODE/data/vasp_calculations", "lammps_templates/...") resolve
    # the same way regardless of where the PyCharm run configuration started.
    os.chdir(repo_root())

    # 1) Load + merge configs.
    yc = _load_yaml(args.config)
    base = _load_base_json(args.base_config)

    _setup_logging(yc.get("output", {}).get("log_level",
                                            "debug" if args.verbose else "info"))
    log = logging.getLogger("run")

    profile = args.profile or yc.get("profile", "demo")
    log.info("profile = %s", profile)

    cfg = _merge_yaml_into_base(base, yc, profile)

    # CLI overrides on top.
    if args.budget is not None:
        cfg["optimization"][profile]["budget"] = args.budget
    if args.replicates is not None:
        cfg["optimization"][profile]["replicates"] = args.replicates
    if args.init_lhs is not None:
        cfg["optimization"][profile]["init_lhs_points"] = args.init_lhs

    # Loss-weight overrides.
    cfg.setdefault("loss_weights", {})
    if args.w_energy is not None:
        cfg["loss_weights"]["energy"] = args.w_energy
    if args.w_forces is not None:
        cfg["loss_weights"]["forces"] = args.w_forces
    if args.w_geometry is not None:
        cfg["loss_weights"]["geometry"] = args.w_geometry
    cat_overrides = {
        "equilibrium":  args.cw_equilibrium,
        "bond_pes":     args.cw_bond_pes,
        "angle_pes":    args.cw_angle_pes,
        "dihedral_pes": args.cw_dihedral_pes,
        "strained":     args.cw_strained,
        "default":      args.cw_default,
    }
    cw = cfg["loss_weights"].setdefault("category_weights", {})
    for k, v in cat_overrides.items():
        if v is not None:
            cw[k] = v

    # Staged-acquisition overrides.
    cfg.setdefault("acquisition", {})
    if args.stage_ucb_until is not None:
        cfg["acquisition"]["stage_ucb_until"] = args.stage_ucb_until
    if args.stage_ei_until is not None:
        cfg["acquisition"]["stage_ei_until"] = args.stage_ei_until
        cfg["acquisition"]["stage_pi_after"] = args.stage_ei_until

    # 2) LAMMPS availability. Mandatory — raises if missing unless
    #    --install-lammps was passed, in which case we install via conda first.
    lmp = ensure_lammps(
        auto_install=args.install_lammps,
        channel=args.conda_channel,
    )
    log.info("LAMMPS: %s", lmp)

    # 3) Build the validator (PIBO-only gate).
    val_cfg = yc.get("validation", {})
    if args.no_validation:
        val_cfg = dict(val_cfg)
        val_cfg["enabled"] = False
    validator = _build_validator(val_cfg, log)

    # 4) Construct the suite and run.
    from pibo_reaxff.benchmark import BenchmarkSuite
    suite = BenchmarkSuite(
        system=yc.get("system", "MoSH"),
        config=cfg,
        profile=profile,
        lammps_mode="lammps",
        validator=validator,
        validate_every_k=int(val_cfg.get("every_k", 0)),
        val_boost=float(val_cfg.get("boost_factor", 2.0)),
        val_shrink=float(val_cfg.get("shrink_factor", 0.7)),
        val_early_stop=bool(val_cfg.get("early_stop", False)),
    )

    optimizers = args.optimizers or yc.get("bo", {}).get("optimizers")
    ablation = bool(yc.get("bo", {}).get("ablation_physics_off", True))

    # ---- Staged BO short-circuit -----------------------------------------
    # When --staged is set, replace the single BenchmarkSuite loop with the
    # block-wise stage sequence in pibo_reaxff/staged_bo.py. The output
    # CSVs/figures still get written via the same posterior_report path,
    # but the BO is replaced by the multi-stage orchestrator.
    if args.staged:
        from pibo_reaxff.staged_bo import StagedBORunner, DEFAULT_STAGES
        from pibo_reaxff.parameters import REAXFF_PARAMETERS
        from pibo_reaxff.optimizers.base import OptimizerResult

        log.info("Staged BO enabled (%d stages)", len(DEFAULT_STAGES))
        runner = StagedBORunner(
            runner=suite.runner,
            specs=REAXFF_PARAMETERS,
            train_frames=suite.train_frames,
            val_frames=suite.val_frames,
            weights=suite.weights,
            optimizer_config={
                "gp":          cfg.get("gp", {}),
                "acquisition": cfg.get("acquisition", {}),
            },
            verbose=True,
        )
        staged = runner.run()
        log.info("Staged BO complete: final_loss=%.4f val=%.4f "
                 "total_evals=%d wall=%.1fs",
                 staged.final_loss, staged.final_val_loss,
                 staged.total_evals, staged.total_wall_s)
        for sr in staged.stages:
            log.info("  stage %-10s best_loss=%.4f  n_evals=%d  wall=%.1fs",
                     sr.stage_name, sr.best_loss, sr.n_evals, sr.wall_clock_s)

        # Persist a structured JSON of the staged run plus a single
        # synthetic RunRecord so the existing posterior/benchmark
        # reporting picks up the final result.
        results_dir = (args.output if args.output
                       else repo_root() / yc.get("output", {}).get(
                           "results_dir", "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        staged_json = {
            "final_loss": staged.final_loss,
            "final_val_loss": staged.final_val_loss,
            "total_evals": staged.total_evals,
            "total_wall_s": staged.total_wall_s,
            "stages": [
                {"name": s.stage_name,
                 "n_active": len(s.active_param_names),
                 "active_param_names": s.active_param_names,
                 "n_train_frames": s.n_train_frames,
                 "best_loss": s.best_loss,
                 "n_evals": s.n_evals,
                 "wall_clock_s": s.wall_clock_s,
                 "params_after_stage": s.params_after_stage}
                for s in staged.stages],
            "final_params": staged.final_params,
        }
        (results_dir / "staged_run.json").write_text(
            json.dumps(staged_json, indent=2, default=str),
            encoding="utf-8")
        log.info("Wrote %s", results_dir / "staged_run.json")

        # Per-stage ffield export: render the merged param dict at the end
        # of every stage as a complete LAMMPS reaxff ffield. Same width-
        # preserving substitution path as best_export.export_lammps_ffield.
        from pibo_reaxff.best_export import export_lammps_ffield
        ffield_dir = results_dir / "ffield"
        ffield_dir.mkdir(parents=True, exist_ok=True)
        template_path = (repo_root() / "lammps_templates" /
                         "ffield.reax.MoSH.NoteS1.template")
        for idx, sr in enumerate(staged.stages, start=1):
            if not sr.params_after_stage:
                continue
            out_path = ffield_dir / (
                f"ffield.reax.MoSH.stage{idx}_{sr.stage_name}.reax")
            export_lammps_ffield(sr.params_after_stage,
                                 str(template_path), str(out_path))
            log.info("Wrote %s  (after stage %d: %s, "
                     "stage_best_loss=%.4f)",
                     out_path, idx, sr.stage_name, sr.best_loss)
        final_ffield = ffield_dir / "ffield.reax.MoSH.calibrated.reax"
        export_lammps_ffield(staged.final_params, str(template_path),
                             str(final_ffield))
        log.info("Wrote %s  (final merged, final_loss=%.4f)",
                 final_ffield, staged.final_loss)

        # Inject the staged result as a RunRecord into the suite so
        # posterior_report / export-best-ffield see it.
        from pibo_reaxff.benchmark import RunRecord
        x_final = np.array([staged.final_params[s.name]
                            for s in REAXFF_PARAMETERS])
        synth = RunRecord(
            optimizer="pibo_staged",
            replicate=0,
            best_loss=float(staged.final_loss),
            loss_mae=float("nan"),
            loss_rmse=float(staged.final_loss),
            E_total_error=float("nan"),
            Ys_mean=float("nan"),
            Ys_std=float("nan"),
            n_evals=int(staged.total_evals),
            fevals_to_convergence=int(staged.total_evals),
            param_robustness=0.0,
            val_rmse=float(staged.final_val_loss),
            wall_clock_s=float(staged.total_wall_s),
            cpu_hours=float(staged.total_wall_s) / 3600.0,
            history=list(staged.history),
            best_x=x_final,
            extras={"all_X": np.array([x_final]),
                    "all_y": np.array([staged.final_loss]),
                    "staged_stages": [s.stage_name for s in staged.stages]},
            physics_informed=True,
        )
        suite.results = [synth]

        # Staged path is complete — synthesise a df + attempts_used so the
        # post-BO output block runs unchanged.
        df = suite.report()
        attempts_used = 1
        log.info("Staged Benchmark complete: %d rows", len(df))

    else:
        # Convergence-on-divergence wrapper.
        #
        # First attempt is the legacy single-pass call — identical to prior
        # pibo_reaxff behavior. Only if a metric comes back inf/NaN (or the
        # validator gate reports infinity) do we bump the budget and re-run.
        # Retries are *additional* attempts: --convergence-retries=0 keeps
        # exactly the legacy behavior.
        from pibo_reaxff.best_export import has_nonfinite_result

        base_budget = int(cfg["optimization"][profile].get("budget", 100))
        df = suite.run_all(optimizers=optimizers,
                           ablation_physics_off=ablation)
        log.info("Benchmark complete: %d rows", len(df))

        attempts_used = 1
        if args.convergence_retries > 0:
            for attempt in range(1, args.convergence_retries + 1):
                bad, reason = has_nonfinite_result(suite)
                if not bad:
                    break
                multiplier = float(args.convergence_budget_multiplier) ** attempt
                new_budget = max(base_budget + 1,
                                 int(round(base_budget * multiplier)))
                log.warning(
                    "convergence: non-finite result detected (%s) - "
                    "retry %d/%d with budget=%d (x%.2g of base=%d)",
                    reason, attempt, args.convergence_retries,
                    new_budget, multiplier, base_budget)
                cfg["optimization"][profile]["budget"] = new_budget
                suite = BenchmarkSuite(
                    system=yc.get("system", "MoSH"),
                    config=cfg, profile=profile, lammps_mode="lammps",
                    validator=validator,
                    validate_every_k=int(val_cfg.get("every_k", 0)),
                    val_boost=float(val_cfg.get("boost_factor", 2.0)),
                    val_shrink=float(val_cfg.get("shrink_factor", 0.7)),
                    val_early_stop=bool(val_cfg.get("early_stop", False)),
                )
                df = suite.run_all(optimizers=optimizers,
                                   ablation_physics_off=ablation)
                attempts_used = attempt + 1
                log.info("Benchmark complete (attempt %d): %d rows",
                         attempts_used, len(df))
            else:
                final_bad, final_reason = has_nonfinite_result(suite)
                if final_bad:
                    log.error(
                        "convergence: exhausted %d retries; still non-finite (%s).",
                        args.convergence_retries, final_reason)
        log.info("convergence: used %d attempt(s); final budget=%d",
                 attempts_used, cfg["optimization"][profile]["budget"])

    # 5) Outputs.
    out_cfg = yc.get("output", {})
    results_dir = (args.output if args.output
                   else repo_root() / out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if out_cfg.get("save_csv", True):
        csv_path = results_dir / "benchmark_per_run.csv"
        suite.to_csv(str(csv_path))
        log.info("Wrote %s", csv_path)
        agg_path = results_dir / "benchmark_aggregate.csv"
        suite.report_aggregate().to_csv(agg_path, index=False)
        log.info("Wrote %s", agg_path)

    if out_cfg.get("save_plots", True):
        plot_dir = repo_root() / out_cfg.get("plot_dir", "results/figures")
        plot_dir.mkdir(parents=True, exist_ok=True)
        try:
            figs = suite.plot_all(save_dir=str(plot_dir))
            log.info("Wrote %d figures to %s", len(figs), plot_dir)
        except Exception as exc:
            log.warning("plot_all failed (non-fatal): %s", exc)

    # Posterior-parameter export (one CSV + one figure per optimizer×rep).
    # Mirrors the Bayesian-OPT convention of reporting the posterior, not
    # just the MAP point, so the manuscript's per-parameter uncertainty
    # band is reproducible from the BO trace.
    try:
        from pibo_reaxff import posterior_report
        written = posterior_report.write_per_optimizer_reports(
            suite, str(results_dir), only_physics=False)
        for p in written["csv"]:
            log.info("Wrote %s", p)
        for p in written["png"]:
            log.info("Wrote %s", p)
        bench_fig = posterior_report.write_optimizer_benchmark_figure(
            suite, str(results_dir / "figures" / "optimizer_benchmark.png"))
        if bench_fig:
            log.info("Wrote %s", bench_fig)
    except Exception as exc:
        log.warning("posterior_report failed (non-fatal): %s", exc)

    # 6) Dump validation log if any PIBO result captured one.
    for r in getattr(suite, "results", []):
        if r.optimizer != "pibo":
            continue
        log_entries = (r.extras or {}).get("validation_log")
        if not log_entries:
            continue
        vlog_path = results_dir / f"validation_log_rep{r.replicate}.json"
        with vlog_path.open("w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, default=str)
        log.info("Wrote %s", vlog_path)

    # 7) Optionally export the best parameter set as a LAMMPS-format ffield.
    if args.export_best_ffield is not None:
        from pibo_reaxff import best_export
        opt_filter = "pibo" if args.export_best_only_pibo else None
        best = best_export.find_best_record(
            suite, optimizer=opt_filter, physics_informed=True)
        if best is None:
            # Relax filters: try ablation runs too.
            best = best_export.find_best_record(
                suite, optimizer=opt_filter, physics_informed=None)
        if best is None:
            log.error("export-best-ffield: no finite-loss record found; "
                      "skipping export.")
        else:
            names = best_export.parameter_names(suite)
            params = best_export.best_params_dict(suite, best)
            template_path = args.ezff_template
            out_ffield = Path(args.export_best_ffield).resolve()
            best_export.export_lammps_ffield(
                params, str(template_path), str(out_ffield))
            sidecar = out_ffield.with_suffix(out_ffield.suffix + ".json")
            best_export.write_best_summary(
                best, names, str(sidecar),
                extra={"template": str(template_path),
                       "attempts_used": attempts_used,
                       "final_budget":
                           int(cfg["optimization"][profile]["budget"])})
            log.info("export-best-ffield: wrote %s  (loss=%.6f, "
                     "optimizer=%s, rep=%d, phys=%s)",
                     out_ffield, best.best_loss, best.optimizer,
                     best.replicate, best.physics_informed)
            log.info("export-best-ffield: sidecar -> %s", sidecar)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())