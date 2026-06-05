"""PIBO recalibration of MoS2 ReaxFF (two stages, one driver).

Stage 1  PES fit   : Bayesian optimization of the ReaxFF parameters against the
                     bond / angle / torsion / non-bonded DFT scans in
                     data/dft_reference/, via the optimizer's PES error
                     function (optimizer_error).
Stage 2  staged BO : LHS -> EI+Thompson -> LCB over the biaxial / uniaxial
                     stress-strain (+ V_S, S-diffusion) targets in
                     data/MoS2_physical_validation.csv, evaluated with LAMMPS.

Run:
    python recalib_staged_bo.py --budget 100 --pes-budget 100 --seed 42

Needs Python >=3.11 (numpy, pandas, scipy, scikit-optimize, scikit-learn) and
a ReaxFF-enabled LAMMPS (set $PIBO_LMP or put `lmp` on PATH).
"""
from __future__ import annotations
import argparse, csv, json, math, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
OUT_DIR = HERE / "output"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

DFT_CSV       = DATA / "MoS2_physical_validation.csv"
PES_DATA      = DATA / "dft_reference"
OPTIMIZER_BOUNDS   = DATA / "optimizer_variable_bounds.txt"
OPTIMIZER_TEMPLATE = REPO / "lammps_templates" / "ffield.reax.MoSH.template"
WARMSTART     = DATA / "ffield.reax.MoSH.pibo_biaxial_v9.reax"  # optional (Supporting Information)
DATA10x10     = DATA / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"

from _lmp_path import find_lmp as _find_lmp
LMP = _find_lmp()

PES_FFIELD = OUT_DIR / "ffield.reax.MoSH.pes_fit.reax"
DST_FFIELD = OUT_DIR / "ffield.reax.MoSH.staged_bo.reax"
LOG_PATH   = OUT_DIR / "RECALIB_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
POST_DIR   = OUT_DIR / "posterior_snapshots"

GPA_PER_ATM = 1.01325e-4
H_EFF_AA    = 6.145
ELEMENTS    = ("Mo", "S")
KCAL_TO_EV  = 1.0 / 23.0605
LMP_CONTROL = (
    "tabulate_long_range 10000\nnbrhood_cutoff 5.0\nhbond_cutoff 6.0\n"
    "bond_graph_cutoff 0.3\nthb_cutoff 0.001\nthb_cutoff_sq 0.00001\nwrite_freq 0\n"
)
_NUM = r"[+-]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"

# Staged BO parameters
STAGE1_END  = 15      # LHS up to this trial
STAGE2_END  = 60      # EI + Thompson up to this trial
P_THOMPSON  = 0.15    # probability of Thompson sample in stage 2
N_POST_DUMP = 10      # save GP posterior snapshot every N trials

TARGET_PCT = 10.0
MU_S_REF   = 2.82
VFE_REF  = {"V_S": 2.35}
DIFF_REF = {"path2": 1.35}

# Reuse the WIDEN bounds, spec builders, decks and LAMMPS evaluator from
# recalib_combined_all (same folder). OUT_DIR is monkey-patched so the
# evaluator writes under this bundle's output/.
sys.path.insert(0, str(HERE))
import recalib_combined_all as _rca
# Monkey-patch the OUT_DIR so evaluate writes to OUR results dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
_rca.OUT_DIR = OUT_DIR
from recalib_combined_all import (
    grab, read_lines, write_lines, parse_offsets, build_spec, write_ffield,
    setup_wd, run_lmp, biax_init_deck, biax_strain_deck, uni_deck, defect_deck,
    saddle_deck, measure_h_S, parse_data_atoms, write_data_with_atoms,
    parse_atom_record, rel_err, evaluate,
)


def pes_fit(budget, seed):
    """Stage 1: BO of the ReaxFF parameters vs the bond/angle/torsion/non-bonded
    DFT scans (optimizer PES energy error). Writes the fitted ffield to
    PES_FFIELD."""
    from skopt import Optimizer
    from skopt.space import Real
    import optimizer_error, optimizer_io
    bounds = optimizer_io.read_variable_bounds(str(OPTIMIZER_BOUNDS))
    template = optimizer_io.read_forcefield_template(str(OPTIMIZER_TEMPLATE))
    names = list(bounds)
    space = [Real(float(bounds[n][0]), float(bounds[n][1])) for n in names]
    bo = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI",
                   n_initial_points=max(10, len(names)),
                   initial_point_generator="lhs", random_state=seed)
    best = (float("inf"), None)
    for t in range(1, budget + 1):
        x = bo.ask()
        e_err, f_err, geom_err = optimizer_error.error_function(
            dict(zip(names, x)), template_string=template,
            dataset_root=str(PES_DATA))
        loss = e_err if e_err == e_err else 1e9
        bo.tell(x, float(loss))
        if loss < best[0]:
            best = (loss, list(x))
            optimizer_io.generate_forcefield(
                template, dict(zip(names, best[1])), FFtype="REAXFF",
                outfile=str(PES_FFIELD), MD="LAMMPS")
        print(f"[pes {t:4d}] E_err={e_err:.4f} eV  best={best[0]:.4f}", flush=True)
    print(f"[pes] fitted ffield -> {PES_FFIELD.name}  (E_err={best[0]:.4f} eV)", flush=True)
    return PES_FFIELD


def thompson_sample_candidate(bo, n_candidates=512, rng=None):
    """Draw n_candidates LHS points, evaluate GP posterior, sample one realization
    from each candidate, pick argmin.

    With skopt's Optimizer, the underlying GP is at bo.models[-1] (after at
    least 2 tells). We sample y_pred ~ N(mu, sigma^2) at each candidate then
    pick min.
    """
    if rng is None: rng = np.random.default_rng()
    if not bo.models:
        return bo.ask()  # fall back
    gp = bo.models[-1]
    # Sample candidates uniformly within normalized bounds (skopt internal)
    d = len(bo.space.dimensions)
    # Use space.rvs for valid candidates (handles Real bounds + transforms)
    cands = bo.space.rvs(n_candidates, random_state=int(rng.integers(0, 2**31-1)))
    cands_t = bo.space.transform(cands)  # normalize for GP
    try:
        mu, sigma = gp.predict(cands_t, return_std=True)
        # Thompson: draw one sample from N(mu, sigma) at each candidate
        samples = rng.normal(loc=mu, scale=np.maximum(sigma, 1e-6))
        idx = int(np.argmin(samples))
        return list(cands[idx]), {"mu": float(mu[idx]), "sigma": float(sigma[idx]),
                                   "sample": float(samples[idx])}
    except Exception as e:
        print(f"  [warn] Thompson failed: {e}; falling back to ask()")
        return bo.ask(), {}


def lcb_candidate(bo, beta=2.0, n_candidates=512, rng=None):
    """Lower confidence bound acquisition: minimize mu - beta * sigma."""
    if rng is None: rng = np.random.default_rng()
    if not bo.models:
        return bo.ask(), {}
    gp = bo.models[-1]
    cands = bo.space.rvs(n_candidates, random_state=int(rng.integers(0, 2**31-1)))
    cands_t = bo.space.transform(cands)
    try:
        mu, sigma = gp.predict(cands_t, return_std=True)
        scores = mu - beta * sigma
        idx = int(np.argmin(scores))
        return list(cands[idx]), {"mu": float(mu[idx]), "sigma": float(sigma[idx]),
                                   "lcb": float(scores[idx])}
    except Exception:
        return bo.ask(), {}


def snapshot_posterior(bo, names, lows, highs, trial, out_dir, n_grid=200):
    """Sample GP posterior at LHS grid points and save to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bo.models:
        return
    gp = bo.models[-1]
    cands = bo.space.rvs(n_grid, random_state=trial)
    cands_t = bo.space.transform(cands)
    try:
        mu, sigma = gp.predict(cands_t, return_std=True)
    except Exception as e:
        print(f"  [warn] posterior snapshot failed: {e}")
        return
    out = out_dir / f"posterior_trial{trial:04d}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_idx", "gp_mean", "gp_std"] + names)
        for i, (m, s, pt) in enumerate(zip(mu, sigma, cands)):
            w.writerow([i, f"{m:.6f}", f"{s:.6f}",
                        *[f"{v:.6f}" for v in pt]])
    print(f"  [posterior] -> {out.name}  ({n_grid} grid points)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--pes-budget", type=int, default=100)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POST_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: PES fit against the bond/angle/torsion/non-bonded DFT scans.
    pes_ffield = pes_fit(args.pes_budget, args.seed)

    # Stage 2 warm-start: the biaxial-tuned v9 (Supporting Information) if
    # present in data/, otherwise the Stage-1 PES fit.
    src_path = WARMSTART if WARMSTART.exists() else pes_ffield
    src = read_lines(src_path)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    lows  = np.array([s["lo"] for s in spec])
    highs = np.array([s["hi"] for s in spec])
    x0    = np.array([s["init"] for s in spec])
    names = [s["name"] for s in spec]
    print(f"[setup] stage2 warm-start={src_path.name}, params={len(spec)}, budget={args.budget}", flush=True)
    print(f"[setup] staged BO: LHS<={STAGE1_END}, EI/Thompson<={STAGE2_END}, LCB after", flush=True)

    SUB = [0.05, 0.10, 0.15, 0.20, 0.25]
    dft = pd.read_csv(DFT_CSV)
    bx = dft[(dft.category=="BIAXIAL") & (dft.condition_var=="true_strain_biaxial")]
    bx = bx.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    refs = {"biax_eps": np.asarray(SUB),
            "biax_sig": np.interp(SUB, bx.condition_value.values, bx.stress_GPa.values)}
    x1 = dft[(dft.category=="UNIAXIAL_X1") & (dft.condition_var=="true_strain_zigzag")]
    x1 = x1.dropna(subset=["stress_GPa"]).reset_index(drop=True)
    refs["x1_eps"] = np.asarray(SUB)
    refs["x1_sig"] = np.interp(SUB, x1.condition_value.values, x1.stress_GPa.values)
    refs["x1_hS"]  = np.interp(SUB, x1.condition_value.values, x1.h_S_AA.values)
    x2 = dft[(dft.category=="UNIAXIAL_X2") & (dft.condition_var=="true_strain_armchair")]
    x2 = x2.dropna(subset=["stress_GPa"]).reset_index(drop=True)
    refs["x2_eps"] = np.asarray(SUB)
    refs["x2_sig"] = np.interp(SUB, x2.condition_value.values, x2.stress_GPa.values)
    refs["x2_hS"]  = np.interp(SUB, x2.condition_value.values, x2.h_S_AA.values)

    from skopt import Optimizer
    from skopt.space import Real
    space = [Real(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
    bo = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI",
                   n_initial_points=STAGE1_END,
                   initial_point_generator="lhs",
                   random_state=args.seed)

    rng = np.random.default_rng(args.seed)
    metric_keys = ["biax_max","x1_sig_max","x2_sig_max","x1_hS_max","x2_hS_max",
                   "V_S_re","diff_path2_re"]
    cols = (["trial","wall_s","worst","acq_type","acq_mu","acq_sigma","acq_extra"]
            + metric_keys + names)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(",".join(cols) + "\n", encoding="utf-8")
    best = {"worst": float("inf"), "x": None, "comps": None, "trial": -1}
    t_start = time.time()
    work_root = OUT_DIR / "_runs"
    work_root.mkdir(parents=True, exist_ok=True)

    for trial in range(1, args.budget + 1):
        # Choose acquisition by stage
        acq_info = {}
        acq_type = "?"
        if trial == 1:
            x_raw = list(x0); acq_type = "warm_start"
        elif trial <= STAGE1_END:
            x_raw = bo.ask(); acq_type = "LHS"
        elif trial <= STAGE2_END:
            if rng.random() < P_THOMPSON:
                x_raw, acq_info = thompson_sample_candidate(bo, rng=rng)
                acq_type = "Thompson"
            else:
                x_raw = bo.ask(); acq_type = "EI"
        else:
            # Stage 3: LCB exploitation
            x_raw, acq_info = lcb_candidate(bo, beta=1.5, rng=rng)
            acq_type = "LCB"

        x = np.minimum(np.maximum(np.asarray(x_raw), lows), highs)
        t0 = time.time()
        r = evaluate(spec, x, src, refs, work_root / f"t{trial:04d}")
        dt = time.time() - t0
        # Recompute worst per current goal: 5 priority only
        r["worst"] = max(r["biax_max"], r["x1_sig_max"], r["x2_sig_max"],
                         r["V_S_re"], r["diff_path2_re"])
        worst = r["worst"]
        try: bo.tell(list(x), float(worst))
        except Exception as e: print(f"  [warn] tell error: {e}")

        # Log row
        acq_extra_str = json.dumps(acq_info, separators=(",",":")) if acq_info else ""
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            row = [str(trial), f"{time.time()-t_start:.2f}", f"{worst:.4f}",
                   acq_type,
                   f"{acq_info.get('mu', float('nan')):.4f}" if acq_info else "",
                   f"{acq_info.get('sigma', float('nan')):.4f}" if acq_info else "",
                   f'"{acq_extra_str}"',
                   *[f"{r[k]:.4f}" for k in metric_keys],
                   *[f"{v:.6f}" for v in x]]
            fh.write(",".join(row) + "\n")

        improved = worst < best["worst"] - 1e-6
        if improved:
            best.update({"worst": worst, "x": x.tolist(), "comps": r, "trial": trial,
                         "acq_type": acq_type})
            write_ffield(src, spec, x, DST_FFIELD)
            BEST_PATH.write_text(
                f"trial {trial}  worst={worst:.4f}  acq={acq_type}\n"
                f"biax={r['biax_max']:.4f}  x1_sig={r['x1_sig_max']:.4f}  "
                f"x2_sig={r['x2_sig_max']:.4f}  V_S={r['V_S_re']:.4f}  "
                f"diff_path2={r['diff_path2_re']:.4f}\n"
                f"(h_S tracked: x1={r['x1_hS_max']:.4f}, x2={r['x2_hS_max']:.4f})\n"
                f"params={dict(zip(names, x.tolist()))}\n",
                encoding="utf-8")
        mark = "  [NEW BEST]" if improved else ""
        print(f"t{trial:4d} [{dt:5.0f}s] {acq_type:9s} w={worst:6.2f} "
              f"(bx={r['biax_max']:5.1f} 1s={r['x1_sig_max']:5.1f} 2s={r['x2_sig_max']:5.1f} "
              f"Vs={r['V_S_re']:5.1f} D2={r['diff_path2_re']:5.1f})  best={best['worst']:6.2f}{mark}",
              flush=True)

        # Posterior snapshot
        if trial % N_POST_DUMP == 0:
            snapshot_posterior(bo, names, lows, highs, trial, POST_DIR)

        if best["worst"] <= TARGET_PCT:
            print(f"\n[TARGET MET at trial {best['trial']} via {best['acq_type']}]", flush=True)
            break

    elapsed = time.time() - t_start
    print(f"\n[done] {trial} trials, {elapsed/60:.1f} min, best worst={best['worst']:.2f}% at trial {best['trial']} ({best.get('acq_type','?')})")
    # Save final posterior
    snapshot_posterior(bo, names, lows, highs, 9999, POST_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
