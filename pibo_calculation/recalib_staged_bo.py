"""PIBO calibration of MoS2 ReaxFF — staged Bayesian optimization (manuscript Eq. 1).

Calibration objective (Eq. 1): the category-weighted ReaxFF-vs-DFT energy
residual over the bond / angle / torsion / non-bonded DFT scans in
data/dft_reference/, evaluated via optimizer_error (Table 1 weights). The
held-out properties (uniaxial/biaxial stress-strain, S-monovacancy) are NOT
part of this objective; run them as validation with --validate.

Schedule (N=100): warm-start(1) -> LHS(14) -> EI with Thompson p=0.15 (45)
-> LCB beta=1.5 (40). GP posterior saved every 10 trials on a 200-point grid.
Warm start x0 = the Note S1 / Cooper-derived prior centre per parameter.

Run:
    python recalib_staged_bo.py --budget 100 --seed 42        # calibrate
    python recalib_staged_bo.py --validate                     # held-out check

Needs Python >=3.11 (numpy, pandas, scikit-optimize, scikit-learn) and a
ReaxFF-enabled LAMMPS (set $PIBO_LMP or put `lmp` on PATH).
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
OUT_DIR = HERE / "output"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

OPTIMIZER_BOUNDS   = DATA / "optimizer_variable_bounds.txt"
OPTIMIZER_TEMPLATE = REPO / "lammps_templates" / "ffield.reax.MoSH.template"
PES_DATA           = DATA / "dft_reference"
DFT_CSV            = DATA / "MoS2_physical_validation.csv"   # held-out validation only

DST_FFIELD = OUT_DIR / "ffield.reax.MoSH.pibo_calibrated.reax"
LOG_PATH   = OUT_DIR / "RECALIB_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
POST_DIR   = OUT_DIR / "posterior_snapshots"

# Staged schedule (manuscript Table 2)
STAGE1_END  = 15      # warm-start (trial 1) + LHS (trials 2..15 = 14)
STAGE2_END  = 60      # EI + Thompson (trials 16..60 = 45)
P_THOMPSON  = 0.15
LCB_BETA    = 1.5     # trials 61..100 = 40
N_POST_DUMP = 10
GRID_M      = 200


def warm_start_x0(names, bounds):
    """x0 = Note S1 / Cooper-derived prior centre (ParameterSpec.notes1_value);
    midpoint of the bounds for any name not found."""
    from pibo_reaxff.parameters import REAXFF_PARAMETERS
    by = {p.name: p for p in REAXFF_PARAMETERS}
    x0 = []
    for n in names:
        sp = by.get(n)
        if sp is not None:
            x0.append(float(sp.notes1_value))
        else:
            lo, hi = bounds[n][0], bounds[n][1]
            x0.append(0.5 * (float(lo) + float(hi)))
    return x0


def thompson_sample_candidate(bo, n_candidates=512, rng=None):
    """EI-stage Thompson injection: draw one GP realization per candidate, take argmin."""
    if rng is None: rng = np.random.default_rng()
    if not bo.models:
        return bo.ask(), {}
    gp = bo.models[-1]
    cands = bo.space.rvs(n_candidates, random_state=int(rng.integers(0, 2**31 - 1)))
    cands_t = bo.space.transform(cands)
    try:
        mu, sigma = gp.predict(cands_t, return_std=True)
        samples = rng.normal(loc=mu, scale=np.maximum(sigma, 1e-6))
        idx = int(np.argmin(samples))
        return list(cands[idx]), {"mu": float(mu[idx]), "sigma": float(sigma[idx]),
                                  "sample": float(samples[idx])}
    except Exception as e:
        print(f"  [warn] Thompson failed: {e}")
        return bo.ask(), {}


def lcb_candidate(bo, beta=LCB_BETA, n_candidates=512, rng=None):
    """LCB acquisition: minimize mu(x) - beta * sigma(x)."""
    if rng is None: rng = np.random.default_rng()
    if not bo.models:
        return bo.ask(), {}
    gp = bo.models[-1]
    cands = bo.space.rvs(n_candidates, random_state=int(rng.integers(0, 2**31 - 1)))
    cands_t = bo.space.transform(cands)
    try:
        mu, sigma = gp.predict(cands_t, return_std=True)
        idx = int(np.argmin(mu - beta * sigma))
        return list(cands[idx]), {"mu": float(mu[idx]), "sigma": float(sigma[idx])}
    except Exception:
        return bo.ask(), {}


def snapshot_posterior(bo, names, trial, out_dir, n_grid=GRID_M):
    """Save GP predictive (mean, std) on an n_grid LHS grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bo.models:
        return
    gp = bo.models[-1]
    cands = bo.space.rvs(n_grid, random_state=trial)
    try:
        mu, sigma = gp.predict(bo.space.transform(cands), return_std=True)
    except Exception as e:
        print(f"  [warn] posterior snapshot failed: {e}")
        return
    out = out_dir / f"posterior_trial{trial:04d}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_idx", "gp_mean", "gp_std"] + names)
        for i, (m, s, pt) in enumerate(zip(mu, sigma, cands)):
            w.writerow([i, f"{m:.6f}", f"{s:.6f}", *[f"{v:.6f}" for v in pt]])
    print(f"  [posterior] -> {out.name} ({n_grid} grid points)", flush=True)


def calibrate(budget, seed):
    """Staged BO minimizing the DFT energy residual (Eq. 1)."""
    from skopt import Optimizer
    from skopt.space import Real
    import optimizer_error, optimizer_io

    bounds = optimizer_io.read_variable_bounds(str(OPTIMIZER_BOUNDS))
    template = optimizer_io.read_forcefield_template(str(OPTIMIZER_TEMPLATE))
    names = list(bounds)
    lows = np.array([bounds[n][0] for n in names], float)
    highs = np.array([bounds[n][1] for n in names], float)
    x0 = warm_start_x0(names, bounds)
    print(f"[setup] params={len(names)}, budget={budget}, warm-start=Note S1 prior centre", flush=True)
    print(f"[setup] schedule: warm(1) -> LHS({STAGE1_END-1}) -> "
          f"EI+Thompson p={P_THOMPSON} ({STAGE2_END-STAGE1_END}) -> LCB beta={LCB_BETA} "
          f"({budget-STAGE2_END})", flush=True)

    space = [Real(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
    bo = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI",
                   n_initial_points=STAGE1_END, initial_point_generator="lhs",
                   random_state=seed)
    rng = np.random.default_rng(seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POST_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["trial", "wall_s", "loss_eV", "E_err_eV", "F_err", "geom_err",
            "acq_type", "acq_mu", "acq_sigma"] + names
    LOG_PATH.write_text(",".join(cols) + "\n", encoding="utf-8")
    best = {"loss": float("inf"), "x": None, "trial": -1, "acq": "?"}
    t0 = time.time()

    for trial in range(1, budget + 1):
        acq_info = {}
        if trial == 1:
            x_raw, acq = list(x0), "warm_start"
        elif trial <= STAGE1_END:
            x_raw, acq = bo.ask(), "LHS"
        elif trial <= STAGE2_END:
            if rng.random() < P_THOMPSON:
                x_raw, acq_info = thompson_sample_candidate(bo, rng=rng); acq = "Thompson"
            else:
                x_raw, acq = bo.ask(), "EI"
        else:
            x_raw, acq_info = lcb_candidate(bo, beta=LCB_BETA, rng=rng); acq = "LCB"

        x = np.minimum(np.maximum(np.asarray(x_raw, float), lows), highs)
        e_err, f_err, geom_err = optimizer_error.error_function(
            dict(zip(names, x)), template_string=template, dataset_root=str(PES_DATA))
        loss = e_err if e_err == e_err else 1e9   # NaN guard
        try:
            bo.tell(list(x), float(loss))
        except Exception as e:
            print(f"  [warn] tell error: {e}")

        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(",".join([
                str(trial), f"{time.time()-t0:.2f}", f"{loss:.6f}",
                f"{e_err:.6f}", f"{f_err:.6f}", f"{geom_err:.6f}", acq,
                f"{acq_info.get('mu', float('nan')):.6f}" if acq_info else "",
                f"{acq_info.get('sigma', float('nan')):.6f}" if acq_info else "",
                *[f"{v:.6f}" for v in x]]) + "\n")

        improved = loss < best["loss"] - 1e-9
        if improved:
            best.update({"loss": loss, "x": list(x), "trial": trial, "acq": acq})
            optimizer_io.generate_forcefield(template, dict(zip(names, x)),
                                             FFtype="REAXFF", outfile=str(DST_FFIELD),
                                             MD="LAMMPS")
            BEST_PATH.write_text(
                f"trial {trial}  acq={acq}  loss(E_RMSE)={loss:.6f} eV\n"
                f"params={dict(zip(names, [float(v) for v in x]))}\n",
                encoding="utf-8")
        print(f"t{trial:4d} {acq:9s} E_err={e_err:.4f} eV  best={best['loss']:.4f}"
              f"{'  [NEW BEST]' if improved else ''}", flush=True)

        if trial % N_POST_DUMP == 0:
            snapshot_posterior(bo, names, trial, POST_DIR)

    snapshot_posterior(bo, names, 9999, POST_DIR)
    print(f"\n[done] {budget} trials, {(time.time()-t0)/60:.1f} min, "
          f"best E_RMSE={best['loss']:.4f} eV at trial {best['trial']} ({best['acq']})\n"
          f"[done] calibrated ffield -> {DST_FFIELD.name}", flush=True)
    return DST_FFIELD


def validate(ffield):
    """Held-out validation (NOT in the calibration objective): evaluate the
    calibrated ffield against the uniaxial/biaxial stress-strain + S-monovacancy
    references in MoS2_physical_validation.csv. Requires the 10x10 LAMMPS deck."""
    import pandas as pd
    import recalib_combined_all as rca
    rca.OUT_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(ffield).exists():
        raise FileNotFoundError(f"{ffield} not found — run calibration first.")
    src = rca.read_lines(Path(ffield))
    spec = rca.build_spec(src, rca.parse_offsets(src))
    x = np.array([s["init"] for s in spec])
    SUB = [0.05, 0.10, 0.15, 0.20, 0.25]
    dft = pd.read_csv(DFT_CSV)
    bx = dft[(dft.category == "BIAXIAL") & (dft.condition_var == "true_strain_biaxial")]
    bx = bx.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    refs = {"biax_eps": np.asarray(SUB),
            "biax_sig": np.interp(SUB, bx.condition_value.values, bx.stress_GPa.values)}
    for cat, var, key in [("UNIAXIAL_X1", "true_strain_zigzag", "x1"),
                          ("UNIAXIAL_X2", "true_strain_armchair", "x2")]:
        d = dft[(dft.category == cat) & (dft.condition_var == var)].dropna(subset=["stress_GPa"])
        refs[f"{key}_eps"] = np.asarray(SUB)
        refs[f"{key}_sig"] = np.interp(SUB, d.condition_value.values, d.stress_GPa.values)
        refs[f"{key}_hS"] = np.interp(SUB, d.condition_value.values, d.h_S_AA.values)
    r = rca.evaluate(spec, x, src, refs, OUT_DIR / "_validation")
    print("[validate] held-out relative errors (%):", flush=True)
    for k in ("biax_max", "x1_sig_max", "x2_sig_max", "V_S_re", "diff_path2_re"):
        print(f"  {k:14s} {r.get(k, float('nan')):.2f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", action="store_true",
                    help="Run held-out validation on the calibrated ffield instead of calibrating.")
    args = ap.parse_args()
    if args.validate:
        validate(DST_FFIELD)
    else:
        calibrate(args.budget, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
