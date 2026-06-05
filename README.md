# PIBO-ReaxFF (MoS₂) — reproducibility repository

Physics-Informed Bayesian Optimization (**PIBO**) for ReaxFF parameter
calibration of MoS₂, together with the uncertainty-quantification (UQ)
analysis reported in the manuscript's reviewer response.

This repository is **reproducibility-oriented**: it contains the PIBO
optimization framework and the UQ analysis pipeline so the figures and
tables in the paper can be regenerated from the released artifacts. It is
**not** a turnkey environment for re-running the LAMMPS validation suite
(stress–strain, vacancy formation, S-diffusion, etc.); that part of the
workflow is deliberately excluded (see *Scope* below).

---

## What's inside

```
pibo_reaxff/      PIBO Bayesian-optimization framework (the method)
  optimizers/     PIBO + baselines (PSO, CMA-ES, INDEEDopt, JAX-ReaxFF)
  core/           GP surrogate, blocked / staged PIBO
  data/           VASP/OUTCAR readers + preprocessor
src/              UQ reviewer-response analysis modules
  gp_utils.py     GP surrogate IO + finite-difference grad/Hessian
  hmc.py          NUTS/HMC posterior over the GP-surrogate likelihood
  sloppy.py       Sethna-style sloppy spectrum + profile likelihood
  uncertainty.py  σ_opt / σ_GP / σ_post comparison
  ffield_parse.py pure ReaxFF text parser (no LAMMPS)  ← see Notes
  plotting.py     shared figure styling
scripts/          one-shot drivers
  phase01..08     the UQ pipeline (run in order)
  build_*.py      regenerate data/ inputs (need raw logs, see Notes)
configs/          run + base configs for the BO framework
data/             pre-built UQ inputs (GP surrogate + replicate table + bounds)
results/ffield/   calibrated ReaxFF force fields (*.reax)
results/reviewer_response/recalib_staged_bo/  minimal staged-BO inputs the UQ pipeline reads
vasp_calculations/  DFT reference frames (bond / angle / torsion / nonbond)
lammps_templates/ ReaxFF ffield templates (no LAMMPS data decks)
run.py, main.py   BO framework entry points
```

## Uncertainty-quantification pipeline (the reviewer response)

The UQ analysis is fully surrogate-based — **it never calls LAMMPS or
VASP** — and reproduces the three distinct uncertainty sources plus the
sloppy-model analysis:

| Source | Meaning | Phase |
|--------|---------|-------|
| **σ_opt**  | optimizer replication variance | `phase05` (from `data/optimizer_replicates.csv`) |
| **σ_GP**   | GP surrogate predictive std | `phase05` |
| **σ_post** | marginal parameter posterior, via HMC on the GP likelihood | `phase04` |
| sloppy spectrum + profile likelihood | structural ill-posedness | `phase02`, `phase03` |

Run, in order (each phase gates on the previous one's diagnostics):

```bash
python scripts/phase01_validate.py     # GP surrogate sanity gate
python scripts/phase02_sloppy.py       # Hessian eigen-spectrum
python scripts/phase03_profile.py      # profile likelihood (stiff/sloppy)
python scripts/phase04_hmc.py          # HMC posterior  (≥4 chains, R̂<1.01, ESS≥400, 0 divergences)
python scripts/phase05_compare.py      # σ_opt vs σ_GP vs σ_post
python scripts/phase06_figure9.py      # composite figure
python scripts/phase07_table2.py       # Table 2 + Table S1
python scripts/phase08_rebuttal.py     # rebuttal text assembly
```

Outputs land under `outputs/{data,figures,diagnostics,tables}/` (gitignored;
the directory skeleton ships with `.gitkeep`).

**Hard-fail discipline:** every phase halts with a clear error if its
diagnostics do not pass (e.g. `phase04` raises if any chain has R̂ ≥ 1.01,
ESS < 400, or a post-warmup divergence). No silent fallbacks.

## Install

```bash
python -m pip install -e .            # core: numpy, scipy, pandas, matplotlib, pyyaml, tqdm
python -m pip install numpyro arviz   # required for phase04 HMC (jax backend)
python -m pip install scikit-learn    # required by the GP surrogate / src.gp_utils
```

Python ≥ 3.11. The BO framework additionally needs a LAMMPS `lmp`
executable on `PATH` (ReaxFF-enabled) — install separately via
`conda install -c conda-forge lammps`. The UQ pipeline does **not** need
LAMMPS.

## Re-calibrating ReaxFF parameters

Two calibration routes produce a force field. **Both require a
ReaxFF-enabled LAMMPS** — set `PIBO_LMP` to the binary, or put `lmp` on
PATH (`conda install -c conda-forge lammps`):

```bash
# Windows
set PIBO_LMP=C:\path\to\lmp.exe
# Linux / macOS
export PIBO_LMP=/path/to/lmp
```

1. **PES-targeted (built-in framework)** — staged BO against the DFT
   potential-energy surface in `vasp_calculations/`, then export the
   resulting parameters:
   ```bash
   python run.py --staged \
     --export-best-ffield results/ffield/ffield.reax.MoSH.recalibrated.reax
   ```
   Writes the new `*.reax` plus a JSON sidecar of metrics.

2. **Biaxial/property-targeted (recalibration drivers)** — the scripts
   that produced the manuscript's force fields. They warm-start from a
   prior `results/ffield/*.reax` and optimize against
   `MoS2_physical_validation.csv` via LAMMPS:
   ```bash
   pip install -e ".[recalib]"          # adds scikit-optimize
   python scripts/recalib_combined_all.py     # 7-objective driver (canonical)
   python scripts/pibo_biaxial_recalib_v13.py # latest biaxial-only chain step
   python scripts/recalib_staged_bo.py        # staged-BO driver (-> staged_bo.reax)
   ```
   The v-series is **runtime-chained**: `vN` reads `v(N-1)`'s output
   `.reax`, so run them in order (or ensure the warm-start file exists).
   `recalib_staged_bo.py` imports helpers from `recalib_combined_all.py`.

## Reproducibility

- Global random seed **42** is fixed and logged by every phase.
- `data/gp_surrogate.pkl` is the trained GP surrogate consumed by the UQ
  pipeline; `data/optimizer_replicates.csv` and `data/parameter_bounds.csv`
  are the replicate table and BO bounds.

## Scope — intentionally excluded

This repository ships code for **reproducibility**. The **biaxial /
property-targeted recalibration drivers** (`scripts/pibo_biaxial_recalib_*`,
`scripts/recalib_staged_bo.py`, `scripts/recalib_combined_all.py`) and the
LAMMPS data decks they need are **included** so the manuscript's force
fields can be regenerated (see *Re-calibrating ReaxFF parameters* above).

Still omitted: the standalone physical-validation/figure post-processing
one-shots (stress–strain aggregation, vacancy/diffusion analysis,
manuscript-figure generators), the large run-history, and backup trees.
Accordingly, `pibo_reaxff/physical_validation.py` is not shipped, so two
BO code paths degrade gracefully: the optional in-loop validation gate
(`optimizers/pibo.py::_run_validation_gate`) and the geometry-anchor loss
term (`loss.py`, which silently contributes 0 when the helper is absent).
The optimizer, the GP surrogate, the recalibration drivers, and the entire
UQ pipeline are unaffected.

## Notes

- `src/ffield_parse.py` is a **pure** ReaxFF-text parser, factored out of
  the recalibration driver (`recalib_combined_all.py`) so the UQ phases can
  read a calibrated force field without importing any LAMMPS/validation code
  (the UQ pipeline stays LAMMPS-free; the recalibration drivers do not).
- `scripts/build_phase01_inputs.py`, `scripts/build_parameter_bounds.py`,
  and `scripts/retrain_gp_posterior_in_eV.py` document how the `data/`
  artifacts were produced. Re-running them from scratch requires the raw
  staged-BO logs (`RECALIB_LOG_clean.csv`, `trial_scan_rmse.csv`,
  `posterior_snapshots/`), which are **not** bundled here — the pre-built
  `data/` artifacts are shipped instead.

## License

MIT (see `pyproject.toml`).
