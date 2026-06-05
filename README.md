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

## Reproducibility

- Global random seed **42** is fixed and logged by every phase.
- `data/gp_surrogate.pkl` is the trained GP surrogate consumed by the UQ
  pipeline; `data/optimizer_replicates.csv` and `data/parameter_bounds.csv`
  are the replicate table and BO bounds.

## Scope — intentionally excluded

Per the project's release policy, this repository ships code for
**reproducibility only**. It deliberately omits:

- the LAMMPS **validation** suite (stress–strain, mono-/vacancy formation,
  S-diffusion) and the scripts that drive it;
- the scripts that **generate LAMMPS input decks** / data files;
- the large run-history, manuscript-figure one-shots, and backup trees.

As a consequence, two BO code paths degrade gracefully and are inactive
here: the optional in-loop physical-validation gate
(`optimizers/pibo.py::_run_validation_gate`) and the geometry-anchor loss
term (`loss.py`, which silently contributes 0 when the validation helpers
are absent). The optimizer algorithm itself, the GP surrogate, and the
entire UQ pipeline are unaffected.

## Notes

- `src/ffield_parse.py` is a **pure** ReaxFF-text parser, factored out of
  the (excluded) recalibration driver so the UQ phases can read a
  calibrated force field without pulling in any LAMMPS/validation code.
- `scripts/build_phase01_inputs.py`, `scripts/build_parameter_bounds.py`,
  and `scripts/retrain_gp_posterior_in_eV.py` document how the `data/`
  artifacts were produced. Re-running them from scratch requires the raw
  staged-BO logs (`RECALIB_LOG_clean.csv`, `trial_scan_rmse.csv`,
  `posterior_snapshots/`), which are **not** bundled here — the pre-built
  `data/` artifacts are shipped instead.

## License

MIT (see `pyproject.toml`).
