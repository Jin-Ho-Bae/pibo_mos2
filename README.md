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

## Folder description

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
