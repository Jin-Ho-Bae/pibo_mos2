# `pibo_calculation/` — PIBO ReaxFF calibration

`recalib_staged_bo.py` calibrates the MoS₂ ReaxFF parameters with the
manuscript's staged Bayesian optimization. The **calibration objective**
(Eq. 1) is the category-weighted ReaxFF-vs-DFT **energy** residual over the
bond/angle/torsion/non-bonded DFT scans. The uniaxial/biaxial stress–strain
and S-monovacancy properties are **held-out validation** (not in the
objective) and are checked with `--validate`.

Schedule (N=100): warm-start(1) → LHS(14) → EI + Thompson p=0.15 (45) →
LCB β=1.5 (40). Warm-start x0 = the Note S1 / Cooper-derived prior centre.

## Install

```bash
pip install numpy pandas scikit-learn scikit-optimize
```

LAMMPS (ReaxFF-enabled) is required — point to it with `PIBO_LMP`, or put
`lmp` on `PATH`:

```bash
# Windows
set PIBO_LMP=C:\path\to\lmp.exe
# Linux / macOS
export PIBO_LMP=/path/to/lmp
```

## Inputs (`data/`)

| File | Status | Used by |
|------|--------|---------|
| `dft_reference/{bond,angle,torsion,nonbond}/` | shipped | calibration objective (Eq. 1) |
| `optimizer_variable_bounds.txt` | shipped | parameter search box |
| `MoS2_physical_validation.csv` | shipped | held-out validation only |
| `data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata` | supply your own | held-out validation only |

## Run

```bash
cd pibo_calculation
python recalib_staged_bo.py --budget 100 --seed 42   # calibrate (energy objective)
python recalib_staged_bo.py --validate               # held-out stress-strain + V_S check
```

Run from inside the repository (the calibration imports the repo's
`pibo_reaxff` for the LAMMPS-backed energy evaluation). Results — the
calibrated force field `ffield.reax.MoSH.pibo_calibrated.reax`,
`RECALIB_LOG.csv`, `BEST_RESULT.txt`, and `posterior_snapshots/` — are
written to `output/`.
