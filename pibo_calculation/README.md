# `pibo_calculation/` — PIBO ReaxFF recalibration

One driver (`recalib_staged_bo.py`) runs the two-stage PIBO recalibration of
MoS₂ ReaxFF: a PES fit against the bond/angle/torsion/non-bonded DFT
scans, then a staged BO against the biaxial/uniaxial stress–strain targets.

## Install

```bash
pip install numpy pandas scipy scikit-learn scikit-optimize
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

| File | Status |
|------|--------|
| `MoS2_physical_validation.csv` | shipped (DFT biaxial + uniaxial stress–strain) |
| `dft_reference/{bond,angle,torsion,nonbond}/` | shipped (DFT PES scans) |
| `data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata` | supply your own |
| `ffield.reax.MoSH.pibo_biaxial_v9.reax` | optional warm-start (Supporting Information) |

## Run

```bash
cd pibo_calculation
python recalib_staged_bo.py --budget 100 --pes-budget 100 --seed 42
```

Run from inside the repository (the PES-fit stage imports the repo's
`pibo_reaxff`). Results — the recalibrated force fields, `RECALIB_LOG.csv`,
`BEST_RESULT.txt`, and `posterior_snapshots/` — are written to `output/`.
