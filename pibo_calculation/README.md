# `pibo_calculation/` — most-recent PIBO recalibration (self-contained)

This folder bundles the Python code and **input data** for the most recent
PIBO ReaxFF re-calibration of MoS₂. Its outputs are the values reported in
the manuscript's `results/reviewer_response/recalib_staged_bo/` figures.

**Results are intentionally not included** — they are shown as manuscript
figures. Running the driver regenerates them under `output/`.

## Method

Staged Bayesian optimization with the acquisition schedule:

```
Stage 1  trials  1–15   LHS exploration (cold start)
Stage 2  trials 16–60   EI  + Thompson-sampling injection (p = 0.15)
Stage 3  trials 61–end  LCB exploitation
```

Objective = worst-of-five relative error vs DFT (biaxial σ, uniaxial x1/x2 σ,
V_S vacancy-formation energy, S-diffusion barrier), each evaluated by LAMMPS.

## Layout

```
pibo_calculation/
  recalib_staged_bo.py     # ENTRY POINT — staged BO driver
  recalib_combined_all.py  # evaluator library (ReaxFF parse/write, LAMMPS decks, evaluate())
  _lmp_path.py             # locates the LAMMPS executable
  data/                    # input DFT data used by the run
    MoS2_physical_validation.csv   # DFT reference — full biaxial + uniaxial
                                   # (x1 zigzag, x2 armchair) stress–strain + h_S
  output/                  # created at run time (results — NOT shipped)
```

The `data/` folder contains the DFT reference data (all biaxial and uniaxial
stress–strain curves). Two run-time inputs are **not** bundled here:

- the **warm-start ReaxFF force field** — provided in the **Supporting
  Information**; place it at `data/ffield.reax.MoSH.pibo_biaxial_v9.reax`;
- the **LAMMPS structure deck** (a 10×10 2H-MoS₂ monolayer) — supply your own
  at `data/data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata`.

Drop both into `data/` to run the driver end-to-end.

## Requirements

- Python ≥ 3.11 with `numpy`, `pandas`, `scipy`, `scikit-optimize`
- A ReaxFF-enabled **LAMMPS** executable

```bash
pip install numpy pandas scipy scikit-optimize
# point to LAMMPS (or put `lmp` on PATH):
#   Windows :  set PIBO_LMP=C:\path\to\lmp.exe
#   Linux/Mac: export PIBO_LMP=/path/to/lmp
```

## Run

First place the two non-bundled inputs into `data/` (warm-start force field
from the Supporting Information + a 10×10 MoS₂ LAMMPS structure deck — see
*Layout*), then:

```bash
cd pibo_calculation
python recalib_staged_bo.py --budget 100 --seed 42
```

The best force field (`ffield.reax.MoSH.staged_bo.reax`), the per-trial
`RECALIB_LOG.csv`, `BEST_RESULT.txt`, and `posterior_snapshots/` are written
to `output/`.

## Notes

- Paths are local to this folder (inputs in `data/`, results in `output/`);
  no edits are needed to run it elsewhere.
- The per-machine hardcoded LAMMPS paths from the original scripts were
  replaced by `_lmp_path.find_lmp()` (`$PIBO_LMP` → PATH → clear error).
- Seed is fixed (`--seed 42`).
