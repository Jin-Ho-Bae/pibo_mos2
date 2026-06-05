# Curation manifest & code-verification report

> Prepared for review **before** pushing to
> `https://github.com/Jin-Ho-Bae/pibo_mos2`.
> This is a **local commit only — nothing has been pushed.**

## 1. How this repository was assembled

Source workspace: `…/PIBO/PIBO_mos` (a ~4 GB research tree). This repo is
an **allowlist copy** of that tree — only the files needed for
reproducibility were copied; the 4 GB of run-history (`results_*`,
`*Backup`, nested `data/vasp_calculations`, `__pycache__`, manuscript
binaries) was left behind.

### Included
- **BO framework** `pibo_reaxff/` — code modules only (flat modules +
  `core/` + `optimizers/` + `data/` readers).
- **UQ pipeline** `src/` (incl. new `ffield_parse.py`) + `scripts/phase01–08`
  + `build_phase01_inputs.py` + `build_parameter_bounds.py` +
  `retrain_gp_posterior_in_eV.py`.
- **Configs / data** `configs/`, `data/` (gp_surrogate.pkl + 2 csv),
  `lammps_templates/*.template` (no `.lammpsdata`).
- **Entry points / packaging** `run.py`, `main.py`, `pyproject.toml`,
  `README.md`, `.gitignore`.

### Included but **gated** (please vet before push)
- `vasp_calculations/` — 172 DFT reference files (CONTCAR/OUTCAR).
- `results/ffield/*.reax` — 23 calibrated ReaxFF force fields.
- `results/reviewer_response/recalib_staged_bo/…` — the single
  `precise_ffields/ffield.reax.staged_bo.reax` + `…/cache/gp_calibration_diag.json`
  that the UQ pipeline reads at runtime.

These correspond to the three categories you asked to personally check:
the **acquisition stage**, the **reused ReaxFF parameters**, and the
**resulting VASP** data. They are committed locally so you can review the
staged content in place; **no remote push has been made.**

### Excluded (per the "reproducibility only — no LAMMPS/validation" rule)
- `pibo_reaxff/physical_validation.py` (stress–strain / vacancy validation).
- All `scripts/` one-shots that run validation, generate LAMMPS decks,
  recalibrate via LAMMPS, or build manuscript figures/docx (~100 files,
  e.g. `recalib_combined_all.py`, `run_quasi_static_*`, `eval_vacancy_*`,
  `aggregate_stress_strain.py`, `manuscript_figures_*`, `make_*`).
- Broken/legacy modules: `pibo_reaxff/physics/` (imports 9 missing
  `tf_*` files), `core/integrated_pibo.py`, `core/blocked_pibo_coulomb.py`.
- `lammps_templates/*.lammpsdata`, `*.pdf`, `*.docx`, logs, run histories.

## 2. Code changes made during curation

1. **New `src/ffield_parse.py`** — the 5 pure ReaxFF-text parsers
   (`read_lines`, `write_lines`, `parse_offsets`, `build_spec`,
   `write_ffield`) extracted verbatim from the excluded LAMMPS driver
   `recalib_combined_all.py`. No LAMMPS code.
2. **Re-pointed imports** in `phase01/02/03/07` and
   `build_parameter_bounds.py`: `import recalib_combined_all as _rca`
   → `from src import ffield_parse as _rca` (call sites unchanged).
   This removes the UQ pipeline's only dependency on validation code.

## 3. Verification ("prove the code")

> **No Python interpreter is available on the preparation machine**
> (the original workflow was Colab-based). Verification below is **static**
> — import-graph / syntax / data-flow analysis, not execution. Please run
> the pipeline once in a real environment before relying on the figures.

### Verified
- **UQ pipeline never touches LAMMPS or VASP.** After the import
  re-pointing, `src/` + `phase01–08` depend only on `numpy/scipy/pandas/
  matplotlib/scikit-learn` and, for `phase04`, `numpyro/arviz/jax`. ✔
- **Import coherence of the BO core.** The optimizer closure
  (`optimizers/* + parameters + gp_surrogate + physics_constraints`) is
  clean and LAMMPS-free at import. Dropping `physics/` is safe — the only
  importer (`core/blocked_pibo.py`) guards it with `try/except`. Dropping
  `physical_validation.py` is import-safe and **run-safe**: `loss.py`
  references it lazily inside a `try/except Exception: pass`, so the
  geometry-anchor term degrades to 0; `pibo.py` references it only inside
  the validation-gate method, which is never reached without a validator. ✔
- **Hard-fail discipline.** `phase04` raises on R̂≥1.01 / ESS<400 /
  divergences>0 / missing σ_n; `phase01` asserts `input_dim==40` and
  surrogate accuracy; `phase08` raises on failed rebuttal validation. ✔

### Known issues to be aware of (not fixed — flagged for your call)
- **Soft-fail gaps (CLAUDE.md rule 6):** `phase03` and `phase07` write
  `passed:true`/exit 0 even when their documented stop-condition fails
  (they log a warning instead of raising). Consider hardening to `raise`.
- **Version fragility:** `phase06` uses `legend_handles` (needs
  matplotlib ≥ 3.7); `phase04` uses `idata.posterior.dims[...]`
  (deprecated in newer xarray — prefer `.sizes`).
- **Silent joins:** `build_phase01_inputs.py`, `phase05`, `phase07` use
  `how="inner"`/`reindex` without row-count assertions, which can shrink
  the parameter set without erroring if an input is partial.
- **σ_opt proxy:** `src/uncertainty.py` computes σ_opt from the spread of
  top-K BO trials rather than 500 independent restarts; this is disclosed
  in its docstring but differs from the literal CLAUDE.md spec.

None of the above are parse-time/syntax errors; the repository imports and
the UQ pipeline runs end-to-end from the shipped `data/` artifacts in a
correctly-provisioned environment.

## 4. How to push (when you're satisfied)

```bash
cd "…/PIBO/pibo_mos2"
git remote add origin https://github.com/Jin-Ho-Bae/pibo_mos2.git
git push -u origin main
```

Review `git status` / `git log -1 --stat` first. To drop any gated
artifact before publishing, `git rm --cached <path>` and re-commit.
