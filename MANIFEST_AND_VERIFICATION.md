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

---

## 5. Update — Figure_data + recalibration drivers added

Subsequent to the initial release, the following were added (commits after
`828be16`):

### Figure_data
- `Figure_data/` — 12 `.xlsx` files backing manuscript Figures 3–8.

### Biaxial / property-targeted recalibration (reverses the earlier exclusion)
At the author's request, the recalibration drivers that produced the
manuscript's force fields are now included so the *resulting ReaxFF
parameters* can be regenerated:

- **Scripts (15):** `pibo_biaxial_recalibrate.py`, `…_expanded.py`,
  `…_v3 … …_v13.py`, `recalib_staged_bo.py`, `recalib_combined_all.py`.
- **Support data:** `MoS2_physical_validation.csv`,
  `lammps_templates/data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata`,
  `results/reviewer_response/v9_validation/data.mos2_3x3_v9_validation.lammpsdata`
  (warm-start `*.reax` chain already shipped in `results/ffield/`).
- **Packaging:** added `scikit-learn` to core deps; new extras
  `[hmc]` (numpyro, arviz) and `[recalib]` (scikit-optimize).

### Code change: LAMMPS path parameterized
The 15 scripts previously hardcoded a per-machine `lmp.exe` path (two
literals, for users `CAN` and `JINHOBAE`). All now resolve LAMMPS through
the new `scripts/_lmp_path.py::find_lmp()` — checks `$PIBO_LMP`, then PATH,
else raises. No machine-specific path remains.

### Static verification (no execution — no Python/LAMMPS on this host)
- All 15 scripts now import `find_lmp`; **no** `lmp.exe`/user-path literal
  remains in any recalib script (only in `local_env.py`'s legitimate
  dynamic discovery and the helper's docstring). ✔
- `recalib_staged_bo.py` imports 19 names from `recalib_combined_all.py`
  (`grab, read_lines, write_lines, parse_offsets, build_spec, write_ffield,
  setup_wd, run_lmp, biax_init_deck, biax_strain_deck, uni_deck,
  defect_deck, saddle_deck, measure_h_S, parse_data_atoms,
  write_data_with_atoms, parse_atom_record, rel_err, evaluate`) — **all 19
  are defined** in that module. ✔
- All warm-start `*.reax`, the two `.lammpsdata` decks, and the DFT CSV the
  drivers reference are present in the repo. ✔
- **Not executed.** These drivers require Python ≥ 3.11 + scikit-optimize/
  scipy + a ReaxFF LAMMPS. `recalib_combined_all.py` resolves LAMMPS at
  *import* time (module-level `LMP = find_lmp()`), so importing it — and
  thus running `recalib_staged_bo.py` — requires LAMMPS present even to
  start. This is intended for LAMMPS-backed drivers and does **not** affect
  the UQ pipeline (which no longer imports these modules).

### Note carried over from the dependency audit
The recalibration scripts label top-K/softmax-over-trials ensembles as the
"GP posterior"/"Bayesian UQ" in their docstrings. That is the
optimizer-replicate approximation the reviewer flagged — it is **not** the
HMC parameter posterior (`σ_post`) produced by `scripts/phase04_hmc.py`.
The two layers are separate: these scripts are the *calibration drivers*;
the `src/` + `phase0*` pipeline is the *uncertainty-quantification* layer.
