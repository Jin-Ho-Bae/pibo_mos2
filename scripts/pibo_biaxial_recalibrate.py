"""PIBO re-calibration with Cooper-2014 biaxial stress-strain as direct target.

User explicitly allowed ffield re-calibration via the PIBO framework only.
Other optimizers stay as-is — purpose is to showcase PIBO's higher applicability.

Strategy:
  - Start from existing PIBO calibrated ffield
  - Vary ~12 most-impactful parameters (Mo-S bond + S-Mo-S angle + Mo/S vdW)
  - Bayesian Optimization via scikit-optimize (gp_minimize)
  - Loss = max relative error vs Cooper-2014 biaxial at 8 sparse strain points
  - Bounds: ±30% around current PIBO calibrated values
  - Budget: 100 evaluations
  - Output: ffield.reax.MoSH.pibo_biaxial_recalibrated.reax
"""
from __future__ import annotations
import math, os, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
from skopt import gp_minimize
from skopt.space import Real

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.calibrated.reax"
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_recalibrated.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA_FILE  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib"
LOG_PATH   = OUT_DIR / "RECALIB_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"

GPA_PER_ATM= 1.01325e-4
H_EFF_AA   = 6.145
ELEMENTS   = ("Mo","S")
LMP_CONTROL = (
    "tabulate_long_range 10000\nnbrhood_cutoff 5.0\nhbond_cutoff 6.0\n"
    "bond_graph_cutoff 0.3\nthb_cutoff 0.001\nthb_cutoff_sq 0.00001\nwrite_freq 0\n"
)
_NUM = r"[+-]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
def grab(t,tag):
    m=re.search(rf"{tag}\s+({_NUM})\b",t,re.I)
    return float(m.group(1)) if m and math.isfinite(float(m.group(1))) else None


# ---- Parameter parser/writer for ReaxFF ffield (line-based) -----------
# The ffield file has fixed column widths. We need to identify the lines
# containing each parameter, replace the value, write back.
# Format pattern: " " + float + " !comment"
# We index parameters by their structural location and override values.

def read_ffield_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def write_ffield_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Identify lines to modify. From parameters.py: bond/atom/offdiag/angle sections.
# We'll hard-code line indices after analyzing the file structure.
# For now, identify lines starting at known offsets in the calibrated ffield.

# Parameter names + their target lines (0-indexed) and column-within-line.
# Determined by inspecting the file structure (general 39 params + atom 33 + bond 16 + offdiag 7 + angle 7 + torsion 7 + hb 4).
# Better: parse by section header and offset.

def parse_section_offsets(lines):
    """Return dict of section -> (header_line, count_line).

    The ffield format is:
      Line 0: title
      Line 1: " 39       ! Number of general parameters" (39 + 1 = 40 next lines)
      ...
    """
    # Find headers by keywords
    offsets = {}
    i = 1
    n_gen = int(lines[i].split()[0]); i += 1
    offsets["general_start"] = i
    i += n_gen  # 39 general params
    n_atoms = int(lines[i].split()[0]); i += 1 + 3  # header + 3 comment lines
    offsets["atom_start"] = i
    offsets["n_atoms"] = n_atoms
    # Each atom takes 4 lines (32 params split)
    i += n_atoms * 4
    n_bonds = int(lines[i].split()[0]); i += 1 + 1  # header + 1 comment
    offsets["bond_start"] = i
    offsets["n_bonds"] = n_bonds
    # Each bond takes 2 lines
    i += n_bonds * 2
    n_off = int(lines[i].split()[0]); i += 1
    offsets["offdiag_start"] = i
    offsets["n_offdiag"] = n_off
    # Each offdiag takes 1 line
    i += n_off
    n_ang = int(lines[i].split()[0]); i += 1
    offsets["angle_start"] = i
    offsets["n_angle"] = n_ang
    return offsets


# Variable spec: (label, line_idx_in_full_lines, col_idx_in_line, init_value, lo_factor, hi_factor)
# We'll fill in after parsing.

def build_param_spec(lines, offsets):
    """Return list of dicts: {name, line, col, init, lo, hi}."""
    spec = []
    # General params: line offsets["general_start"] + index, single value per line
    # We don't vary general params -- skip.

    # Atom params: 5 atoms (Mo, S, H, ...) typically, 4 lines each, 8 values per line
    # We'll vary atom_0 (Mo) line 1 col [2,3,4,5] (RvdW, Dij, gamma, ro_pi)
    #                       and atom_1 (S) line 1 col [2,3,4,5]
    # In the actual format, atom block has 32 params per atom across 4 lines.
    # Easier to vary specific values by line+col index.
    # For now, just bond + offdiag + angle (Mo-S terms).

    # Bond block: 6 bonds * 2 lines each
    # Bond ordering: typically Mo-Mo, Mo-S, Mo-H, S-S, S-H, H-H -> we want Mo-S (index 1)
    bond_start = offsets["bond_start"]
    # Mo-S is bond #2 (after Mo-Mo), at lines bond_start + 2*1 = bond_start+2 and bond_start+3
    # Line 1 of bond block: index 1 type, then 8 values
    # Line 2 of bond block: 8 more values
    # The Mo-S bond line 1 has: De_sigma, De_pi, De_pipi, p_be1, p_bo5, 13corr, p_bo6, p_ovun1
    mos_l1 = bond_start + 2  # Mo-S, line 1
    mos_l2 = bond_start + 3  # Mo-S, line 2
    # Sample current values: parse the line
    p1 = lines[mos_l1].split()
    # The first token is bond-type pair (e.g. "1 2"); values start at col 2
    # Actually format: "  1  2  De_sigma  De_pi  De_pipi  p_be1  p_bo5  13corr  p_bo6  p_ovun1"
    spec.append({"name":"De_sigma_MoS","line":mos_l1,"col_in_floats":0,"init":float(p1[2])})
    spec.append({"name":"De_pi_MoS",  "line":mos_l1,"col_in_floats":1,"init":float(p1[3])})
    spec.append({"name":"p_be1_MoS", "line":mos_l1,"col_in_floats":3,"init":float(p1[5])})
    spec.append({"name":"p_bo5_MoS", "line":mos_l1,"col_in_floats":4,"init":float(p1[6])})
    spec.append({"name":"p_bo6_MoS", "line":mos_l1,"col_in_floats":6,"init":float(p1[8])})
    spec.append({"name":"p_ovun1_MoS","line":mos_l1,"col_in_floats":7,"init":float(p1[9])})
    # Line 2: p_be2, p_bo3, p_bo4, nu, p_bo1, p_bo2, nu, nu
    p2 = lines[mos_l2].split()
    spec.append({"name":"p_bo1_MoS", "line":mos_l2,"col_in_floats":4,"init":float(p2[4])})
    spec.append({"name":"p_bo2_MoS", "line":mos_l2,"col_in_floats":5,"init":float(p2[5])})

    # Off-diagonal Mo-S: offsets["offdiag_start"] + 0 (Mo-S is first)
    od_l = offsets["offdiag_start"]
    p3 = lines[od_l].split()
    # Format: "  1  2  Dij  RvdW  Alfa  ro_sigma  ro_pi  ro_pipi"
    spec.append({"name":"Dij_MoS",     "line":od_l,"col_in_floats":0,"init":float(p3[2])})
    spec.append({"name":"RvdW_MoS",    "line":od_l,"col_in_floats":1,"init":float(p3[3])})
    spec.append({"name":"Alfa_MoS",    "line":od_l,"col_in_floats":2,"init":float(p3[4])})
    spec.append({"name":"ro_sigma_MoS","line":od_l,"col_in_floats":3,"init":float(p3[5])})

    # Angle block: find S-Mo-S line (the angle controlling biaxial Poisson)
    # Angle format: "i j k  Theta_o  p_val1  p_val2  p_coa1  p_val7  p_pen1  p_val4"
    # Search for the line with " 2 1 2" (S-Mo-S in Mo,S,H ordering: Mo=1, S=2)
    ang_start = offsets["angle_start"]
    for ai in range(offsets["n_angle"]):
        ln_idx = ang_start + ai
        toks = lines[ln_idx].split()
        if len(toks) >= 3 and toks[0] == "2" and toks[1] == "1" and toks[2] == "2":
            ps = lines[ln_idx].split()
            spec.append({"name":"Thetao_SMoS", "line":ln_idx,"col_in_floats":0,
                         "init":float(ps[3])})
            spec.append({"name":"p_val1_SMoS","line":ln_idx,"col_in_floats":1,
                         "init":float(ps[4])})
            spec.append({"name":"p_val2_SMoS","line":ln_idx,"col_in_floats":2,
                         "init":float(ps[5])})
            spec.append({"name":"p_val7_SMoS","line":ln_idx,"col_in_floats":4,
                         "init":float(ps[7])})
            spec.append({"name":"p_val4_SMoS","line":ln_idx,"col_in_floats":6,
                         "init":float(ps[9])})
            break
    # Also Mo-S-Mo angle (governs interlayer-direction Poisson)
    for ai in range(offsets["n_angle"]):
        ln_idx = ang_start + ai
        toks = lines[ln_idx].split()
        if len(toks) >= 3 and toks[0] == "1" and toks[1] == "2" and toks[2] == "1":
            ps = lines[ln_idx].split()
            spec.append({"name":"Thetao_MoSMo", "line":ln_idx,"col_in_floats":0,
                         "init":float(ps[3])})
            spec.append({"name":"p_val1_MoSMo","line":ln_idx,"col_in_floats":1,
                         "init":float(ps[4])})
            break

    # Add bounds: ±30% of init value (sign-preserving for negative params)
    for s in spec:
        v = s["init"]
        margin = 0.30 * abs(v) if abs(v) > 0.01 else 0.1
        s["lo"] = v - margin
        s["hi"] = v + margin
    return spec


def write_ffield_with_overrides(src_lines, spec, overrides, dst_path):
    """Write a new ffield by replacing specific (line, col) float entries.

    Preserves column structure by replacing the entire line with a fixed
    formatted version. Other lines copied verbatim.
    """
    lines = list(src_lines)
    # Group spec by line
    by_line = {}
    for s, v in zip(spec, overrides):
        by_line.setdefault(s["line"], []).append((s["col_in_floats"], v, s["init"]))

    for ln_idx, mods in by_line.items():
        orig = lines[ln_idx]
        # Parse: leading "  i  j  " then floats
        parts = orig.split()
        # First, identify how many leading int tokens (atom-pair indices)
        n_lead = 0
        for tok in parts:
            try:
                _ = int(tok)
                n_lead += 1
            except ValueError:
                break
        # The floats start at index n_lead
        floats = []
        for tok in parts[n_lead:]:
            try:
                floats.append(float(tok))
            except ValueError:
                break
        # Apply mods
        for col, new_val, _ in mods:
            if col < len(floats):
                floats[col] = new_val
        # Reconstruct line with original column widths (10 chars per value, 4 decimals)
        lead = "  " + "  ".join(parts[:n_lead])
        body = "".join(f"{f:10.4f}" for f in floats)
        lines[ln_idx] = lead + body
    write_ffield_lines(dst_path, lines)


# ---- LAMMPS eval ----------------------------------------------------------

def setup_eval_dir(ff_path):
    wd = OUT_DIR / "_active"
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
        if wd.exists():
            time.sleep(0.2); shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ff_path, wd / "ffield.reax")
    (wd / "lmp_control").write_text(LMP_CONTROL, encoding="utf-8")
    (wd / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata").write_text(
        DATA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return wd


def run_lmp(deck, wd, tag, timeout=120):
    in_path = wd / f"in.{tag}.lmp"
    in_path.write_text(deck, encoding="utf-8")
    log_path = wd / f"log.{tag}.lammps"
    try:
        p = subprocess.run([LMP, "-in", in_path.name, "-log", log_path.name, "-screen", "none"],
                           cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, ""
    try:
        return p.returncode, log_path.read_text(errors="replace")
    except Exception:
        return -1, ""


def init_deck():
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        fix br all box/relax x 0.0 y 0.0 vmax 0.001
        minimize 1.0e-10 1.0e-12 3000 30000
        unfix br
        minimize 1.0e-10 1.0e-12 3000 30000
        write_data data.relaxed.lammpsdata
        print "RELAXED_LX $(lx)"
        print "RELAXED_LY $(ly)"
        """)


def strain_deck(target_lx, target_ly):
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data data.relaxed.lammpsdata
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        change_box all x final 0.0 {target_lx:.10f} y final 0.0 {target_ly:.10f} remap units box
        fix freeze all setforce 0.0 0.0 NULL
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        variable sig_xx_gpa equal -pxx*{GPA_PER_ATM}*lz/{H_EFF_AA}
        variable sig_yy_gpa equal -pyy*{GPA_PER_ATM}*lz/{H_EFF_AA}
        variable sig_iso_gpa equal 0.5*(v_sig_xx_gpa+v_sig_yy_gpa)
        write_data data.eps.lammpsdata
        print "FINAL_SIG_ISO_GPA ${{sig_iso_gpa}}"
        """)


def evaluate(spec, override_values, src_lines, dft_eps, dft_sig):
    """Return (max_rel_err, sigs, lx0). Penalize failed runs heavily."""
    # Write ffield with overrides to a temp path
    tmp_ff = OUT_DIR / "tmp_ffield.reax"
    write_ffield_with_overrides(src_lines, spec, override_values, tmp_ff)
    wd = setup_eval_dir(tmp_ff)
    rc, log = run_lmp(init_deck(), wd, "init", timeout=180)
    if rc != 0:
        return 9999.0, [], None
    lx0 = grab(log, "RELAXED_LX"); ly0 = grab(log, "RELAXED_LY")
    if lx0 is None:
        return 9999.0, [], None
    sigs = []
    for i, eps in enumerate(dft_eps):
        tlx = lx0 * math.exp(float(eps)); tly = ly0 * math.exp(float(eps))
        rc, log = run_lmp(strain_deck(tlx, tly), wd, f"s{i}", timeout=90)
        s = grab(log, "FINAL_SIG_ISO_GPA") if rc == 0 else None
        if s is None or not math.isfinite(s):
            return 9999.0, sigs, lx0
        sigs.append(s)
        # Copy relaxed-strain cell forward as next iteration's input
        shutil.copyfile(wd / "data.eps.lammpsdata", wd / "data.relaxed.lammpsdata")
    sigs = np.array(sigs)
    # RAW comparison (no zero-shift): PIBO sigma at eps_PIBO directly compared
    # to DFT sigma at the same eps. Both curves are anchored at their own
    # equilibrium (PIBO by box-relax, DFT by definition), so sigma(eps=0)=0
    # for both already; no shift needed.
    denom = np.where(np.abs(dft_sig) > 1e-6, np.abs(dft_sig), 1.0)
    rel = np.where(np.abs(dft_sig) > 1e-6, np.abs(sigs - dft_sig)/denom*100, np.nan)
    valid = (np.abs(dft_sig) > 1e-6) & np.isfinite(rel)
    if not valid.any():
        return 9999.0, sigs, lx0
    max_re = float(np.nanmax(rel[valid]))
    return max_re, sigs.tolist(), lx0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_lines = read_ffield_lines(SRC_FFIELD)
    offsets = parse_section_offsets(src_lines)
    spec = build_param_spec(src_lines, offsets)
    print(f"[setup] Varying {len(spec)} parameters:", flush=True)
    for s in spec:
        print(f"  {s['name']:20s}  init={s['init']:.4f}  bounds=[{s['lo']:.4f}, {s['hi']:.4f}]",
              flush=True)

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)

    # Log header
    if not LOG_PATH.exists():
        header = "trial,timestamp,max_re," + ",".join(s["name"] for s in spec) + ",sigs\n"
        LOG_PATH.write_text(header, encoding="utf-8")

    best = {"max_re": float("inf"), "params": None, "sigs": None}
    trial_count = [0]

    def objective(x):
        trial_count[0] += 1
        t0 = time.time()
        max_re, sigs, lx0 = evaluate(spec, x, src_lines, dft_eps, dft_sig)
        rt = time.time() - t0
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{trial_count[0]},{ts},{max_re:.3f}," +
                     ",".join(f"{v:.6f}" for v in x) + f",\"{sigs_str}\"\n")
        marker = ""
        if max_re < best["max_re"]:
            best["max_re"] = max_re
            best["params"] = list(x)
            best["sigs"] = sigs
            marker = " ← NEW BEST"
            # Save best ffield
            write_ffield_with_overrides(src_lines, spec, x, DST_FFIELD)
            BEST_PATH.write_text(
                f"trial {trial_count[0]}\nmax_re = {max_re:.2f}%\n"
                f"sigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                f"params = {dict(zip([s['name'] for s in spec], list(x)))}\n",
                encoding="utf-8")
        print(f"trial {trial_count[0]:3d}  [{rt:5.1f}s]  max_re={max_re:7.2f}%  "
              f"best={best['max_re']:7.2f}%{marker}", flush=True)
        if max_re <= 5.0:
            print(f"\n*** TARGET MET at trial {trial_count[0]}: max_re={max_re:.2f}% ***",
                  flush=True)
        return max_re

    # BO search
    bounds = [Real(s["lo"], s["hi"], name=s["name"]) for s in spec]
    x0 = [s["init"] for s in spec]
    print(f"\n[setup] Initial guess (PIBO calibrated) eval first...", flush=True)
    result = gp_minimize(
        objective, bounds, x0=x0, n_calls=100, n_initial_points=10,
        acq_func="EI", random_state=42, verbose=False,
    )
    print(f"\nFinal best: max_re = {best['max_re']:.2f}%")
    print(f"PIBO biaxial-recalibrated ffield: {DST_FFIELD}")


if __name__ == "__main__":
    main()
