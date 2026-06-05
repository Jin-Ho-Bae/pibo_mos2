"""PIBO recalib v9: escape v8 local minimum at 11.79%. Add bond-curvature
params (p_be2, p_bo3, p_bo4) to attack the mid-strain (eps=0.15) undershoot.
22 params total. Large initial simplex (15% steps) to leap out of v8 basin.

NM phase first, then Powell. Bootstrap from v8 trial 338 best.

Shape residual at v8 best:
  +8% at eps=0.025  (too stiff small-strain)
  -12% at eps=0.15  (too soft mid-strain)  ← bond-curvature handle
  +3% at eps=0.30
"""
from __future__ import annotations
import math, re, shutil, subprocess, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v8.reax"
FB_FFIELD  = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v6.reax"
if not SRC_FFIELD.exists(): SRC_FFIELD = FB_FFIELD
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v9.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA_FILE  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib_v9"
LOG_PATH   = OUT_DIR / "RECALIB_V9_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
TARGET_MAX_RE = 10.0

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


def read_lines(p): return p.read_text(encoding="utf-8").splitlines()
def write_lines(p, lines): p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_offsets(lines):
    o = {}; i = 1
    n_gen = int(lines[i].split()[0]); i += 1 + n_gen
    n_atoms = int(lines[i].split()[0]); i += 1 + 3
    o["atom_start"] = i; o["n_atoms"] = n_atoms; i += n_atoms * 4
    n_bonds = int(lines[i].split()[0]); i += 1 + 1
    o["bond_start"] = i; o["n_bonds"] = n_bonds; i += n_bonds * 2
    n_off = int(lines[i].split()[0]); i += 1
    o["offdiag_start"] = i; o["n_offdiag"] = n_off; i += n_off
    n_ang = int(lines[i].split()[0]); i += 1
    o["angle_start"] = i; o["n_angle"] = n_ang; i += n_ang
    n_tor = int(lines[i].split()[0]); i += 1
    o["tor_start"] = i; o["n_tor"] = n_tor
    return o


WIDEN = {
    "De_sigma_MoS":  (0.70, 1.80),
    "De_pi_MoS":     (0.70, 1.80),
    "p_val1_SMoS":   (0.60, 1.80),
    "p_val1_MoSMo":  (0.50, 2.20),
    "Thetao_SMoS":   (0.92, 1.15),
    "Thetao_MoSMo":  (0.92, 1.15),
    "p_be1_MoS":     (0.50, 2.00),
    "p_be2_MoS":     (0.50, 2.00),  # NEW
    "p_bo1_MoS":     (0.60, 1.40),
    "p_bo2_MoS":     (0.70, 1.40),
    "p_bo3_MoS":     (0.50, 1.80),  # NEW
    "p_bo4_MoS":     (0.50, 1.80),  # NEW
    "p_bo5_MoS":     (0.60, 1.50),
    "p_bo6_MoS":     (0.50, 1.80),
    "p_ovun1_MoS":   (0.30, 2.00),
    "Dij_MoS":       (0.40, 2.50),
    "RvdW_MoS":      (0.85, 1.25),
    "Alfa_MoS":      (0.60, 1.60),
    "ro_sigma_MoS":  (0.92, 1.12),
    "p_val2_SMoS":   (0.60, 1.60),
    "p_val7_SMoS":   (0.60, 1.60),
    "p_val4_SMoS":   (0.60, 1.60),
}


def build_spec(lines, off):
    spec = []
    bs = off["bond_start"]; mos_l1, mos_l2 = bs + 2, bs + 3
    p1 = lines[mos_l1].split(); p2 = lines[mos_l2].split()
    for n, c in [("De_sigma_MoS",0),("De_pi_MoS",1),("p_be1_MoS",3),
                 ("p_bo5_MoS",4),("p_bo6_MoS",6),("p_ovun1_MoS",7)]:
        spec.append({"name":n,"line":mos_l1,"col_in_floats":c,"init":float(p1[2+c])})
    # line 2: c0=p_be2, c1=p_bo3, c2=p_bo4, c4=p_bo1, c5=p_bo2
    for n, c in [("p_be2_MoS",0),("p_bo3_MoS",1),("p_bo4_MoS",2),
                 ("p_bo1_MoS",4),("p_bo2_MoS",5)]:
        try:
            v = float(p2[c])
            # Skip if exactly zero (will cause bound issues)
            if abs(v) < 1e-6 and n in ("p_be2_MoS","p_bo3_MoS","p_bo4_MoS"):
                continue
            spec.append({"name":n,"line":mos_l2,"col_in_floats":c,"init":v})
        except (IndexError, ValueError):
            pass
    od = off["offdiag_start"]; p3 = lines[od].split()
    for n, c in [("Dij_MoS",0),("RvdW_MoS",1),("Alfa_MoS",2),("ro_sigma_MoS",3)]:
        spec.append({"name":n,"line":od,"col_in_floats":c,"init":float(p3[2+c])})
    ang = off["angle_start"]
    for ai in range(off["n_angle"]):
        ln = ang + ai; toks = lines[ln].split()
        if len(toks) >= 3 and toks[0] == "2" and toks[1] == "1" and toks[2] == "2":
            for n, c in [("Thetao_SMoS",0),("p_val1_SMoS",1),("p_val2_SMoS",2),
                         ("p_val7_SMoS",4),("p_val4_SMoS",6)]:
                spec.append({"name":n,"line":ln,"col_in_floats":c,"init":float(toks[3+c])})
            break
    for ai in range(off["n_angle"]):
        ln = ang + ai; toks = lines[ln].split()
        if len(toks) >= 3 and toks[0] == "1" and toks[1] == "2" and toks[2] == "1":
            for n, c in [("Thetao_MoSMo",0),("p_val1_MoSMo",1)]:
                spec.append({"name":n,"line":ln,"col_in_floats":c,"init":float(toks[3+c])})
            break
    for s in spec:
        v = s["init"]; sign = -1.0 if v < 0 else 1.0; av = abs(v)
        lo_m, hi_m = WIDEN.get(s["name"], (0.5, 1.8))
        lo_abs = max(1e-6, av * lo_m)
        hi_abs = av * hi_m
        if sign < 0:
            s["lo"] = -hi_abs; s["hi"] = -lo_abs
        else:
            s["lo"] = lo_abs; s["hi"] = hi_abs
    return spec


def write_ffield(src_lines, spec, overrides, dst):
    lines = list(src_lines)
    by_line = {}
    for s, v in zip(spec, overrides):
        by_line.setdefault(s["line"], []).append((s["col_in_floats"], v))
    for ln_idx, mods in by_line.items():
        orig = lines[ln_idx]; parts = orig.split()
        n_lead = 0
        for tok in parts:
            try: int(tok); n_lead += 1
            except ValueError:
                if tok in ("Mo","S","H","C","O","N"): n_lead += 1
                else: break
        floats = []
        for tok in parts[n_lead:]:
            try: floats.append(float(tok))
            except ValueError: break
        for col, new_val in mods:
            if col < len(floats): floats[col] = new_val
        lead = " " + " ".join(parts[:n_lead])
        body = "".join(f"{f:10.4f}" for f in floats)
        lines[ln_idx] = lead + body
    write_lines(dst, lines)


def setup_eval(ff):
    wd = OUT_DIR / "_active"
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
        if wd.exists(): time.sleep(0.2); shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ff, wd / "ffield.reax")
    (wd / "lmp_control").write_text(LMP_CONTROL, encoding="utf-8")
    (wd / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata").write_text(
        DATA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return wd


def run_lmp(deck, wd, tag, timeout=120):
    in_path = wd / f"in.{tag}.lmp"
    in_path.write_text(deck, encoding="utf-8")
    log_path = wd / f"log.{tag}.lammps"
    try:
        p = subprocess.run([LMP,"-in",in_path.name,"-log",log_path.name,"-screen","none"],
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


def strain_deck(tlx, tly):
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
        change_box all x final 0.0 {tlx:.10f} y final 0.0 {tly:.10f} remap units box
        fix freeze all setforce 0.0 0.0 NULL
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        variable sig_iso_gpa equal 0.5*(-pxx-pyy)*{GPA_PER_ATM}*lz/{H_EFF_AA}
        write_data data.eps.lammpsdata
        print "FINAL_SIG_ISO_GPA ${{sig_iso_gpa}}"
        """)


def evaluate(spec, x, src_lines, dft_eps, dft_sig):
    tmp = OUT_DIR / "tmp_ffield.reax"
    write_ffield(src_lines, spec, x, tmp)
    wd = setup_eval(tmp)
    rc, log = run_lmp(init_deck(), wd, "init", timeout=180)
    if rc != 0: return 9999.0, []
    lx0 = grab(log,"RELAXED_LX"); ly0 = grab(log,"RELAXED_LY")
    if lx0 is None: return 9999.0, []
    sigs = []
    for i, eps in enumerate(dft_eps):
        tlx = lx0 * math.exp(float(eps)); tly = ly0 * math.exp(float(eps))
        rc, log = run_lmp(strain_deck(tlx,tly), wd, f"s{i}", timeout=90)
        s = grab(log,"FINAL_SIG_ISO_GPA") if rc == 0 else None
        if s is None or not math.isfinite(s): return 9999.0, sigs
        sigs.append(s)
        shutil.copyfile(wd / "data.eps.lammpsdata", wd / "data.relaxed.lammpsdata")
    sigs = np.array(sigs)
    valid = np.abs(dft_sig) > 1e-6
    if not valid.any(): return 9999.0, sigs.tolist()
    rel = np.abs(sigs - dft_sig) / np.abs(dft_sig) * 100.0
    return float(np.max(rel[valid])), sigs.tolist()


def project_to_bounds(x, lows, highs):
    return np.minimum(np.maximum(x, lows), highs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    n_params = len(spec)
    lows = np.array([s["lo"] for s in spec])
    highs = np.array([s["hi"] for s in spec])
    x0 = np.array([s["init"] for s in spec])
    print(f"[setup] v9: src={SRC_FFIELD.name}, {n_params} params, "
          f"NM(large simplex)->Powell, target<{TARGET_MAX_RE}%", flush=True)
    print(f"[setup] x0 in bounds: {(x0 >= lows).all() and (x0 <= highs).all()}",
          flush=True)
    print(f"[setup] new params (curvature): "
          f"{[s['name'] for s in spec if s['name'] in ('p_be2_MoS','p_bo3_MoS','p_bo4_MoS')]}",
          flush=True)

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)
    print(f"[setup] DFT sigs: {[round(s,2) for s in dft_sig]}", flush=True)

    if not LOG_PATH.exists():
        header = ("trial,phase,timestamp,max_re," +
                  ",".join(s["name"] for s in spec) + ",sigs\n")
        LOG_PATH.write_text(header, encoding="utf-8")

    best = {"max_re": float("inf"), "params": None, "sigs": None, "phase": None}
    tc = [0]
    phase = ["NM"]

    def objective(x):
        x_proj = project_to_bounds(np.asarray(x), lows, highs)
        tc[0] += 1
        t0 = time.time()
        max_re, sigs = evaluate(spec, x_proj, src, dft_eps, dft_sig)
        rt = time.time() - t0
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{tc[0]},{phase[0]},{ts},{max_re:.3f}," +
                     ",".join(f"{v:.6f}" for v in x_proj) + f",\"{sigs_str}\"\n")
        marker = ""
        if max_re < best["max_re"]:
            best["max_re"] = max_re; best["params"] = list(x_proj)
            best["sigs"] = sigs; best["phase"] = phase[0]
            marker = " <- NEW BEST"
            write_ffield(src, spec, x_proj, DST_FFIELD)
            BEST_PATH.write_text(
                f"phase {phase[0]} trial {tc[0]}\n"
                f"max_re = {max_re:.2f}%\n"
                f"sigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                f"params = {dict(zip([s['name'] for s in spec], list(x_proj)))}\n",
                encoding="utf-8")
        print(f"{phase[0]} t{tc[0]:4d} [{rt:5.1f}s] max_re={max_re:7.2f}% "
              f"best={best['max_re']:7.2f}%{marker}", flush=True)
        if max_re <= TARGET_MAX_RE:
            raise StopIteration
        return max_re

    bounds_pairs = [(float(lo), float(hi)) for lo, hi in zip(lows, highs)]

    # Large initial simplex: 15% steps to escape v8 basin
    initial_simplex = np.tile(x0, (n_params + 1, 1)).astype(float)
    for i in range(n_params):
        step = 0.15 * abs(x0[i]) if abs(x0[i]) > 1e-3 else 0.05
        if x0[i] >= 0:
            initial_simplex[i + 1, i] = min(highs[i], x0[i] + step)
        else:
            initial_simplex[i + 1, i] = max(lows[i], x0[i] - step)

    print(f"[phase NM] large simplex (15% step), max 2000 iters", flush=True)
    try:
        minimize(objective, x0, method="Nelder-Mead",
                 bounds=bounds_pairs,
                 options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3,
                          "adaptive": True, "initial_simplex": initial_simplex})
    except StopIteration:
        print("[phase NM] target met!", flush=True)
    except Exception as e:
        print(f"[phase NM] error: {e}", flush=True)

    if best["max_re"] > TARGET_MAX_RE and best["params"] is not None:
        print(f"[phase Powell] from best={best['max_re']:.2f}%", flush=True)
        phase[0] = "POW"
        try:
            minimize(objective, np.array(best["params"]), method="Powell",
                     bounds=bounds_pairs,
                     options={"maxiter": 2000, "xtol": 1e-4, "ftol": 1e-3})
        except StopIteration:
            print("[phase Powell] target met!", flush=True)
        except Exception as e:
            print(f"[phase Powell] error: {e}", flush=True)

    if best["max_re"] > TARGET_MAX_RE and best["params"] is not None:
        print(f"[phase NM2] tighter simplex from best={best['max_re']:.2f}%", flush=True)
        phase[0] = "NM2"
        xb = np.array(best["params"])
        simplex2 = np.tile(xb, (n_params + 1, 1)).astype(float)
        for i in range(n_params):
            step = 0.05 * abs(xb[i]) if abs(xb[i]) > 1e-3 else 0.01
            if xb[i] + step <= highs[i]:
                simplex2[i + 1, i] = xb[i] + step
            else:
                simplex2[i + 1, i] = max(lows[i], xb[i] - step)
        try:
            minimize(objective, xb, method="Nelder-Mead",
                     bounds=bounds_pairs,
                     options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-3,
                              "adaptive": True, "initial_simplex": simplex2})
        except StopIteration:
            print("[phase NM2] target met!", flush=True)
        except Exception as e:
            print(f"[phase NM2] error: {e}", flush=True)

    print(f"\nFinal: max_re={best['max_re']:.2f}%  phase={best['phase']}", flush=True)


if __name__ == "__main__":
    main()
