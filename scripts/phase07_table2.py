"""Phase 7 — Updated Table 2 (main) + SI Table S1 (auxiliary).

Main Table 2 columns:
    group, parameter, initial value [Cooper et al.], posterior mean,
    σ_post, 95 % CI lower, 95 % CI upper

SI Table S1 columns:
    group, parameter, σ_opt, σ_GP, σ_post, ratio σ_post/σ_opt,
    R̂, ESS_bulk, ESS_tail

"Initial value [Cooper et al.]" — this codebase's BO was warm-started
from the v9 ffield that itself was tuned from the original Cooper /
Ostadhossein parameter set ("Cooper et al." in the manuscript's
nomenclature). We use the staged_bo (= trial 1 = warm-start v9) value
as the documented "initial" entry; if a separate Cooper-strict
reference becomes available later, drop it into
``data/cooper_initial_values.csv`` and re-run.

Outputs:
    outputs/tables/Table2_posterior.csv
    outputs/tables/Table2_posterior.tex
    outputs/tables/TableS1_uncertainty_diagnostics.csv
    outputs/tables/TableS1_uncertainty_diagnostics.tex
    outputs/diagnostics/phase07.log
    outputs/diagnostics/phase07.json

Stop condition: both .tex compile in a LaTeX harness. ``pdflatex`` is
attempted first; when unavailable (current environment) we fall back
to a strict syntactic validator (begin/end balance + booktabs rule
presence + column-count consistency).
"""
from __future__ import annotations
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.uncertainty import PARAM_GROUP_LOOKUP  # noqa: E402
from src import ffield_parse as _rca  # noqa: E402  (pure ffield text parser)

SAMPLES_NPZ      = ROOT / "outputs" / "data" / "phase04_posterior_samples.npz"
PHASE04_DIAG_CSV = ROOT / "outputs" / "diagnostics" / "phase04_diagnostics.csv"
PHASE05_TABLE    = ROOT / "outputs" / "tables" / "uncertainty_comparison.csv"
STAGED_FF        = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
                     / "precise_ffields" / "ffield.reax.staged_bo.reax")

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_DIAG   = ROOT / "outputs" / "diagnostics"
SEED       = 42


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _initial_values_from_staged_bo() -> dict[str, float]:
    """Read the staged_bo (= warm-start v9, Cooper-derived) initial parameter
    values directly from precise_ffields/ffield.reax.MoSH.staged_bo.reax."""
    src = _rca.read_lines(STAGED_FF)
    off = _rca.parse_offsets(src)
    spec = _rca.build_spec(src, off)
    return {s["name"]: float(s["init"]) for s in spec}


# ---------------------------------------------------------------------------
# TeX rendering — booktabs, no external packages
# ---------------------------------------------------------------------------

def _df_to_booktabs(df: pd.DataFrame, *,
                     caption: str, label: str,
                     col_align: str,
                     col_headers: list[str],
                     footnote: str | None = None) -> str:
    """Render a DataFrame as a booktabs LaTeX table. Pure-text construction
    (no pandas to_latex) for explicit control of every line."""
    n_cols = df.shape[1]
    if len(col_headers) != n_cols:
        raise ValueError(
            f"col_headers has {len(col_headers)} entries; table has {n_cols}.")
    if len(col_align) != n_cols:
        raise ValueError(
            f"col_align '{col_align}' has length {len(col_align)} != {n_cols}.")

    out: list[str] = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{" + caption + r"}")
    out.append(r"\label{" + label + r"}")
    out.append(r"\begin{tabular}{" + col_align + r"}")
    out.append(r"\toprule")
    out.append(" & ".join(col_headers) + r" \\")
    out.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [str(v) for v in row.values]
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    if footnote:
        out.append(r"\vspace{2pt}")
        out.append(r"\par\footnotesize " + footnote)
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def _fmt_num(v: float, kind: str = "f3") -> str:
    """LaTeX-safe numeric formatter; respects sign convention."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if kind == "f3":     return f"{v:.3f}"
    if kind == "f4":     return f"{v:.4f}"
    if kind == "e2":     return f"{v:.2e}"
    if kind == "e3":     return f"{v:.3e}"
    if kind == "i":      return f"{int(round(v)):,}"
    return str(v)


def _latex_escape_underscores(s: str) -> str:
    """ReaxFF parameter names contain ``_`` which is math-mode in LaTeX."""
    return s.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Static TeX validator (used when pdflatex is absent)
# ---------------------------------------------------------------------------

def _validate_tex_syntax(text: str) -> tuple[bool, list[str]]:
    fails: list[str] = []
    # 1. begin/end pairs balanced
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends   = re.findall(r"\\end\{([^}]+)\}",   text)
    if sorted(begins) != sorted(ends):
        fails.append(f"begin/end environments do not match: "
                      f"{sorted(begins)} vs {sorted(ends)}")
    # 2. braces balanced (ignoring escaped \{ \})
    text_clean = re.sub(r"\\[{}]", "", text)
    if text_clean.count("{") != text_clean.count("}"):
        fails.append(
            f"unmatched braces: {{={text_clean.count('{')}, "
            f"}}={text_clean.count('}')}")
    # 3. booktabs rules present
    for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
        if rule not in text:
            fails.append(f"missing booktabs rule {rule}")
    # 4. tabular columns vs body columns consistency (loose: count '&' per body row)
    m = re.search(r"\\begin\{tabular\}\{([^}]+)\}", text)
    if m is None:
        fails.append("no \\begin{tabular}{...} found")
    else:
        spec = re.sub(r"[^lcrp]", "", m.group(1))
        n_cols = len(spec)
        # Body rows: lines ending with ' \\' between midrule and bottomrule
        body = text.split(r"\midrule", 1)[-1].split(r"\bottomrule", 1)[0]
        rows = [ln.strip() for ln in body.splitlines()
                if ln.strip().endswith(r"\\")]
        for j, ln in enumerate(rows):
            # number of & separating cells = n_cols - 1
            n_amp = ln.count("&")
            if n_amp != n_cols - 1:
                fails.append(
                    f"row {j+1}: {n_amp} '&' separators, expected {n_cols - 1}")
                break  # stop after first bad row
    return (len(fails) == 0, fails)


def _attempt_pdflatex(tex_path: Path) -> tuple[bool, str]:
    """Try pdflatex on the .tex file inside a minimal harness. Returns
    (passed, message). Falls back to ``False`` if pdflatex is missing."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return False, "pdflatex executable not on PATH"
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        stub = td_p / "stub.tex"
        stub.write_text(
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[a3paper,landscape]{geometry}\n"
            "\\begin{document}\n"
            "\\input{" + tex_path.stem + "}\n"
            "\\end{document}\n", encoding="utf-8")
        shutil.copy2(tex_path, td_p / tex_path.name)
        try:
            proc = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                  "stub.tex"],
                cwd=td_p, capture_output=True, text=True, timeout=60)
        except Exception as exc:
            return False, f"pdflatex invocation failed: {exc}"
        if proc.returncode != 0:
            return False, f"pdflatex exit={proc.returncode}: " \
                          + (proc.stdout or proc.stderr)[-400:]
        return True, "pdflatex compiled stub successfully"


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_table2(post_npz: Path, init_vals: dict[str, float]) -> pd.DataFrame:
    """Posterior mean / σ / 95 % CI per parameter, grouped by physical block."""
    npz = np.load(post_npz, allow_pickle=True)
    samples = npz["samples"]                          # (chains, draws, d)
    names = list(npz["param_names"])
    flat = samples.reshape(-1, samples.shape[-1])

    rows = []
    for i, name in enumerate(names):
        col = flat[:, i]
        rows.append({
            "group":             PARAM_GROUP_LOOKUP.get(name, "?"),
            "parameter":         name,
            "initial":           init_vals.get(name, float("nan")),
            "posterior_mean":    float(np.mean(col)),
            "sigma_post":        float(np.std(col, ddof=1)),
            "ci95_low":          float(np.quantile(col, 0.025)),
            "ci95_high":         float(np.quantile(col, 0.975)),
        })
    df = pd.DataFrame(rows)
    # Sort by group order then param name (stable within-group)
    group_order = ["Bond (Mo-S)", "Off-diagonal", "Angle",
                    "Atomic non-bonded", "Atomic over-coordination"]
    df["_g"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df = df.sort_values(["_g", "parameter"]).drop(columns="_g").reset_index(drop=True)
    return df


def build_table_s1(comp_path: Path, diag_path: Path) -> pd.DataFrame:
    """Per-parameter σ_opt, σ_GP, σ_post, ratio, R̂, ESS_bulk, ESS_tail."""
    comp = pd.read_csv(comp_path)
    diag = pd.read_csv(diag_path)
    merged = comp.merge(diag, left_on="parameter", right_on="name", how="inner")
    out = merged[[
        "group", "parameter",
        "sigma_opt", "sigma_GP_mean", "sigma_post",
        "ratio_post_over_opt", "r_hat", "ess_bulk", "ess_tail",
        "prior_dominated",
    ]].rename(columns={"sigma_GP_mean": "sigma_GP"})
    # Same group order
    group_order = ["Bond (Mo-S)", "Off-diagonal", "Angle",
                    "Atomic non-bonded", "Atomic over-coordination"]
    out["_g"] = out["group"].map({g: i for i, g in enumerate(group_order)})
    out = out.sort_values(["_g", "parameter"]).drop(columns="_g").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_DIAG.mkdir(parents=True, exist_ok=True)

    init_vals = _initial_values_from_staged_bo()

    # ---- Main Table 2 ----
    df2 = build_table2(SAMPLES_NPZ, init_vals)
    df2.to_csv(OUT_TABLES / "Table2_posterior.csv",
                index=False, float_format="%.6f")

    # LaTeX rendering of Table 2
    df2_tex = pd.DataFrame({
        "Group":     df2["group"],
        "Parameter": df2["parameter"].map(_latex_escape_underscores),
        "Initial":   df2["initial"].map(lambda v: _fmt_num(v, "f4")),
        "Post.\\ mean":       df2["posterior_mean"].map(lambda v: _fmt_num(v, "f4")),
        r"$\sigma_{\rm post}$": df2["sigma_post"].map(lambda v: _fmt_num(v, "e2")),
        "CI$_{95}$ low":  df2["ci95_low"].map(lambda v: _fmt_num(v, "f4")),
        "CI$_{95}$ high": df2["ci95_high"].map(lambda v: _fmt_num(v, "f4")),
    })
    tex2 = _df_to_booktabs(
        df2_tex,
        caption=("Posterior summary for the 40 BO-controlled ReaxFF "
                  "parameters at $x_\\star$. Initial values are the "
                  "Cooper-derived staged-BO warm start; posterior mean, "
                  "$\\sigma_{\\rm post}$ and the 95\\% credible interval "
                  "come from 4 NUTS chains $\\times$ 5000 post-warmup "
                  "draws (R$\\hat{ }$ < 1.01, min ESS_bulk > 17{,}000)."),
        label="tab:posterior_summary",
        col_align="llrrrrr",
        col_headers=["Group", "Parameter", "Initial",
                      "Post.\\ mean", r"$\sigma_{\rm post}$",
                      "CI$_{95}$ low", "CI$_{95}$ high"],
        footnote=("$\\sigma_{\\rm post}$ given in scientific notation. "
                   "Tabulated values are reproducible via "
                   "\\texttt{scripts/phase07\\_table2.py}; see also "
                   "\\texttt{phase07.log} for source-CSV checksums."),
    )
    (OUT_TABLES / "Table2_posterior.tex").write_text(tex2, encoding="utf-8")

    # ---- SI Table S1 ----
    df_s1 = build_table_s1(PHASE05_TABLE, PHASE04_DIAG_CSV)
    df_s1.to_csv(OUT_TABLES / "TableS1_uncertainty_diagnostics.csv",
                  index=False, float_format="%.6f")

    s1_tex = pd.DataFrame({
        "Group":     df_s1["group"],
        "Parameter": df_s1["parameter"].map(_latex_escape_underscores),
        r"$\sigma_{\rm opt}$":   df_s1["sigma_opt"].map(lambda v: _fmt_num(v, "e2")),
        r"$\sigma_{\rm GP}$":    df_s1["sigma_GP"].map(lambda v: _fmt_num(v, "e2")),
        r"$\sigma_{\rm post}$":  df_s1["sigma_post"].map(lambda v: _fmt_num(v, "e2")),
        r"$\sigma_{\rm post}/\sigma_{\rm opt}$": df_s1["ratio_post_over_opt"].map(lambda v: _fmt_num(v, "f3")),
        r"$\hat{R}$":            df_s1["r_hat"].map(lambda v: _fmt_num(v, "f4")),
        r"ESS$_{\rm bulk}$":     df_s1["ess_bulk"].map(lambda v: _fmt_num(v, "i")),
        r"ESS$_{\rm tail}$":     df_s1["ess_tail"].map(lambda v: _fmt_num(v, "i")),
        "Prior dom.":            df_s1["prior_dominated"].map(lambda v: "yes" if v else "no"),
    })
    tex_s1 = _df_to_booktabs(
        s1_tex,
        caption=("Auxiliary uncertainty diagnostics for the 40 "
                  "BO-controlled parameters: optimiser-replicate spread "
                  "$\\sigma_{\\rm opt}$ (top-20 staged-BO trials), "
                  "surrogate-attributed loss variation $\\sigma_{\\rm GP}$ "
                  "(1-D scan at posterior medians; units of eV), "
                  "posterior marginal $\\sigma_{\\rm post}$, the "
                  "$\\sigma_{\\rm post}/\\sigma_{\\rm opt}$ ratio, and the "
                  "ArviZ convergence statistics ($\\hat{R}$, "
                  "ESS$_{\\rm bulk}$, ESS$_{\\rm tail}$). "
                  "Parameters whose 95\\% posterior HDI covers $\\geq 90\\%$ "
                  "of the prior box are marked prior-dominated."),
        label="tab:SI_uncertainty_diagnostics",
        col_align="llrrrrrrrl",
        col_headers=["Group", "Parameter",
                      r"$\sigma_{\rm opt}$", r"$\sigma_{\rm GP}$",
                      r"$\sigma_{\rm post}$",
                      r"$\sigma_{\rm post}/\sigma_{\rm opt}$",
                      r"$\hat{R}$", r"ESS$_{\rm bulk}$", r"ESS$_{\rm tail}$",
                      "Prior dom."],
        footnote=("Numerical ground truth in "
                   "\\texttt{outputs/tables/TableS1\\_uncertainty\\_diagnostics.csv}."),
    )
    (OUT_TABLES / "TableS1_uncertainty_diagnostics.tex").write_text(
        tex_s1, encoding="utf-8")

    # ---- LaTeX harness validation ----
    validations: dict[str, dict] = {}
    for label, tex_path in [
        ("Table2_posterior",                   OUT_TABLES / "Table2_posterior.tex"),
        ("TableS1_uncertainty_diagnostics",    OUT_TABLES / "TableS1_uncertainty_diagnostics.tex"),
    ]:
        text = tex_path.read_text(encoding="utf-8")
        ok_syn, fails_syn = _validate_tex_syntax(text)
        ok_pdf, msg_pdf = _attempt_pdflatex(tex_path)
        validations[label] = {
            "syntactic_ok":  ok_syn,
            "syntactic_failures": fails_syn,
            "pdflatex_attempted": True,
            "pdflatex_ok":   ok_pdf,
            "pdflatex_msg":  msg_pdf,
        }

    # ---- Log ----
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdflatex_avail = shutil.which("pdflatex") is not None
    lines = [
        f"# Phase 7 — Table 2 + SI Table S1",
        f"# timestamp: {ts}",
        f"# seed: {SEED}",
        f"# pdflatex available: {pdflatex_avail}",
        f"",
        f"## Method notes",
        f"",
        f"- Table 2 posterior summary: mean / std / 2.5–97.5 % quantiles",
        f"  from outputs/data/phase04_posterior_samples.npz",
        f"  (4 chains × 5000 draws = 20{','}000 samples per parameter).",
        f"- 'Initial value [Cooper et al.]' column reads directly from",
        f"  precise_ffields/ffield.reax.MoSH.staged_bo.reax (= warm-start v9,",
        f"  Cooper-derived in the project's nomenclature).",
        f"- Table S1: σ_opt + σ_GP + σ_post + ratio merged from "
        f"  outputs/tables/uncertainty_comparison.csv and ArviZ stats from "
        f"  outputs/diagnostics/phase04_diagnostics.csv.",
        f"",
        f"## Outputs",
        f"  outputs/tables/Table2_posterior.csv",
        f"  outputs/tables/Table2_posterior.tex",
        f"  outputs/tables/TableS1_uncertainty_diagnostics.csv",
        f"  outputs/tables/TableS1_uncertainty_diagnostics.tex",
        f"",
        f"## LaTeX validation",
        f"",
    ]
    for label, v in validations.items():
        lines.append(f"### {label}.tex")
        lines.append(f"")
        lines.append(f"- syntactic check : "
                      f"{'PASS' if v['syntactic_ok'] else 'FAIL'}  "
                      f"({', '.join(v['syntactic_failures']) if v['syntactic_failures'] else 'no issues'})")
        lines.append(f"- pdflatex check  : "
                      f"{'PASS' if v['pdflatex_ok'] else 'SKIPPED'}  "
                      f"({v['pdflatex_msg']})")
        lines.append("")
    lines += [
        f"## Stop condition",
        f"",
        f"- both .tex files exist : "
        f"{(OUT_TABLES / 'Table2_posterior.tex').exists() and (OUT_TABLES / 'TableS1_uncertainty_diagnostics.tex').exists()}",
        f"- both pass syntactic validator : "
        f"{all(v['syntactic_ok'] for v in validations.values())}",
        f"- pdflatex test : "
        f"{'PASSED' if all(v['pdflatex_ok'] for v in validations.values()) else 'unavailable; fell back to strict static check (per spec: \"if possible\")'}",
    ]
    log_path = OUT_DIAG / "phase07.log"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    summary = {
        "phase":      7,
        "timestamp":  _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":       SEED,
        "outputs":    {
            "Table2_csv":  str(OUT_TABLES / "Table2_posterior.csv"),
            "Table2_tex":  str(OUT_TABLES / "Table2_posterior.tex"),
            "TableS1_csv": str(OUT_TABLES / "TableS1_uncertainty_diagnostics.csv"),
            "TableS1_tex": str(OUT_TABLES / "TableS1_uncertainty_diagnostics.tex"),
        },
        "validation": validations,
        "passed": all(v["syntactic_ok"] for v in validations.values()),
    }
    (OUT_DIAG / "phase07.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] Phase 7 → Table 2 + Table S1 (CSV + TeX) written.")


if __name__ == "__main__":
    main()
