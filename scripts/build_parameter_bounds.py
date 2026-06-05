"""Persist the WIDEN parameter bounds (from staged_bo.reax + recalib spec)
to data/parameter_bounds.csv for downstream HMC and uncertainty phases.

One row per BO-controlled parameter with columns:
    name, lo, hi, init (= staged_bo value)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ffield_parse as _rca  # noqa: E402  (pure ffield text parser)

STAGED_FF = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
              / "precise_ffields" / "ffield.reax.staged_bo.reax")
OUT_CSV   = ROOT / "data" / "parameter_bounds.csv"


def main():
    src = _rca.read_lines(STAGED_FF)
    off = _rca.parse_offsets(src)
    spec = _rca.build_spec(src, off)
    rows = [{
        "name": s["name"],
        "lo":   float(s["lo"]),
        "hi":   float(s["hi"]),
        "init": float(s["init"]),
    } for s in spec]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.8f")
    print(f"[csv] {OUT_CSV.relative_to(ROOT)}  ({len(df)} params)")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
