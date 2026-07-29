#!/usr/bin/env python3
"""AE T7 prep — read-only timing-provenance scanner.

Simulator-mode runs record T_auc = 0 by construction (train_darts.py zeroes
auction wall-clock in simulator mode). Nonzero auction-timing columns therefore
mark deployed / timing-bearing cells for curation into data/raw/timing_auction
and data/raw/agent_scaling versus behavioral.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from util_csv import safe_read_csv  # type: ignore
except ImportError:  # pragma: no cover — inline fallback
    def safe_read_csv(path, **kwargs):  # type: ignore[misc]
        path = Path(path)
        if not path.exists():
            return None
        base = dict(engine="python", on_bad_lines="skip")
        base.update(kwargs)
        try:
            df = pd.read_csv(path, **base)
            return df if not df.empty else None
        except Exception as e:
            sys.stderr.write(f"[safe_read_csv] {path}: {e}\n")
            return None

TIMING_COLS = ("max_t_auc_ms", "mean_t_auc_ms", "max_delta_auc_ms")


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col].replace("N/A", pd.NA)
    return pd.to_numeric(s, errors="coerce")


def scan_file(path: Path, min_rows: int) -> Optional[Dict[str, Any]]:
    df = safe_read_csv(path)
    if df is None:
        return None
    df = df.copy()
    df.columns = df.columns.str.strip()
    if len(df) < min_rows:
        return None

    present = [c for c in TIMING_COLS if c in df.columns]
    missing = [c for c in TIMING_COLS if c not in df.columns]

    max_t = _numeric_series(df, "max_t_auc_ms")
    mean_t = _numeric_series(df, "mean_t_auc_ms")
    max_d = _numeric_series(df, "max_delta_auc_ms")

    max_max_t = float(max_t.max()) if len(max_t) and max_t.notna().any() else 0.0
    n_mean_nz = int((mean_t.fillna(0.0) != 0.0).sum()) if len(mean_t) else 0
    max_max_d = float(max_d.max()) if len(max_d) and max_d.notna().any() else 0.0

    # Missing columns treated as zero for provenance.
    if "max_t_auc_ms" not in df.columns:
        max_max_t = 0.0
    if "mean_t_auc_ms" not in df.columns:
        n_mean_nz = 0
    if "max_delta_auc_ms" not in df.columns:
        max_max_d = 0.0

    ep = _numeric_series(df, "episode")
    if len(ep) and ep.notna().any():
        ep_min, ep_max = float(ep.min()), float(ep.max())
        ep_range = f"{ep_min:g}-{ep_max:g}"
    else:
        ep_range = "n/a"

    mean_has_nz = bool(len(mean_t) and (mean_t.fillna(0.0) != 0.0).any()) if "mean_t_auc_ms" in df.columns else False
    max_has_nz = bool(max_max_t != 0.0) if "max_t_auc_ms" in df.columns else False
    timing_bearing = max_has_nz or mean_has_nz

    return {
        "path": str(path),
        "n_rows": len(df),
        "ep_range": ep_range,
        "max_max_t_auc_ms": max_max_t,
        "n_mean_t_auc_nz": n_mean_nz,
        "max_max_delta_auc_ms": max_max_d,
        "timing_bearing": timing_bearing,
        "missing_cols": missing,
        "present_cols": present,
    }


def iter_log_csvs(root: Path) -> List[Path]:
    """Both exp_*/B*/<stress>/ and legacy <exp_name>/ layouts via recursive glob."""
    return sorted(p for p in root.rglob("*_log.csv") if p.is_file())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more result roots to scan",
    )
    ap.add_argument(
        "--min_rows",
        type=int,
        default=1,
        help="Skip CSVs with fewer than this many data rows (default: 1)",
    )
    args = ap.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    missing_schema: List[Tuple[str, List[str]]] = []

    for root in args.roots:
        root = root.resolve()
        if not root.is_dir():
            sys.stderr.write(f"[warn] not a directory, skipping: {root}\n")
            continue
        for path in iter_log_csvs(root):
            rec = scan_file(path, args.min_rows)
            if rec is None:
                continue
            rows.append(rec)
            if rec["missing_cols"]:
                missing_schema.append((rec["path"], list(rec["missing_cols"])))

    n_total = len(rows)
    bearing = [r for r in rows if r["timing_bearing"]]
    allzero = [r for r in rows if not r["timing_bearing"]]
    n_nonzero = len(bearing)
    n_allzero = len(allzero)

    # Sorted table of timing-bearing files.
    bearing_sorted = sorted(
        bearing,
        key=lambda r: (
            -r["max_max_t_auc_ms"],
            -r["n_mean_t_auc_nz"],
            -r["max_max_delta_auc_ms"],
            r["path"],
        ),
    )

    if bearing_sorted:
        hdr = (
            f"{'max_t_auc':>12}  {'n_mean_nz':>9}  {'max_delta':>12}  "
            f"{'rows':>6}  {'episodes':>14}  path"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in bearing_sorted:
            print(
                f"{r['max_max_t_auc_ms']:12.4f}  "
                f"{r['n_mean_t_auc_nz']:9d}  "
                f"{r['max_max_delta_auc_ms']:12.4f}  "
                f"{r['n_rows']:6d}  "
                f"{r['ep_range']:>14}  "
                f"{r['path']}"
            )
        print()
    else:
        print("(no timing-bearing files)\n")

    print(
        f"{n_nonzero} timing-bearing files / {n_total} scanned; {n_allzero} all-zero"
    )

    # Legacy / incomplete schemas (any of the three timing columns absent).
    if missing_schema:
        print()
        print(f"Files lacking timing columns ({len(missing_schema)}):")
        for path, miss in sorted(missing_schema):
            print(f"  missing={miss}  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
