#!/usr/bin/env python3
"""Manual recovery of 63-col rows from corrupted DARTS sweep CSVs.

The new sweep's B1/{light,medium,heavy}/{darts, darts_nocert, cea_edf, sadcher}
workspace logs have mixed-schema rows ({62, 59, 63}). pandas with on_bad_lines='skip'
keeps the header-matching 62-col rows and drops the 63-col rows that carry
n_miss_n3_unreached and n_miss_full_fidelity_late.

This script line-parses each corrupted CSV, picks out 63-field lines, parses them
with a canonical 63-col header read from a known-clean file, and aggregates the
N3 / FFL breakdown.

Outputs Markdown to stdout. Run:
    python3 scripts/recover_63col.py
"""
from __future__ import annotations
import argparse
import glob, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
NEW_SWEEP = REPO_ROOT / "results_a1a2b1_20260521"

# A file that diagnose_field_counts() showed has 63 cols ONLY.
# Adjust if your environment differs. Recomputed when --new_sweep_dir is set.
CANONICAL_CLEAN = NEW_SWEEP / "exp_1" / "B1" / "energy_b02" / "darts_log.csv"

# Cells to recover: only the corrupted ones (B1 light/medium/heavy × 4 methods).
# darts_no_incentive cells are already clean (already covered by Prompt #16).
RECOVERY_CELLS = [
    (stress, method)
    for stress in ('light', 'medium', 'heavy')
    for method in ('darts', 'darts_nocert', 'cea_edf', 'sadcher')
]

EXPECTED_N_COLS = 63


def load_canonical_header() -> list[str]:
    """Read the first line of the canonical clean file as the 63-col header."""
    if not CANONICAL_CLEAN.exists():
        raise FileNotFoundError(
            f"Canonical clean file missing: {CANONICAL_CLEAN}. "
            f"Pick another clean cell (any with {{63: N}} only in the Prompt #12 diagnostic) "
            f"and update CANONICAL_CLEAN."
        )
    with open(CANONICAL_CLEAN) as f:
        header_line = f.readline().rstrip('\n').rstrip('\r')
    cols = header_line.split(',')
    if len(cols) != EXPECTED_N_COLS:
        raise ValueError(
            f"Expected {EXPECTED_N_COLS} cols in {CANONICAL_CLEAN}, got {len(cols)}. "
            f"Pick a different canonical-clean file."
        )
    return cols


def extract_63col_data_rows(path: Path, header: list[str]) -> list[dict]:
    """Yield dict rows for all 63-col DATA lines in path (skip header-row matches)."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            fields = line.split(',')
            if len(fields) != EXPECTED_N_COLS:
                continue
            # Skip rows that ARE a 63-col header line (first field is 'episode')
            if fields[0].strip().lower() == 'episode':
                continue
            try:
                rows.append(dict(zip(header, fields)))
            except Exception:
                continue
    return rows


def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Recover 63-col N3/FFL breakdown from mixed-schema CSVs.")
    ap.add_argument(
        "--new_sweep_dir",
        type=str,
        default=str(REPO_ROOT / "results_a1a2b1_20260521"),
        help="Root for NEW_SWEEP globs / canonical header (default: <repo_root>/results_a1a2b1_20260521)",
    )
    args = ap.parse_args()
    global NEW_SWEEP, CANONICAL_CLEAN
    NEW_SWEEP = Path(args.new_sweep_dir).expanduser().resolve()
    CANONICAL_CLEAN = NEW_SWEEP / "exp_1" / "B1" / "energy_b02" / "darts_log.csv"
    print(f"[recover_63col] NEW_SWEEP={NEW_SWEEP}", file=sys.stderr)
    print(f"[recover_63col] CANONICAL_CLEAN={CANONICAL_CLEAN}", file=sys.stderr)

    header = load_canonical_header()
    # Verify the four key columns exist in the canonical header
    needed = [
        'task_miss_rate',
        'n_miss_n1_drop',
        'n_miss_n2_fallback_insufficient',
        'n_miss_n3_unreached',
        'n_miss_full_fidelity_late',
    ]
    missing = [c for c in needed if c not in header]
    if missing:
        sys.stderr.write(
            f"⚠ canonical header is missing required columns: {missing}\n"
            f"   header = {header}\n"
        )
        sys.exit(2)

    col_idx = {c: header.index(c) for c in needed}

    print("# Recovered N3 / FFL breakdown for corrupted B1 cells")
    print()
    try:
        canon_disp = str(CANONICAL_CLEAN.relative_to(REPO_ROOT))
    except ValueError:
        canon_disp = str(CANONICAL_CLEAN)
    print(f"Canonical header source: `{canon_disp}`")
    print(f"Method: line-by-line parse of mixed-schema CSVs; keep rows with exactly 63 fields;")
    print(f"        re-parse via canonical header; aggregate across recovered rows × seeds.")
    print()
    print("| Stress | Method | n_files | n_recovered_rows | task_miss% | N1 drop | N2 fallback-insuf. | N3 unreached | FF late |")
    print("|---|---|---|---|---|---|---|---|---|")

    for stress, method in RECOVERY_CELLS:
        pattern = f"{NEW_SWEEP}/exp_*/B1/{stress}/{method}_log.csv"
        files = sorted(glob.glob(pattern))
        all_rows = []
        n_files_with_rows = 0
        for p in files:
            rows = extract_63col_data_rows(Path(p), header)
            if rows:
                all_rows.extend(rows)
                n_files_with_rows += 1
        if not all_rows:
            print(f"| {stress} | {method} | {len(files)} | 0 | — | — | — | — | — |")
            continue
        # Aggregate
        def avg(col):
            vals = [to_float(r.get(col, '')) for r in all_rows]
            vals = [v for v in vals if v is not None]
            return np.mean(vals) if vals else None

        miss_pct = avg('task_miss_rate')
        n1 = avg('n_miss_n1_drop')
        n2 = avg('n_miss_n2_fallback_insufficient')
        n3 = avg('n_miss_n3_unreached')
        ffl = avg('n_miss_full_fidelity_late')

        def fmt(v, pct=False):
            if v is None:
                return '—'
            return f"{v*100:.1f}" if pct else f"{v:.3f}"

        print(f"| {stress} | {method} | {n_files_with_rows}/{len(files)} | {len(all_rows)} | "
              f"{fmt(miss_pct, pct=True)} | {fmt(n1)} | {fmt(n2)} | {fmt(n3)} | {fmt(ffl)} |")


if __name__ == '__main__':
    main()
