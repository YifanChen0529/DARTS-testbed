#!/usr/bin/env python3
"""Package integrity gate for `make verify` (Supp. Table III).

Recomputes the Supp. Table III miss-rate grid from a curated behavioral tree
(default: conv-window semantics over episodes {4500, 5000}) and checks it
against the packaged reference values. Exit 0 iff all 40 cells PASS.

Two extraction modes (--semantics):
  lastrow     — duplicate scripts/summarize_sweep.py last-row semantics
  convwindow  — per-seed mean of task_miss_rate over episodes {4500, 5000}
                after clean_df (default; matches the packaged reference grid)

Directory layouts (--layout):
  exp     — results_dir/exp_*/B*/{stress}/{method}_log.csv (default)
  legacy  — results_dir/{method}_{workload_id}_{stress}_seed{seed}/{method}_log.csv
  both    — evaluate each layout separately; print side-by-side (exp | legacy)

Optional --min_mtime_file restores the old LAUNCH_REF gate for auditing the
polluted live results/ root only; curated trees never need it.

--cell BENCH:STRESS:METHOD fingerprint mode lists per-file provenance and
optionally searches n_expected-sized subsets whose conv-window mean rounds to
the reference value.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from util_csv import safe_read_csv  # type: ignore
except ImportError:  # pragma: no cover
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

# Packaged reference grid (round(mean, 3)) — what curated sweep_d1 must reproduce.
# Rows: (benchmark, stress); cols order = METHODS.
# None ⇒ "---" (not run): PASS iff zero matching CSVs.
METHODS = (
    "darts",
    "darts_nocert",
    "darts_no_incentive",
    "cea_edf",
    "sadcher",
)

REFERENCE: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {
    ("B1", "light"): {
        "darts": 0.005,
        "darts_nocert": 0.005,
        "darts_no_incentive": 0.005,
        "cea_edf": 0.010,
        "sadcher": 0.015,
    },
    ("B1", "medium"): {
        "darts": 0.095,
        "darts_nocert": 0.075,
        "darts_no_incentive": 0.085,
        "cea_edf": 0.190,
        "sadcher": 0.315,
    },
    ("B1", "heavy"): {
        "darts": 0.170,
        "darts_nocert": 0.165,
        "darts_no_incentive": 0.175,
        "cea_edf": 0.610,
        "sadcher": 0.645,
    },
    ("B2", "light"): {
        "darts": 0.030,
        "darts_nocert": 0.033,
        "darts_no_incentive": None,
        "cea_edf": 0.125,
        "sadcher": 0.005,
    },
    ("B2", "medium"): {
        "darts": 0.030,
        "darts_nocert": 0.030,
        "darts_no_incentive": 0.030,
        "cea_edf": 0.135,
        "sadcher": 0.010,
    },
    ("B2", "heavy"): {
        "darts": 0.030,
        "darts_nocert": 0.030,
        "darts_no_incentive": None,
        "cea_edf": 0.140,
        "sadcher": 0.035,
    },
    ("B3", "medium"): {
        "darts": 0.015,
        "darts_nocert": None,
        "darts_no_incentive": None,
        "cea_edf": 0.185,
        "sadcher": 0.290,
    },
    ("B4", "medium"): {
        "darts": 0.010,
        "darts_nocert": None,
        "darts_no_incentive": None,
        "cea_edf": 0.010,
        "sadcher": 0.000,
    },
}

# Default expected seed count is 10; only B2/light/darts_nocert is partial (exp_1..3).
EXPECTED_N: Dict[Tuple[str, str, str], int] = {
    ("B2", "light", "darts_nocert"): 3,
}

# Legacy flat-dir naming: {method}_{workload_id}_{stress}_seed{seed}/{method}_log.csv
WORKLOAD_TO_BENCH = {
    "B1_CDR": "B1",
    "B2_LML": "B2",
    "B3_IQI": "B3",
    "B4_DEM": "B4",
}
LEGACY_DIR_RE = re.compile(
    r".*/(?P<method>.+)_(?P<wl>B[1-4]_(?:CDR|LML|IQI|DEM))"
    r"_(?P<stress>\w+)_seed(?P<seed>\d+)/(?P<mfile>.+)_log\.csv$"
)

CONV_EPS = {4500, 5000}
SUBSET_SEARCH_CAP = 5000


def clean_df(path: Path) -> Optional[pd.DataFrame]:
    """Inline copy of fig0/fig1/fig2 clean_df.

    Cross-ref:
      scripts/fig0_headline_bar.py:64-74
      scripts/fig1_b1_grid.py:79-89
      scripts/fig2_generalization_scaling.py:94-104
    """
    df = safe_read_csv(path)
    if df is None or df.empty or "episode" not in df.columns:
        return None
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    df = df.drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)
    df = df[df["episode"] <= 5000]
    return df


def _conv_window_seed_mean(path: str) -> Optional[float]:
    """Per-seed mean of task_miss_rate over episodes in {4500, 5000} after clean_df.

    No PARTIAL fallback here — Table III cells that lack the window contribute
    no seed value (skipped). "N/A" → NaN via to_numeric errors='coerce'.
    """
    df = clean_df(Path(path))
    if df is None or "task_miss_rate" not in df.columns:
        return None
    strict = df[df["episode"].isin(sorted(CONV_EPS))]
    if strict.empty:
        return None
    # Replace literal "N/A" then coerce (errors='coerce' also maps "N/A" → NaN).
    s = strict["task_miss_rate"].replace("N/A", pd.NA)
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _read_last_miss(csv_path: str) -> Tuple[bool, Optional[float]]:
    """Mirror scripts/summarize_sweep.py last-row extraction for task_miss_rate.

    Returns (ok, value). ok=False ⇒ skip file (unreadable / empty), matching
    summarize_sweep's continue paths. ok=True ⇒ count the seed even if value
    is None (groupby mean will skip NaNs; n still counts the row).

    Cross-ref: scripts/summarize_sweep.py records[] loop.
    """
    try:
        df = safe_read_csv(csv_path)
        if df is None:
            return False, None
        df.columns = df.columns.str.strip()
        if len(df) == 0:
            return False, None
        last = df.iloc[-1]
    except Exception as e:
        sys.stderr.write(f"[skip] {csv_path}: {e}\n")
        return False, None

    def f(col: str) -> Optional[float]:
        try:
            v = last[col]
            return (
                float(v)
                if pd.notna(v) and str(v).replace(".", "").replace("-", "").isdigit()
                else None
            )
        except (KeyError, ValueError, TypeError):
            try:
                return float(last[col])
            except Exception:
                return None

    return True, f("task_miss_rate")


def _iter_exp_csvs(
    results_dirs: Sequence[str],
    min_mtime: Optional[float],
) -> Iterable[Tuple[str, str, str, int, str]]:
    """Yield (bench, stress, method, seed, path) for exp_*/B*/*/*_log.csv."""
    for results_dir in results_dirs:
        pattern = f"{results_dir}/exp_*/B*/*/*_log.csv"
        for csv in sorted(glob.glob(pattern)):
            if min_mtime is not None and os.path.getmtime(csv) < min_mtime:
                continue  # OLD v8 leftover (audit-time gate; was LAUNCH_T in summarize)
            m = re.match(r".*/exp_(\d+)/(B\d+)/(\w+)/(\w+)_log\.csv$", csv)
            if not m:
                continue
            exp_idx, bench, stress, method = m.groups()
            seed = int(exp_idx) * 100
            yield bench, stress, method, seed, csv


def _iter_legacy_csvs(
    results_dirs: Sequence[str],
    min_mtime: Optional[float],
) -> Iterable[Tuple[str, str, str, int, str]]:
    """Yield (bench, stress, method, seed, path) for legacy flat dirs under root.

    Layout: {root}/{method}_{workload_id}_{stress}_seed{seed}/{method}_log.csv
    with workload_id ∈ {B1_CDR, B2_LML, B3_IQI, B4_DEM}.
    """
    for results_dir in results_dirs:
        pattern = f"{results_dir}/*_seed*/*_log.csv"
        for csv in sorted(glob.glob(pattern)):
            if min_mtime is not None and os.path.getmtime(csv) < min_mtime:
                continue
            m = LEGACY_DIR_RE.match(csv)
            if not m:
                continue
            method = m.group("method")
            if method != m.group("mfile"):
                continue
            wl = m.group("wl")
            bench = WORKLOAD_TO_BENCH.get(wl)
            if bench is None:
                continue
            stress = m.group("stress")
            seed = int(m.group("seed"))
            yield bench, stress, method, seed, csv


def _iter_layout_csvs(
    results_dirs: Sequence[str],
    min_mtime: Optional[float],
    layout: str,
) -> Iterable[Tuple[str, str, str, int, str]]:
    if layout == "exp":
        yield from _iter_exp_csvs(results_dirs, min_mtime)
    elif layout == "legacy":
        yield from _iter_legacy_csvs(results_dirs, min_mtime)
    else:
        raise ValueError(f"layout must be 'exp' or 'legacy', got {layout!r}")


def collect_cells(
    results_dirs: Sequence[str],
    min_mtime: Optional[float],
    semantics: str,
    layout: str = "exp",
) -> Tuple[
    Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]],
    Dict[Tuple[str, str, str], int],
]:
    """Return (seed→metric map, file counts per cell).

    File counts include every matching CSV (for --- zero-CSV checks). The
    seed→metric map only includes seeds with an extractable value.

    Ambiguity abort applies within one layout across --results_dir roots only;
    --layout both collects layouts separately (cross-layout duplicates OK).
    """
    cells: Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]] = {}
    file_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    seen_keys: Dict[Tuple[str, str, str, int], str] = {}

    for bench, stress, method, seed, csv in _iter_layout_csvs(
        results_dirs, min_mtime, layout
    ):
        key = (bench, stress, method, seed)
        if key in seen_keys:
            raise SystemExit(
                f"ERROR: ambiguous cell {key}:\n"
                f"  {seen_keys[key]}\n"
                f"  {csv}\n"
                f"(same benchmark/stress/method/seed under more than one --results_dir)"
            )
        seen_keys[key] = csv
        file_counts[(bench, stress, method)] += 1

        if semantics == "lastrow":
            ok, miss = _read_last_miss(csv)
            if not ok:
                continue
            cells[key] = (miss, csv)
        else:  # convwindow
            miss = _conv_window_seed_mean(csv)
            if miss is None:
                continue
            cells[key] = (miss, csv)

    return cells, dict(file_counts)


def expected_n(bench: str, stress: str, method: str) -> int:
    return EXPECTED_N.get((bench, stress, method), 10)


def _cell_summary(
    expected: Optional[float],
    vals: List[Optional[float]],
    n_files: int,
    bench: str,
    stress: str,
    method: str,
) -> Tuple[bool, int, str, str, str]:
    """Return (ok, n, found_s, expected_s, n_exp_s) for one layout's cell."""
    if expected is None:
        ok = n_files == 0
        return ok, n_files, "---", "---", "0"

    n_vals = len(vals)
    n_exp = expected_n(bench, stress, method)
    n_exp_s = str(n_exp)
    exp_s = f"{expected:.3f}"
    if n_vals == 0:
        return False, 0, "---", exp_s, n_exp_s
    numeric = [float(v) for v in vals if v is not None and pd.notna(v)]
    if not numeric:
        return False, n_vals, "nan", exp_s, n_exp_s
    mean = sum(numeric) / len(numeric)
    found = round(mean, 3)
    found_s = f"{found:.3f}"
    ok = (found == expected) and (n_vals == n_exp)
    return ok, n_vals, found_s, exp_s, n_exp_s


def _values_by_cell(
    cells: Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]],
) -> Dict[Tuple[str, str, str], List[Optional[float]]]:
    values: Dict[Tuple[str, str, str], List[Optional[float]]] = defaultdict(list)
    for (bench, stress, method, _seed), (miss, _path) in cells.items():
        values[(bench, stress, method)].append(miss)
    return values


def evaluate(
    cells: Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]],
    file_counts: Dict[Tuple[str, str, str], int],
) -> Tuple[List[str], bool]:
    values = _values_by_cell(cells)

    lines: List[str] = []
    lines.append(
        f"{'status':<6}  {'bench':<4}  {'stress':<6}  {'method':<20}  "
        f"{'n':>3}  {'found':>8}  {'expected':>8}  {'n_exp':>5}"
    )
    lines.append("-" * 78)

    all_pass = True
    for (bench, stress) in REFERENCE:
        for method in METHODS:
            expected = REFERENCE[(bench, stress)][method]
            vals = values.get((bench, stress, method), [])
            n_files = file_counts.get((bench, stress, method), 0)
            ok, n, found_s, exp_s, n_exp_s = _cell_summary(
                expected, vals, n_files, bench, stress, method
            )
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            lines.append(
                f"{status:<6}  {bench:<4}  {stress:<6}  {method:<20}  "
                f"{n:3d}  {found_s:>8}  {exp_s:>8}  {n_exp_s:>5}"
            )

    return lines, all_pass


def evaluate_both(
    cells_exp: Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]],
    file_counts_exp: Dict[Tuple[str, str, str], int],
    cells_legacy: Dict[Tuple[str, str, str, int], Tuple[Optional[float], str]],
    file_counts_legacy: Dict[Tuple[str, str, str], int],
) -> Tuple[List[str], bool]:
    """Side-by-side grid: each cell under exp | legacy separately."""
    values_exp = _values_by_cell(cells_exp)
    values_leg = _values_by_cell(cells_legacy)

    lines: List[str] = []
    lines.append(
        f"{'st_exp':<6}  {'st_leg':<6}  {'bench':<4}  {'stress':<6}  {'method':<20}  "
        f"{'n_e':>3}  {'found_exp':>9}  |  {'n_l':>3}  {'found_leg':>9}  "
        f"{'expected':>8}  {'n_exp':>5}"
    )
    lines.append("-" * 110)

    all_pass = True
    for (bench, stress) in REFERENCE:
        for method in METHODS:
            expected = REFERENCE[(bench, stress)][method]
            key = (bench, stress, method)
            ok_e, n_e, found_e, exp_s, n_exp_s = _cell_summary(
                expected,
                values_exp.get(key, []),
                file_counts_exp.get(key, 0),
                bench,
                stress,
                method,
            )
            ok_l, n_l, found_l, _, _ = _cell_summary(
                expected,
                values_leg.get(key, []),
                file_counts_legacy.get(key, 0),
                bench,
                stress,
                method,
            )
            st_e = "PASS" if ok_e else "FAIL"
            st_l = "PASS" if ok_l else "FAIL"
            # --- cells: both layouts must be empty. Else: PASS if ≥1 layout matches.
            if expected is None:
                if not (ok_e and ok_l):
                    all_pass = False
            elif not (ok_e or ok_l):
                all_pass = False
            lines.append(
                f"{st_e:<6}  {st_l:<6}  {bench:<4}  {stress:<6}  {method:<20}  "
                f"{n_e:3d}  {found_e:>9}  |  {n_l:3d}  {found_l:>9}  "
                f"{exp_s:>8}  {n_exp_s:>5}"
            )

    return lines, all_pass


def _raw_episode_range(df: pd.DataFrame) -> Tuple[str, str]:
    if "episode" not in df.columns or df.empty:
        return "n/a", "n/a"
    ep = pd.to_numeric(df["episode"].replace("N/A", pd.NA), errors="coerce").dropna()
    if ep.empty:
        return "n/a", "n/a"
    return f"{float(ep.min()):g}", f"{float(ep.max()):g}"


def fingerprint_cell(
    results_dirs: Sequence[str],
    min_mtime: Optional[float],
    bench: str,
    stress: str,
    method: str,
    root_label: str,
    layout: str = "exp",
) -> Tuple[List[str], int]:
    """Per-file fingerprint for one reference cell; subset search if n > n_exp."""
    published = REFERENCE.get((bench, stress), {}).get(method)
    n_exp = expected_n(bench, stress, method)

    layouts = ("exp", "legacy") if layout == "both" else (layout,)
    files: List[Tuple[str, int, str]] = []  # (layout_tag, seed, path)
    for lay in layouts:
        for b, s, m, seed, csv in _iter_layout_csvs(results_dirs, min_mtime, lay):
            if b == bench and s == stress and m == method:
                files.append((lay, seed, csv))
    files.sort(key=lambda t: (t[0], t[1]))

    lines: List[str] = [
        f"# Fingerprint {bench}:{stress}:{method}  root_label={root_label or '(none)'}",
        f"- results_dir: {', '.join(results_dirs)}",
        f"- layout: {layout}",
        f"- published: {published if published is not None else '---'}",
        f"- n_expected: {n_exp}",
        f"- n_files: {len(files)}",
        "",
        f"{'lay':<6}  {'seed':>5}  {'ncols':>5}  {'nrows':>5}  {'ep_first':>8}  {'ep_last':>8}  "
        f"{'t_auc':>5}  {'rho':>5}  {'conv':>8}  {'bytes':>8}  path",
        "-" * 130,
    ]

    per_file_conv: List[Tuple[int, Optional[float], str]] = []
    for lay, seed, path in files:
        p = Path(path)
        raw = safe_read_csv(path)
        if raw is None:
            lines.append(
                f"{lay:<6}  {seed:5d}  {'ERR':>5}  {'ERR':>5}  {'n/a':>8}  {'n/a':>8}  "
                f"{'?':>5}  {'?':>5}  {'n/a':>8}  {p.stat().st_size if p.exists() else 0:8d}  {path}"
            )
            per_file_conv.append((seed, None, path))
            continue
        raw = raw.copy()
        raw.columns = raw.columns.str.strip()
        ncols = len(raw.columns)
        nrows = len(raw)
        ep0, ep1 = _raw_episode_range(raw)
        has_tauc = "max_t_auc_ms" in raw.columns
        has_rho = "rho_hat" in raw.columns
        conv = _conv_window_seed_mean(path)
        per_file_conv.append((seed, conv, path))
        conv_s = f"{conv:.6f}" if conv is not None else "n/a"
        lines.append(
            f"{lay:<6}  {seed:5d}  {ncols:5d}  {nrows:5d}  {ep0:>8}  {ep1:>8}  "
            f"{'Y' if has_tauc else 'N':>5}  {'Y' if has_rho else 'N':>5}  "
            f"{conv_s:>8}  {p.stat().st_size:8d}  {path}"
        )

    usable = [(seed, v, path) for seed, v, path in per_file_conv if v is not None]
    lines.append("")
    if usable:
        all_mean = sum(v for _, v, _ in usable) / len(usable)
        lines.append(
            f"conv-window mean over ALL {len(usable)} files with values: "
            f"{all_mean:.6f}  round3={round(all_mean, 3)}  "
            f"published={published}"
        )
    else:
        lines.append("conv-window mean over ALL files: (no usable values)")
        all_mean = float("nan")

    # Subset search when file count exceeds expected n.
    if published is not None and len(usable) > n_exp:
        n_combo = 1
        for k in range(n_exp):
            n_combo = n_combo * (len(usable) - k) // (k + 1)
        lines.append("")
        lines.append(
            f"Subset search: C({len(usable)},{n_exp})={n_combo} "
            f"(cap {SUBSET_SEARCH_CAP}); target round3={published:.3f}"
        )
        matches: List[Tuple[Tuple[int, ...], float]] = []
        truncated = False
        if n_combo > SUBSET_SEARCH_CAP:
            lines.append(
                f"WARNING: C({len(usable)},{n_exp})={n_combo} exceeds cap "
                f"{SUBSET_SEARCH_CAP}; enumerating first {SUBSET_SEARCH_CAP} "
                f"combinations only (lexicographic by seed)."
            )
            truncated = True
        count = 0
        for combo in itertools.combinations(usable, n_exp):
            count += 1
            if count > SUBSET_SEARCH_CAP:
                truncated = True
                break
            mean = sum(v for _, v, _ in combo) / n_exp
            if round(mean, 3) == published:
                seeds = tuple(sorted(s for s, _, _ in combo))
                matches.append((seeds, mean))
        lines.append(f"matching subsets: {len(matches)}"
                     + (" (search truncated)" if truncated else ""))
        if matches:
            unique = len(matches) == 1 and not truncated
            lines.append(
                f"match unique: {'YES' if unique else 'NO'}"
                + (" (incomplete search)" if truncated and len(matches) >= 1 else "")
            )
            for seeds, mean in matches:
                lines.append(
                    f"  seeds={list(seeds)}  mean={mean:.6f}  round3={round(mean, 3)}"
                )
                # Also list paths for each matching subset
                seed_set = set(seeds)
                for s, v, path in usable:
                    if s in seed_set:
                        lines.append(f"    exp_seed={s}  conv={v:.6f}  {path}")
        else:
            lines.append("match unique: n/a (zero matching subsets)")
    elif published is not None and len(usable) == n_exp:
        lines.append("")
        lines.append(
            f"File count equals n_expected={n_exp}; "
            f"all-files mean round3={round(all_mean, 3)} vs published={published:.3f} → "
            f"{'MATCH' if round(all_mean, 3) == published else 'NO MATCH'}"
        )
    elif published is not None:
        lines.append("")
        lines.append(
            f"File count {len(usable)} < n_expected={n_exp}; no subset search."
        )

    lines.append("")
    return lines, 0 if usable and published is not None and round(
        sum(v for _, v, _ in usable) / len(usable), 3
    ) == published and len(usable) == n_exp else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir",
        action="append",
        required=True,
        dest="results_dirs",
        help="Root with exp_*/B*/*/*_log.csv and/or legacy flat dirs (repeatable)",
    )
    ap.add_argument(
        "--min_mtime_file",
        type=str,
        default=None,
        help="Skip CSVs older than this file's mtime (live results/ audit only)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional markdown/text output path (default: stdout)",
    )
    ap.add_argument(
        "--semantics",
        choices=("lastrow", "convwindow"),
        default="convwindow",
        help="Extraction semantics (default: convwindow)",
    )
    ap.add_argument(
        "--layout",
        choices=("exp", "legacy", "both"),
        default="exp",
        help="Directory layout to scan (default: exp). "
        "'both' evaluates each layout separately and prints side by side.",
    )
    ap.add_argument(
        "--root_label",
        type=str,
        default="",
        help="Free-text tag echoed in the report header",
    )
    ap.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Fingerprint mode: BENCH:STRESS:METHOD (e.g. B1:medium:darts_nocert)",
    )
    args = ap.parse_args(argv)

    min_mtime: Optional[float] = None
    if args.min_mtime_file:
        min_mtime = os.path.getmtime(args.min_mtime_file)

    if args.cell:
        parts = args.cell.split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"ERROR: --cell must be BENCH:STRESS:METHOD, got {args.cell!r}"
            )
        bench, stress, method = parts
        lines, rc = fingerprint_cell(
            args.results_dirs,
            min_mtime,
            bench,
            stress,
            method,
            args.root_label,
            layout=args.layout,
        )
        text = "\n".join(lines)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return rc

    if args.layout == "both":
        cells_e, fc_e = collect_cells(
            args.results_dirs, min_mtime, args.semantics, layout="exp"
        )
        cells_l, fc_l = collect_cells(
            args.results_dirs, min_mtime, args.semantics, layout="legacy"
        )
        lines, all_pass = evaluate_both(cells_e, fc_e, cells_l, fc_l)
        n_ingested = len(cells_e) + len(cells_l)
    else:
        cells, file_counts = collect_cells(
            args.results_dirs, min_mtime, args.semantics, layout=args.layout
        )
        lines, all_pass = evaluate(cells, file_counts)
        n_ingested = len(cells)

    header = [
        "# Supp. Table III spot-check (ae_spotcheck_tab3)",
        "",
        f"- root_label: {args.root_label or '(none)'}",
        f"- semantics: {args.semantics}",
        f"- layout: {args.layout}",
        f"- results_dir: {', '.join(args.results_dirs)}",
        f"- min_mtime_file: {args.min_mtime_file or '(none)'}",
        f"- cells ingested: {n_ingested}",
        "",
        "```",
    ]
    footer = [
        "```",
        "",
        f"**Overall: {'PASS' if all_pass else 'FAIL'}**",
        "",
    ]
    text = "\n".join(header + lines + footer)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(text)
    else:
        print(text)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
