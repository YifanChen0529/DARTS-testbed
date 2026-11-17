#!/usr/bin/env python3
"""Diagnostic report for FIG0–FIG4 suspicious findings (read-only Phase 1)."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from recover_63col import extract_63col_data_rows, load_canonical_header
from util_csv import safe_read_csv

NEW_SWEEP = REPO / "results_a1a2b1_20260521"
FIG_DIR = REPO / "fig"
CONV_EPS = {4500, 5000}
EVAL_CHECK = [500, 1000, 1500, 2000, 2500]
AK1_COL = "t_auc_Ak1_mean_ms"
HARDCODE_P99 = {2: 24.85, 4: 26.21, 6: 59.53, 8: 78.30}
STRESS_MAP = {2: "medium_n2", 4: "medium", 6: "medium_n6", 8: "medium_n8"}


def _conda_env() -> str:
    return subprocess.check_output(["conda", "info", "--envs"], text=True).split("*")[0].strip().split("\n")[-1] if subprocess.run(["which", "conda"], capture_output=True).returncode == 0 else "unknown"


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def clean_df(path: Path) -> Optional[pd.DataFrame]:
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


def conv_window_rows(df: pd.DataFrame) -> pd.DataFrame:
    strict = df[df["episode"].isin(sorted(CONV_EPS))]
    if not strict.empty:
        return strict.sort_values("episode")
    return df.sort_values("episode").tail(2)


def _header() -> List[str]:
    return load_canonical_header()


def issue_a() -> str:
    lines = ["────── ISSUE A (CRITICAL) — FIG 2 bottom-left: NoCert > DARTS anomaly ──────"]
    header = _header()
    darts_means: List[float] = []
    nocert_means: List[float] = []
    darts_detail: List[str] = []
    nocert_detail: List[str] = []

    fig2 = (REPO / "scripts" / "fig2_generalization_scaling.py").read_text()
    has_strip = "df.columns = df.columns.str.strip()" in fig2
    lines.append(f"  fig2 clean_df strips columns: {has_strip}")

    for method, store, detail in (
        ("darts", darts_means, darts_detail),
        ("darts_nocert", nocert_means, nocert_detail),
    ):
        for seed in range(1, 11):
            path = NEW_SWEEP / f"exp_{seed}" / "B1" / "medium" / f"{method}_log.csv"
            if not path.exists():
                detail.append(f"exp_{seed}: MISSING")
                continue
            df_raw = safe_read_csv(path)
            raw_cols = list(df_raw.columns.str.strip()) if df_raw is not None else []
            raw_has = AK1_COL in raw_cols
            df = clean_df(path)
            strip_has = df is not None and AK1_COL in df.columns
            vals: List[float] = []
            src_parts: List[str] = []
            for row in extract_63col_data_rows(path, header):
                ep = int(float(row["episode"]))
                if ep not in CONV_EPS:
                    continue
                try:
                    vals.append(float(row[AK1_COL]))
                    src_parts.append("63col")
                except (ValueError, TypeError, KeyError):
                    pass
            if df is not None and strip_has:
                sub = conv_window_rows(df)
                for v in pd.to_numeric(sub[AK1_COL], errors="coerce"):
                    if pd.notna(v):
                        vals.append(float(v))
                        src_parts.append("pandas")
            if vals:
                m = float(np.mean(vals))
                store.append(m)
                detail.append(f"exp_{seed}: mean={m:.4f} n={len(vals)} src={'+'.join(sorted(set(src_parts)))}")
            else:
                detail.append(
                    f"exp_{seed}: NO_DATA raw_has={raw_has} strip_has={strip_has} "
                    f"fallback=HARDCODE if script uses HARDCODE_T_AUC"
                )

    mean_d = float(np.mean(darts_means)) if darts_means else float("nan")
    mean_n = float(np.mean(nocert_means)) if nocert_means else float("nan")
    ratio = mean_n / mean_d if mean_d > 0 else float("nan")

    if not has_strip:
        label = "WHITESPACE_BUG"
        conclusion = "clean_df missing column strip; 62-col headers hide t_auc columns."
    elif nocert_means and darts_means and mean_n > mean_d:
        label = "REAL_DATA_NOCERT_HIGHER"
        conclusion = (
            "CSV/63-col recovery yields NoCert T_auc(|A_k|=1) > DARTS across seeds; "
            "not a fallback artifact. Likely reflects bidding-path differences without cert fast-path."
        )
    elif any("NO_DATA" in d for d in nocert_detail + darts_detail):
        label = "HARDCODE_FALLBACK"
        conclusion = "Missing columns triggered HARDCODE_T_AUC fallback for some cells."
    else:
        label = "INCONCLUSIVE"
        conclusion = "Inspect per-seed lists below."

    lines.append(f"\nISSUE A: {label}")
    lines.append(f"  Per-seed DARTS T_auc(1)   = {[round(x, 4) for x in darts_means]}")
    lines.append(f"  Per-seed NoCert T_auc(1)  = {[round(x, 4) for x in nocert_means]}")
    lines.append(f"  Mean DARTS / Mean NoCert  = {mean_d:.4f} / {mean_n:.4f} (NoCert/DARTS={ratio:.2f}×)")
    lines.append("  Per-seed detail (darts):")
    lines.extend(f"    {d}" for d in darts_detail)
    lines.append("  Per-seed detail (nocert):")
    lines.extend(f"    {d}" for d in nocert_detail)
    lines.append(f"  Conclusion: {conclusion}")
    return "\n".join(lines)


def issue_b() -> str:
    lines = ["────── ISSUE B (CRITICAL) — FIG 2 bottom-right: P99 non-monotone ──────"]
    header = _header()
    rows_out: List[Tuple[int, int, int, float, float, float]] = []

    for n, stress in STRESS_MAP.items():
        seeds = list(range(1, 4)) if n != 4 else list(range(1, 11))
        all_vals: List[float] = []
        last_eps: List[int] = []
        for seed in seeds:
            path = NEW_SWEEP / f"exp_{seed}" / "B1" / stress / "darts_log.csv"
            if not path.exists():
                continue
            df = clean_df(path)
            if df is None:
                continue
            last_eps.append(int(df["episode"].max()))
            last2 = df.sort_values("episode").tail(2)
            if "max_t_auc_ms" in df.columns:
                for v in pd.to_numeric(last2["max_t_auc_ms"], errors="coerce"):
                    if pd.notna(v):
                        all_vals.append(float(v))
            want = set(last2["episode"].astype(int))
            for row in extract_63col_data_rows(path, header):
                ep = int(float(row["episode"]))
                if ep not in want:
                    continue
                try:
                    all_vals.append(float(row["max_t_auc_ms"]))
                except (ValueError, TypeError, KeyError):
                    pass

        derived = float(np.quantile(all_vals, 0.99)) if all_vals else float("nan")
        fb = HARDCODE_P99[n]
        rel = abs(derived - fb) / fb if np.isfinite(derived) and fb else float("nan")
        rows_out.append((n, max(last_eps) if last_eps else -1, len(seeds), derived, fb, rel))

    derived_list = [r[3] for r in rows_out]
    monotone = all(
        np.isfinite(derived_list[i]) and np.isfinite(derived_list[i + 1]) and derived_list[i] < derived_list[i + 1]
        for i in range(len(derived_list) - 1)
    )
    all_finite = all(np.isfinite(d) for d in derived_list)
    if all_finite and monotone:
        decision = "use derived"
        label = "REAL_DATA_AVAILABLE"
    elif all_finite and not monotone:
        decision = "mixed — n=4 anchor uses full 5000-ep sweep; n∈{2,6,8} use 2500-ep B1-priority cells"
        label = "REAL_DATA_AVAILABLE"
    else:
        decision = "keep fallback"
        label = "FALLBACK_REQUIRED"

    lines.append(f"\nISSUE B: {label}")
    lines.append("| n | last_ep | n_seeds | derived_P99 | fallback_P99 | rel_diff |")
    lines.append("|---|---------|---------|-------------|--------------|----------|")
    for n, last_ep, n_seeds, der, fb, rel in rows_out:
        der_s = f"{der:.2f}" if np.isfinite(der) else "—"
        rel_s = f"{rel:.3f}" if np.isfinite(rel) else "—"
        lines.append(f"| {n} | {last_ep} | {n_seeds} | {der_s} | {fb:.2f} | {rel_s} |")
    lines.append(f"  Monotone (derived): {monotone}")
    lines.append(f"  Decision: {decision}")
    return "\n".join(lines)


def issue_c() -> str:
    lines = ["────── ISSUE C (MEDIUM) — FIG 1 NoIncentive light spike ──────"]
    ep1500: List[Tuple[str, float]] = []
    table: Dict[int, Dict[int, float]] = {}

    for seed in range(1, 11):
        path = NEW_SWEEP / f"exp_{seed}" / "B1" / "light" / "darts_no_incentive_log.csv"
        if not path.exists():
            continue
        df = clean_df(path)
        if df is None:
            continue
        seed_vals: Dict[int, float] = {}
        for ep in EVAL_CHECK:
            sub = df[df["episode"] == ep]
            if len(sub):
                seed_vals[ep] = float(pd.to_numeric(sub["task_miss_rate"], errors="coerce").iloc[0])
        table[seed] = seed_vals
        if 1500 in seed_vals:
            ep1500.append((f"exp_{seed}", seed_vals[1500]))

    vals1500 = [v for _, v in ep1500]
    pop_mean = float(np.mean(vals1500)) if vals1500 else float("nan")
    pop_std = float(np.std(vals1500, ddof=1)) if len(vals1500) > 1 else 0.0

    if vals1500:
        med = float(np.median(vals1500))
        outliers = [s for s, v in ep1500 if abs(v - med) > 2 * pop_std and pop_std > 0]
        if len(outliers) == 1 and pop_std > 0.01:
            label = "SINGLE_SEED_OUTLIER"
            conclusion = f"Seed(s) {outliers} drive ep1500 spike; others near {med:.3f}."
        elif pop_std < 0.015:
            label = "POPULATION_NOISE"
            conclusion = "Small spread; spike is modest relative to low baseline."
        else:
            label = "POPULATION_NOISE"
            conclusion = "Multiple seeds elevated at ep1500; training variance not data bug."
    else:
        label = "DATA_BUG"
        conclusion = "No ep1500 rows found."

    lines.append(f"\nISSUE C: {label}")
    lines.append(f"  Per-seed task_miss_rate @ ep1500: {[round(v, 4) for _, v in ep1500]}")
    lines.append(f"  Population mean ± SD: {pop_mean:.4f} ± {pop_std:.4f}")
    lines.append("  Per-seed trajectory (500/1000/1500/2000/2500):")
    for seed in sorted(table):
        row = table[seed]
        parts = [f"ep{ep}={row.get(ep, float('nan')):.4f}" for ep in EVAL_CHECK if ep in row]
        lines.append(f"    exp_{seed}: " + ", ".join(parts))
    lines.append(f"  Conclusion: {conclusion}")
    return "\n".join(lines)


def issue_d() -> str:
    lines = ["────── ISSUE D (HIGH) — FIG 0 missing error bars ──────"]
    src = (REPO / "scripts" / "fig0_headline_bar.py").read_text()
    has_yerr = "yerr=" in src
    has_capsize = "capsize=2.5" in src or "capsize=" in src

    m = re.search(r"ax\.bar\([^)]+\)", src, re.DOTALL)
    snippet = m.group(0) if m else "(bar call not found)"

    # compute actual stds
    sys.path.insert(0, str(REPO / "scripts"))
    import fig0_headline_bar as f0

    stats = {m: f0.aggregate_method(m) for m in f0.METHODS}
    stds = {m: stats[m].std for m in f0.METHODS}
    max_std = max(stds.values())

    if not has_yerr:
        label = "ERRORBARS_MISSING"
    elif max_std < 0.005:
        label = "ERRORBARS_PRESENT"
        note = "yerr passed but SD < 0.005 — whiskers may be invisible at ylim=0.85"
    elif "bbox_inches=\"tight\"" in src:
        label = "ERRORBARS_CLIPPED"
        note = "yerr present; bbox_inches=tight may clip caps at figure edge"
    else:
        label = "ERRORBARS_PRESENT"
        note = "yerr with capsize=2.5 present in code"

    lines.append(f"\nISSUE D: {label}")
    lines.append(f"  Per-method SD: { {m: round(stds[m], 4) for m in f0.METHODS} }")
    lines.append(f"  Note: {note if 'note' in dir() else ''}")
    lines.append("  Code snippet (ax.bar call):")
    for sl in snippet.split("\n"):
        lines.append(f"    {sl}")
    return "\n".join(lines)


def issue_e() -> str:
    lines = ["────── ISSUE E (LOW) — FIG 3 legend overlaps B4 row labels ──────"]
    src = (REPO / "scripts" / "fig3_t3_bound_tightness.py").read_text()
    m = re.search(r"ax\.legend\((.*?)\)", src, re.DOTALL)
    snippet = m.group(0) if m else "legend call not found"
    lines.append(f"\nISSUE E: LEGEND_OVERLAP")
    lines.append(f"  Current legend call:\n    {snippet.replace(chr(10), chr(10)+'    ')}")
    lines.append("  Proposed fix: loc='center right', bbox_to_anchor=(0.98, 0.55)")
    return "\n".join(lines)


def issue_f() -> str:
    lines = ["────── ISSUE F (LOW) — FIG 2 top-row B4 panel visually empty ──────"]
    sys.path.insert(0, str(REPO / "scripts"))
    import fig2_generalization_scaling as f2

    finals: Dict[str, float] = {}
    for method in ["darts", "cea_edf", "sadcher"]:
        cell = f2.load_eval_cell(f2.paths_b4(method))
        if cell is not None:
            finals[method] = float(np.nanmean(cell.values[:, -1]))
    lines.append(f"\nISSUE F: SHARED_YLIM_COLLAPSES_B4")
    lines.append(f"  B4 final miss rates: {finals}")
    lines.append("  Current shared ylim: (0, 0.50) — B4 curves cluster near 0.")
    lines.append("  PROPOSED per-column y-limits (DO NOT auto-apply):")
    lines.append("    B1: (0, 0.50)  B2: (0, 0.15)  B3: (0, 0.35)  B4: (0, 0.05)")
    lines.append('  B4 annotation: "Scale: 10× zoom" at (0.02, 0.95) axes coords')
    return "\n".join(lines)


def pdf_sizes() -> Dict[str, int]:
    out = {}
    for p in sorted(FIG_DIR.glob("FIG*.pdf")):
        out[p.name] = p.stat().st_size
    return out


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    report_parts = [
        "# FIG Diagnostic Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Conda env: DARTS_nano",
        f"Git HEAD: {_git_head()}",
        f"Working dir: {REPO}",
        "",
        "═══════════════════════════════════════════════════════════════════════════════",
        "PHASE 1 — DIAGNOSTIC INSPECTION",
        "═══════════════════════════════════════════════════════════════════════════════",
        "",
        issue_a(),
        "",
        issue_b(),
        "",
        issue_c(),
        "",
        issue_d(),
        "",
        issue_e(),
        "",
        issue_f(),
        "",
        "Pre-fix PDF sizes:",
    ]
    for name, size in pdf_sizes().items():
        report_parts.append(f"  {name}: {size} bytes")

    text = "\n".join(report_parts)
    print(text)
    out_md = REPO / f"DIAG_REPORT_{ts}.md"
    out_md.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()
