#!/usr/bin/env python3
"""FIG1: 2×3 B1 grid (task_miss_rate / rho_hat × light/medium/heavy) — §V-B main result."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_csv import safe_read_csv

REPO = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO / "results_a1a2b1_20260521"
OUT_PATH = REPO / "fig" / "FIG1_b1_grid.pdf"
_FIG_NAME = "FIG1_b1_grid.pdf"

STRESS_LEVELS = ["light", "medium", "heavy"]
METHODS = ["darts", "darts_nocert", "darts_no_incentive", "cea_edf", "sadcher"]
EVAL_EPS = list(range(500, 5001, 500))
CONV_EPS = {4500, 5000}
N_EXP = 10

STYLE = {
    "darts": {"color": "#403990", "ls": "-", "marker": "o", "label": "DARTS"},
    "darts_nocert": {
        "color": "#D9A441",
        "ls": ":",
        "marker": "D",
        "label": "DARTS-NoCert",
    },
    "darts_no_incentive": {
        "color": "#58A14C",
        "ls": "-",
        "marker": "v",
        "label": "DARTS-NoIncentive",
    },
    "cea_edf": {"color": "#F46F43", "ls": "--", "marker": "s", "label": "CEA-EDF"},
    "sadcher": {"color": "#6C91C2", "ls": "-.", "marker": "^", "label": "SADCHER"},
}

TOP_YLIM = {
    "light": ((0.0, 0.10), [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]),
    "medium": ((0.0, 0.50), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "heavy": ((0.0, 1.00), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
}
BOT_YLIM = {
    "light": ((0.80, 1.02), [0.80, 0.85, 0.90, 0.95, 1.00]),
    "medium": ((0.85, 1.02), [0.85, 0.90, 0.95, 1.00]),
    "heavy": ((0.70, 1.02), [0.70, 0.80, 0.90, 1.00]),
}


@dataclass
class CellData:
    episode: np.ndarray
    values: np.ndarray  # (n_seeds, n_eval_eps)
    n_seeds: int
    max_ep_plotted: int


@dataclass
class FinalStats:
    n_seeds: int
    miss_final: float
    rho_final: float
    window_used: str


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


def _ensure_rho_hat(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "rho_hat" in df.columns:
        return df
    if "n_full_fidelity" in df.columns and "n_total_jobs" in df.columns:
        df = df.copy()
        num = pd.to_numeric(df["n_full_fidelity"], errors="coerce")
        den = pd.to_numeric(df["n_total_jobs"], errors="coerce")
        df["rho_hat"] = num / den.replace(0, np.nan)
        return df
    print(f"[skip] {path}: rho_hat missing and cannot derive")
    return df


def _align_eval_rows(df: pd.DataFrame, metric: str) -> Optional[np.ndarray]:
    if metric not in df.columns:
        return None
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    sub = df[df["episode"].isin(EVAL_EPS)].sort_values("episode")
    if sub.empty:
        return None
    idx = {int(e): v for e, v in zip(sub["episode"], sub[metric])}
    aligned = np.array([idx.get(ep, np.nan) for ep in EVAL_EPS], dtype=float)
    if np.all(np.isnan(aligned)):
        return None
    return aligned


def _load_cell(method: str, stress: str, metric: str) -> Optional[CellData]:
    paths = sorted(RESULTS_ROOT.glob(f"exp_*/B1/{stress}/{method}_log.csv"))
    seed_rows: List[np.ndarray] = []
    for path in paths:
        df = clean_df(path)
        if df is None:
            continue
        if metric == "rho_hat":
            df = _ensure_rho_hat(df, path)
        row = _align_eval_rows(df, metric)
        if row is not None:
            seed_rows.append(row)

    if not seed_rows:
        print(f"[skip] m={method} s={stress}")
        return None

    values = np.vstack(seed_rows)
    covered_cols = ~np.all(np.isnan(values), axis=0)
    if not np.any(covered_cols):
        print(f"[skip] m={method} s={stress}")
        return None

    last_j = int(np.where(covered_cols)[0][-1])
    max_ep = int(EVAL_EPS[last_j])
    return CellData(
        episode=np.array(EVAL_EPS, dtype=float),
        values=values,
        n_seeds=len(seed_rows),
        max_ep_plotted=max_ep,
    )


def _conv_window_values(df: pd.DataFrame, metric: str) -> Tuple[List[float], str]:
    if metric not in df.columns:
        return [], "NO_DATA"
    strict = df[df["episode"].isin(sorted(CONV_EPS))]
    if len(strict) >= 1:
        vals = pd.to_numeric(strict[metric], errors="coerce").dropna().tolist()
        if vals:
            return vals, "ep4500-5000"
    last2 = df.sort_values("episode").tail(2)
    if len(last2) >= 1:
        vals = pd.to_numeric(last2[metric], errors="coerce").dropna().tolist()
        if vals:
            ep_min = int(last2["episode"].min())
            ep_max = int(last2["episode"].max())
            return vals, f"ep{ep_min}-{ep_max}-PARTIAL"
    return [], "NO_DATA"


def _aggregate_final(method: str, stress: str) -> FinalStats:
    paths = sorted(RESULTS_ROOT.glob(f"exp_*/B1/{stress}/{method}_log.csv"))
    miss_vals: List[float] = []
    rho_vals: List[float] = []
    windows: List[str] = []
    n_seeds = 0

    for path in paths:
        df = clean_df(path)
        if df is None:
            continue
        df_rho = _ensure_rho_hat(df, path)
        miss, w_miss = _conv_window_values(df, "task_miss_rate")
        rho, w_rho = _conv_window_values(df_rho, "rho_hat")
        if not miss and not rho:
            continue
        n_seeds += 1
        miss_vals.extend(miss)
        rho_vals.extend(rho)
        windows.append(w_miss if miss else w_rho)

    if windows and all(w == "ep4500-5000" for w in windows):
        window_used = "ep4500-5000"
    elif any("PARTIAL" in w for w in windows):
        partial = sorted({w for w in windows if "PARTIAL" in w})
        window_used = partial[0] if len(partial) == 1 else "PARTIAL (mixed)"
    elif windows:
        window_used = windows[0]
    else:
        window_used = "NO_DATA"

    miss_final = float(np.mean(miss_vals)) if miss_vals else float("nan")
    rho_final = float(np.mean(rho_vals)) if rho_vals else float("nan")
    return FinalStats(n_seeds, miss_final, rho_final, window_used)


def _plot_series(ax, cell: CellData, method: str) -> None:
    x_all = cell.episode / 1000.0
    with np.errstate(all="ignore"):
        mean = np.nanmean(cell.values, axis=0)
    valid = np.isfinite(mean)
    if not np.any(valid):
        return
    st = STYLE[method]
    ax.plot(
        x_all[valid],
        mean[valid],
        color=st["color"],
        ls=st["ls"],
        marker=st["marker"],
        markersize=2.8,
        markevery=1,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.2,
        label=st["label"],
    )


def _verify_heavy(finals: Dict[Tuple[str, str], FinalStats]) -> None:
    refs = {
        "darts": (0.17, 0.03),
        "cea_edf": (0.61, 0.05),
        "sadcher": (0.65, 0.05),
    }
    for method, (target, tol) in refs.items():
        val = finals[("heavy", method)].miss_final
        if not np.isfinite(val):
            print(f"[WARN] heavy/{method}: miss_final is NaN")
        elif abs(val - target) > tol:
            print(f"[WARN] heavy/{method}: miss_final={val:.4f} (expected ~{target:.3f} ±{tol})")


def _verify_rho_span(cells: Dict[Tuple[str, str], Optional[CellData]]) -> None:
    for stress in ("light", "medium"):
        vals: List[float] = []
        for method in METHODS:
            cell = cells.get((method, stress))
            if cell is None:
                continue
            with np.errstate(all="ignore"):
                mean = np.nanmean(cell.values, axis=0)
            vals.extend([float(v) for v in mean if np.isfinite(v)])
        if not vals:
            print(f"[WARN] {stress} rho panel: no data")
            continue
        span = max(vals) - min(vals)
        if span < 0.02:
            print(f"[WARN] {stress} rho panel: curves span only {span:.4f} (may sit at y≈1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description="FIG1: B1 L/M/H miss-rate + rho-hat grid.")
    ap.add_argument(
        "--new_sweep_dir",
        type=str,
        default=str(REPO / "results_a1a2b1_20260521"),
        help="Root for exp_*/B1/{light,medium,heavy} CSVs (default: <repo_root>/results_a1a2b1_20260521)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO / "fig"),
        help="Directory for FIG1_b1_grid.pdf (default: <repo_root>/fig)",
    )
    args = ap.parse_args()
    global RESULTS_ROOT, OUT_PATH
    RESULTS_ROOT = Path(args.new_sweep_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_PATH = out_dir / _FIG_NAME
    print(f"[fig1] NEW_SWEEP={RESULTS_ROOT}", file=sys.stderr)
    print(f"[fig1] OUT_DIR={out_dir}", file=sys.stderr)

    miss_cells: Dict[Tuple[str, str], Optional[CellData]] = {}
    rho_cells: Dict[Tuple[str, str], Optional[CellData]] = {}
    for stress in STRESS_LEVELS:
        for method in METHODS:
            miss_cells[(method, stress)] = _load_cell(method, stress, "task_miss_rate")
            rho_cells[(method, stress)] = _load_cell(method, stress, "rho_hat")

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.4))

    for col, stress in enumerate(STRESS_LEVELS):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        for method in METHODS:
            miss = miss_cells.get((method, stress))
            if miss is not None:
                _plot_series(ax_top, miss, method)
            rho = rho_cells.get((method, stress))
            if rho is not None:
                _plot_series(ax_bot, rho, method)

        ax_top.set_title(f"B1 | {stress}", fontsize=8.5)

        ylim, yticks = TOP_YLIM[stress]
        ax_top.set_ylim(*ylim)
        ax_top.set_yticks(yticks)
        ax_top.tick_params(labelsize=7)

        ylim_b, yticks_b = BOT_YLIM[stress]
        ax_bot.set_ylim(*ylim_b)
        ax_bot.set_yticks(yticks_b)
        ax_bot.tick_params(labelsize=7)
        ax_bot.set_xlabel(r"Episode ($\times 10^3$)", fontsize=8)

        if stress == "medium":
            ax_top.text(
                0.02,
                0.95,
                "Scale: 5×",
                transform=ax_top.transAxes,
                fontsize=6.5,
                color="#666",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )
        elif stress == "heavy":
            ax_top.text(
                0.02,
                0.95,
                "Scale: 10×",
                transform=ax_top.transAxes,
                fontsize=6.5,
                color="#666",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )

    axes[0, 0].set_ylabel("Task miss rate", fontsize=8)
    axes[1, 0].set_ylabel(r"Certified coverage $\hat\rho$", fontsize=8)
    for row in (0, 1):
        axes[row, 1].set_ylabel("")
        axes[row, 2].set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=7.5,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.4,
    )

    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0], h_pad=0.6, w_pad=0.8)
    fig.savefig(OUT_PATH, bbox_inches=None, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")

    finals: Dict[Tuple[str, str], FinalStats] = {}
    print()
    header = f"{'stress':<8} {'method':<20} {'n_seeds':>8} {'miss_final':>12} {'rho_final':>12} {'window_used':<18}"
    print(header)
    print("-" * len(header))
    for stress in STRESS_LEVELS:
        for method in METHODS:
            st = _aggregate_final(method, stress)
            finals[(stress, method)] = st
            print(
                f"{stress:<8} {method:<20} {st.n_seeds:8d} "
                f"{st.miss_final:12.4f} {st.rho_final:12.4f} {st.window_used:<18}"
            )

    _verify_heavy(finals)
    _verify_rho_span(rho_cells)

    for stress in ("light", "heavy"):
        cell = miss_cells.get(("darts_no_incentive", stress))
        if cell is not None and cell.n_seeds < 5:
            print(
                f"[partial] noinc {stress}: {cell.n_seeds} seeds plotted to episode {cell.max_ep_plotted}"
            )


if __name__ == "__main__":
    main()
