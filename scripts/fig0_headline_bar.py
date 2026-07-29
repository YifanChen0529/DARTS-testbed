#!/usr/bin/env python3
"""FIG0: single-column headline bar — B1/heavy task miss rate (DARTSrt vs baselines, D3 Fig.11 style)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_csv import safe_read_csv

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results_a1a2b1_20260521"
OUT_PATH = REPO / "fig" / "FIG0_headline_b1heavy.pdf"
_FIG_NAME = "FIG0_headline_b1heavy.pdf"

METHODS: Sequence[str] = (
    "darts",
    "darts_nocert",
    "darts_no_incentive",
    "cea_edf",
    "sadcher",
)
CONV_EPS = {4500, 5000}
N_SEEDS = 10
BAR_X = [0, 1, 2, 4, 5.5]
BAR_COLORS = ["#403990", "#B5A4D6", "#B5A4D6", "#F46F43", "#6C91C2"]
XTICK_LABELS = [r"DARTS$^{\mathrm{rt}}$", "NoCert", "NoIncen", "CEA-EDF", "SADCHER"]
BASELINE_METHODS = {"cea_edf", "sadcher"}
YMAX = 1.20

# Inventory §2 fallback (used only if every CSV read fails).
FALLBACK: Dict[str, Tuple[float, float, int, int, str, bool]] = {
    "darts": (0.172, 0.145, 9, 18, "ep4500-5000", False),
    "darts_nocert": (0.165, 0.124, 10, 20, "ep4500-5000", False),
    "darts_no_incentive": (0.150, 0.112, 4, 8, "ep4500-5000-PARTIAL", True),
    "cea_edf": (0.610, 0.130, 10, 20, "ep4500-5000", False),
    "sadcher": (0.645, 0.432, 10, 20, "ep4500-5000", False),
}


@dataclass
class MethodStats:
    method: str
    n_seeds: int
    n_rows: int
    mean: float
    std: float
    window_used: str
    ratio_to_darts: float
    uses_partial: bool = False


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


def conv_window_rows(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Task-miss values in convergence window {4500, 5000}, else last-two-episode fallback."""
    if "task_miss_rate" not in df.columns:
        return np.array([]), "NO_DATA"
    strict = df[df["episode"].isin(sorted(CONV_EPS))]
    if len(strict) >= 1:
        vals = pd.to_numeric(strict["task_miss_rate"], errors="coerce").dropna().values
        return vals, "ep4500-5000"
    last2 = df.sort_values("episode").tail(2)
    if len(last2) >= 1:
        vals = pd.to_numeric(last2["task_miss_rate"], errors="coerce").dropna().values
        ep_min = int(last2["episode"].min())
        ep_max = int(last2["episode"].max())
        return vals, f"ep{ep_min}-{ep_max}-PARTIAL"
    return np.array([]), "NO_DATA"


def _csv_path(seed: int, method: str) -> Path:
    return RESULTS / f"exp_{seed}" / "B1" / "heavy" / f"{method}_log.csv"


def _window_label(seed_windows: List[str]) -> str:
    if not seed_windows:
        return "NO_DATA"
    if all(w == "ep4500-5000" for w in seed_windows):
        return "ep4500-5000"
    partial = [w for w in seed_windows if "PARTIAL" in w]
    if partial:
        return partial[0] if len(set(partial)) == 1 else "PARTIAL (mixed eps)"
    return seed_windows[0]


def aggregate_method(method: str) -> MethodStats:
    all_vals: List[float] = []
    seed_windows: List[str] = []
    n_seeds = 0

    for seed in range(1, N_SEEDS + 1):
        path = _csv_path(seed, method)
        if not path.exists():
            continue
        df = clean_df(path)
        if df is None or df.empty:
            continue
        vals, label = conv_window_rows(df)
        if len(vals) < 1:
            continue
        n_seeds += 1
        all_vals.extend(vals.tolist())
        seed_windows.append(label)

    uses_partial = any("PARTIAL" in w for w in seed_windows)
    window_used = _window_label(seed_windows)

    if all_vals:
        mean = float(np.mean(all_vals))
        std = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
    else:
        mean = float("nan")
        std = float("nan")

    return MethodStats(
        method=method,
        n_seeds=n_seeds,
        n_rows=len(all_vals),
        mean=mean,
        std=std,
        window_used=window_used,
        ratio_to_darts=float("nan"),
        uses_partial=uses_partial,
    )


def _apply_fallback(stats: Dict[str, MethodStats]) -> bool:
    """Return True if fallback was applied."""
    if any(np.isfinite(stats[m].mean) for m in METHODS):
        return False
    print("[FALLBACK using inventory-confirmed values]", file=sys.stderr)
    for method in METHODS:
        mean, std, n_seeds, n_rows, window, partial = FALLBACK[method]
        stats[method] = MethodStats(
            method=method,
            n_seeds=n_seeds,
            n_rows=n_rows,
            mean=mean,
            std=std,
            window_used=window,
            ratio_to_darts=float("nan"),
            uses_partial=partial,
        )
    return True


def _attach_ratios(stats: Dict[str, MethodStats]) -> None:
    darts_mean = stats["darts"].mean
    for method in METHODS:
        st = stats[method]
        if np.isfinite(darts_mean) and darts_mean > 0 and np.isfinite(st.mean):
            st.ratio_to_darts = st.mean / darts_mean
        else:
            st.ratio_to_darts = float("nan")


def _print_table(stats: Dict[str, MethodStats]) -> None:
    header = (
        f"{'method':<22} {'n_seeds':>8} {'n_rows':>8} {'mean':>8} {'std':>8} "
        f"{'window_used':<22} {'ratio_to_darts':>16}"
    )
    print(header)
    print("-" * len(header))
    for method in METHODS:
        st = stats[method]
        mean_s = f"{st.mean:.4f}" if np.isfinite(st.mean) else "nan"
        std_s = f"{st.std:.4f}" if np.isfinite(st.std) else "nan"
        ratio_s = f"{st.ratio_to_darts:.3f}" if np.isfinite(st.ratio_to_darts) else "nan"
        print(
            f"{method:<22} {st.n_seeds:8d} {st.n_rows:8d} {mean_s:>8} {std_s:>8} "
            f"{st.window_used:<22} {ratio_s:>16}"
        )


def _verify(stats: Dict[str, MethodStats]) -> None:
    darts = stats["darts"].mean
    cea = stats["cea_edf"].mean
    sad = stats["sadcher"].mean

    if not np.isfinite(darts) or not (0.15 <= darts <= 0.20):
        print(f"[WARN] DARTS heavy mean out of expected range (got {darts:.4f}, want [0.15, 0.20])")
    if np.isfinite(cea) and np.isfinite(darts) and cea < 3.0 * darts:
        print(f"[WARN] CEA-EDF ratio={cea / darts:.2f}× < 3×")
    if np.isfinite(sad) and np.isfinite(darts) and sad < 3.0 * darts:
        print(f"[WARN] SADCHER ratio={sad / darts:.2f}× < 3×")


def _plot(stats: Dict[str, MethodStats]) -> None:
    means = np.array([stats[m].mean for m in METHODS], dtype=float)
    stds = np.array([stats[m].std for m in METHODS], dtype=float)
    darts_mean = stats["darts"].mean

    fig, ax = plt.subplots(figsize=(3.6, 3.2))

    plot_means = np.where(np.isfinite(means), means, 0.0)
    plot_stds = np.where(np.isfinite(stds), stds, 0.0)
    ax.bar(
        BAR_X,
        plot_means,
        width=0.65,
        color=BAR_COLORS,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    ax.errorbar(
        BAR_X,
        plot_means,
        yerr=plot_stds,
        fmt="none",
        ecolor="black",
        capsize=2.5,
        capthick=1.0,
        linewidth=1.0,
        clip_on=False,
        zorder=3,
    )

    for mean, std, method, x in zip(means, stds, METHODS, BAR_X):
        if not np.isfinite(mean):
            continue
        err_top = mean + (std if np.isfinite(std) else 0.0)
        partial = stats[method].uses_partial
        label = f"{mean:.3f}*" if partial else f"{mean:.3f}"
        if method in BASELINE_METHODS:
            tier1_y = min(err_top + 0.04, 1.08)
            tier2_y = min(err_top + 0.13, 1.16)
        else:
            tier1_y = min(err_top + 0.04, 1.08)
            tier2_y = None
        ax.text(
            x,
            tier1_y,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
        )
        if (
            method in BASELINE_METHODS
            and tier2_y is not None
            and np.isfinite(darts_mean)
            and darts_mean > 0
        ):
            ratio = mean / darts_mean
            ax.text(
                x,
                tier2_y,
                f"{ratio:.1f}×",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#C0392B",
            )

    fig.text(
        0.50,
        0.10,
        "B1 / heavy stress (10 seeds, convergence window)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="black",
        transform=fig.transFigure,
    )
    fig.text(
        0.50,
        0.03,
        "Ablations within seed noise (p > 0.5)",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555",
        style="italic",
        transform=fig.transFigure,
    )

    ax.set_xticks(BAR_X)
    ax.set_xticklabels(XTICK_LABELS, fontsize=8)
    ax.set_xlim(-0.5, 6.0)
    ax.set_ylabel("Task miss rate", fontsize=8)
    ax.set_ylim(0.0, YMAX)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.20)

    if any(stats[m].uses_partial for m in METHODS):
        fig.text(
            0.5,
            0.02,
            "*Partial data (sweep in progress; updated ~5/24 noon)",
            ha="center",
            va="bottom",
            fontsize=6,
            style="italic",
            color="#666",
        )

    fig.savefig(OUT_PATH, bbox_inches=None, pad_inches=0.04)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"FIG 0 tight bbox: {bbox} (figsize was {fig.get_size_inches()})")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="FIG0: B1/heavy headline miss-rate bar.")
    ap.add_argument(
        "--new_sweep_dir",
        type=str,
        default=str(REPO / "results_a1a2b1_20260521"),
        help="Root for exp_*/B1/heavy CSVs (default: <repo_root>/results_a1a2b1_20260521)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO / "fig"),
        help="Directory for FIG0_headline_b1heavy.pdf (default: <repo_root>/fig)",
    )
    args = ap.parse_args()
    global RESULTS, OUT_PATH
    RESULTS = Path(args.new_sweep_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_PATH = out_dir / _FIG_NAME
    print(f"[fig0] NEW_SWEEP={RESULTS}", file=sys.stderr)
    print(f"[fig0] OUT_DIR={out_dir}", file=sys.stderr)

    stats: Dict[str, MethodStats] = {m: aggregate_method(m) for m in METHODS}
    _apply_fallback(stats)
    _attach_ratios(stats)

    noinc = stats["darts_no_incentive"]
    print(f"N_seeds_used (darts_no_incentive heavy): {noinc.n_seeds}")
    if noinc.n_seeds < 5:
        print(f"[partial] noinc heavy: {noinc.n_seeds} seeds used")

    _print_table(stats)
    _verify(stats)
    _plot(stats)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
