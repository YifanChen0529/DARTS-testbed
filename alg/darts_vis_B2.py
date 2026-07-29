#!/usr/bin/env python3
"""Benchmark-specific visualization for B2_LML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Script lives under alg/; default results root is the repo root (matches today's cwd=repo Path("results")).
RESULTS_DIR = _HERE.parent / "results"

BENCHMARK_ID = "B2_LML"
STRESS_LEVELS = ["light", "medium", "heavy"]
N_SEEDS = 3
METHODS = ["darts", "darts_nocert", "cea_edf", "sadcher"]
METRICS = ["task_miss_rate", "rho_hat", "task_time", "max_rt_ms"]

STYLE = {
    "darts": {"color": "#403990", "linestyle": "-", "marker": "o", "label": "DARTS"},
    "darts_nocert": {
        "color": "#D9A441",
        "linestyle": ":",
        "marker": "D",
        "label": "DARTS-NoCert",
    },
    "cea_edf": {"color": "#F46F43", "linestyle": "--", "marker": "s", "label": "CEA-EDF"},
    "sadcher": {"color": "#6C91C2", "linestyle": "-.", "marker": "^", "label": "SADCHER"},
}


def _seed_list() -> List[int]:
    return [100 * (i + 1) for i in range(N_SEEDS)]


def _benchmark_dir_name() -> str:
    return BENCHMARK_ID.split("_", 1)[0]


def _csv_path(method: str, stress_level: str, run_idx: int, seed: int) -> Path:
    # Aggregate sweep layout: darts_nocert → darts_nocert_log.csv (see run_all_experiments_*.py).
    new_path = (
        RESULTS_DIR
        / f"exp_{run_idx}"
        / _benchmark_dir_name()
        / stress_level
        / f"{method}_log.csv"
    )
    if new_path.is_file():
        return new_path
    exp_name = f"{method}_{BENCHMARK_ID}_{stress_level}_seed{seed}"
    return RESULTS_DIR / exp_name / f"{method}_{BENCHMARK_ID}_{stress_level}_log.csv"


def _method_has_any_csv(method: str) -> bool:
    for run_idx, seed in enumerate(_seed_list(), start=1):
        for stress in STRESS_LEVELS:
            if _csv_path(method, stress, run_idx, seed).is_file():
                return True
    return False


def _load_series(method: str, stress_level: str, metric: str) -> Optional[Dict[str, np.ndarray]]:
    rows = []
    episode_ref = None
    for run_idx, seed in enumerate(_seed_list(), start=1):
        p = _csv_path(method, stress_level, run_idx, seed)
        if not p.is_file():
            return None
        df = pd.read_csv(p).replace("N/A", np.nan)
        if "episode" not in df.columns:
            raise FileNotFoundError(f"CSV missing required column 'episode': {p}")
        if metric not in df.columns:
            raise FileNotFoundError(f"CSV missing required metric '{metric}': {p}")
        ep = pd.to_numeric(df["episode"], errors="coerce").to_numpy(dtype=float)
        vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        if episode_ref is None:
            episode_ref = ep
        rows.append(vals)

    min_len = min(len(r) for r in rows)
    return {"episode": episode_ref[:min_len], "values": np.vstack([r[:min_len] for r in rows])}


def _plot_metric(metric: str, out_dir: Path) -> Path:
    n_rows = len(STRESS_LEVELS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.8 * n_rows), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for row_idx, stress in enumerate(STRESS_LEVELS):
        ax = axes[row_idx]
        plotted = 0
        for method in METHODS:
            d = _load_series(method, stress, metric)
            if d is None:
                continue
            plotted += 1
            x = d["episode"] / 1000.0
            y = d["values"]
            mean = np.nanmean(y, axis=0)
            std = np.nanstd(y, axis=0)
            st = STYLE[method]
            markevery = max(1, len(x) // 5)
            ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.15)
            ax.plot(
                x,
                mean,
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=3.5,
                markevery=markevery,
                label=st["label"],
            )
        if plotted == 0:
            raise FileNotFoundError(f"No CSVs found for benchmark={BENCHMARK_ID}, stress={stress}, metric={metric}")
        ax.set_title(f"{BENCHMARK_ID} | stress={stress}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.set_ylabel(metric)

    axes[-1].set_xlabel("Episode (×10³)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        ncol = max(1, len(handles))
        fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)

    out_path = out_dir / f"{BENCHMARK_ID}_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bidding_time(out_dir: Path) -> None:
    """Bidding-time figure for paper v6 Eq. 44.

    Two subplots:
      (A) Per-(method, episode) scatter of mean T_auc(k) for a given |A_k|
          bucket. Each point is one (method, stress, seed, episode, A_k)
          tuple. Linearity in A_k is the empirical witness for Eq. 44.
      (B) Box-plot of mean_t_auc_ms across (seed, episode) for each
          (method, stress) cell. Shows deployment-time auction overhead
          distribution on Jetson Orin Nano.
    """
    BUCKET_COLS = {
        1: "t_auc_Ak1_mean_ms",
        2: "t_auc_Ak2_mean_ms",
        3: "t_auc_Ak3_mean_ms",
        4: "t_auc_Ak4plus_mean_ms",
    }
    COUNT_COLS = {
        1: "n_epochs_Ak1",
        2: "n_epochs_Ak2",
        3: "n_epochs_Ak3",
        4: "n_epochs_Ak4plus",
    }

    scatter_rows: List[Dict[str, object]] = []  # method, A_k_bucket, t_auc_ms
    box_rows: List[Dict[str, object]] = []  # method, stress, mean_t_auc_ms

    for method in METHODS:
        for stress in STRESS_LEVELS:
            for run_idx, seed in enumerate(_seed_list(), start=1):
                p = _csv_path(method, stress, run_idx, seed)
                if not p.is_file():
                    continue
                df = pd.read_csv(p).replace("N/A", np.nan)
                df.columns = [c.strip() for c in df.columns]
                if "mean_t_auc_ms" not in df.columns:
                    continue
                for _, ep_row in df.iterrows():
                    mt = ep_row.get("mean_t_auc_ms")
                    if pd.notna(mt):
                        try:
                            box_rows.append(
                                {
                                    "method": method,
                                    "stress": stress,
                                    "mean_t_auc_ms": float(mt),
                                }
                            )
                        except (TypeError, ValueError):
                            pass
                    for ak, col in BUCKET_COLS.items():
                        if col not in df.columns:
                            continue
                        n_col = COUNT_COLS[ak]
                        if n_col not in df.columns:
                            continue
                        val = ep_row.get(col)
                        n_ep = ep_row.get(n_col)
                        n_ep_f = pd.to_numeric(n_ep, errors="coerce")
                        if pd.notna(val) and pd.notna(n_ep_f) and float(n_ep_f) > 0:
                            try:
                                scatter_rows.append(
                                    {
                                        "method": method,
                                        "A_k": ak,
                                        "t_auc_ms": float(val),
                                    }
                                )
                            except (TypeError, ValueError):
                                pass

    if not scatter_rows and not box_rows:
        print("[plot_bidding_time] no data — skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # ── Subplot A: scatter T_auc vs |A_k| ──────────────────────────────
    axA = axes[0]
    if scatter_rows:
        sdf = pd.DataFrame(scatter_rows)
        rng = np.random.default_rng(0)
        for method in METHODS:
            sub = sdf[sdf["method"] == method]
            if sub.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(sub))
            sty = STYLE.get(method, {})
            axA.scatter(
                sub["A_k"].to_numpy() + jitter,
                sub["t_auc_ms"].to_numpy(),
                color=sty.get("color", "gray"),
                marker=sty.get("marker", "o"),
                label=sty.get("label", method),
                alpha=0.55,
                s=18,
            )
            x = sub["A_k"].to_numpy(dtype=float)
            y = sub["t_auc_ms"].to_numpy(dtype=float)
            if len(x) >= 2 and np.sum(x * x) > 0:
                slope = float(np.sum(x * y) / np.sum(x * x))
                xs = np.linspace(0.5, 4.5, 50)
                axA.plot(
                    xs,
                    slope * xs,
                    color=sty.get("color", "gray"),
                    linestyle=sty.get("linestyle", "-"),
                    linewidth=1.4,
                    alpha=0.9,
                )
                axA.text(
                    4.55,
                    slope * 4.5,
                    f"δ≈{slope:.2f} ms",
                    fontsize=8,
                    color=sty.get("color", "gray"),
                    va="center",
                )
        axA.set_xticks([1, 2, 3, 4])
        axA.set_xticklabels(["1", "2", "3", "≥4"])
        axA.set_xlabel(r"Ready-frontier size $|\mathcal{A}_k|$")
        axA.set_ylabel(r"Per-bucket mean $T_{\mathrm{auc}}(k)$ (ms)")
        axA.set_title("(A) Auction time vs. ready-frontier (Eq. 44)")
        axA.legend(loc="upper left", fontsize=8, framealpha=0.9)
        axA.grid(True, alpha=0.3)
    else:
        axA.set_title("(A) Auction time vs. ready-frontier (Eq. 44)")
        axA.text(0.5, 0.5, "no bucket data", ha="center", va="center", transform=axA.transAxes)

    # ── Subplot B: box-plot mean_t_auc_ms by (method, stress) ─────────
    axB = axes[1]
    if box_rows:
        bdf = pd.DataFrame(box_rows)
        positions: List[float] = []
        data: List[np.ndarray] = []
        colors: List[str] = []
        xticklabels: List[str] = []
        x_pos = 0.0
        for stress in STRESS_LEVELS:
            for method in METHODS:
                sub = bdf[(bdf["method"] == method) & (bdf["stress"] == stress)]
                if sub.empty:
                    x_pos += 0.9
                    continue
                positions.append(x_pos)
                data.append(sub["mean_t_auc_ms"].to_numpy())
                colors.append(STYLE.get(method, {}).get("color", "gray"))
                x_pos += 0.9
            x_pos += 0.6
            xticklabels.append(stress)
        if data:
            bp = axB.boxplot(
                data,
                positions=positions,
                widths=0.7,
                patch_artist=True,
                showfliers=False,
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            group_centers: List[float] = []
            cursor = 0.0
            for _ in STRESS_LEVELS:
                centers_for_group: List[float] = []
                for _m in METHODS:
                    centers_for_group.append(cursor)
                    cursor += 0.9
                group_centers.append(float(np.mean(centers_for_group)))
                cursor += 0.6
            axB.set_xticks(group_centers)
            axB.set_xticklabels(xticklabels)
        axB.set_xlabel("Stress level")
        axB.set_ylabel(r"Per-episode mean $T_{\mathrm{auc}}$ (ms)")
        axB.set_title("(B) Auction overhead distribution on Jetson")
        handles = [
            Patch(facecolor=STYLE[m]["color"], alpha=0.55, label=STYLE[m]["label"])
            for m in METHODS
            if m in STYLE
        ]
        axB.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
        axB.grid(True, axis="y", alpha=0.3)
    else:
        axB.set_title("(B) Auction overhead distribution on Jetson")
        axB.text(0.5, 0.5, "no mean_t_auc_ms data", ha="center", va="center", transform=axB.transAxes)

    fig.suptitle("DARTS bidding-time validation (B2_LML, Jetson Orin Nano)", fontsize=11)
    out_path = out_dir / "B2_bidding_time.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_bidding_time] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-specific visualization for B2_LML.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(_HERE.parent / "results"),
        help="Root directory containing sweep CSVs (default: <repo_root>/results)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Figure output directory (default: <results_dir>/figures/<BENCHMARK_ID>)",
    )
    args = parser.parse_args()
    global RESULTS_DIR
    RESULTS_DIR = Path(args.results_dir).expanduser().resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = RESULTS_DIR / "figures" / BENCHMARK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        if not _method_has_any_csv(m):
            print(f"[skip] method={m}: no matching CSVs under results/ (ok until sweep completes)")
    for metric in METRICS:
        p = _plot_metric(metric, out_dir)
        print(f"Saved {p} | rows={len(STRESS_LEVELS)}")
    plot_bidding_time(out_dir)


if __name__ == "__main__":
    main()
