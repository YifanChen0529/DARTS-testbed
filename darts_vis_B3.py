#!/usr/bin/env python3
"""Benchmark-specific visualization for B3_IQI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"

BENCHMARK_ID = "B3_IQI"
STRESS_LEVELS = ["medium"]
N_SEEDS = 5
METHODS = ["darts", "cea_edf", "sadcher"]
METRICS = ["task_miss_rate", "rho_hat", "task_time", "max_rt_ms"]

STYLE = {
    "darts": {"color": "#403990", "linestyle": "-", "marker": "o", "label": "DARTS"},
    "cea_edf": {"color": "#F46F43", "linestyle": "--", "marker": "s", "label": "CEA-EDF"},
    "sadcher": {"color": "#6C91C2", "linestyle": "-.", "marker": "^", "label": "SADCHER"},
}


def _seed_list() -> List[int]:
    return [100 * (i + 1) for i in range(N_SEEDS)]


def _benchmark_dir_name() -> str:
    return BENCHMARK_ID.split("_", 1)[0]


def _csv_path(method: str, stress_level: str, run_idx: int, seed: int) -> Path:
    new_path = RESULTS_DIR / f"exp_{run_idx}" / _benchmark_dir_name() / stress_level / f"{method}_log.csv"
    if new_path.is_file():
        return new_path
    exp_name = f"{method}_{BENCHMARK_ID}_{stress_level}_seed{seed}"
    return RESULTS_DIR / exp_name / f"{method}_{BENCHMARK_ID}_{stress_level}_log.csv"


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
            ax.plot(x, mean, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markersize=3.5, markevery=markevery, label=st["label"])
        if plotted == 0:
            raise FileNotFoundError(f"No CSVs found for benchmark={BENCHMARK_ID}, stress={stress}, metric={metric}")
        ax.set_title(f"{BENCHMARK_ID} | stress={stress}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.set_ylabel(metric)

    axes[-1].set_xlabel("Episode (×10³)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)

    out_path = out_dir / f"{BENCHMARK_ID}_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-specific visualization for B3_IQI.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
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
    for metric in METRICS:
        p = _plot_metric(metric, out_dir)
        print(f"Saved {p} | rows={len(STRESS_LEVELS)}")


if __name__ == "__main__":
    main()
