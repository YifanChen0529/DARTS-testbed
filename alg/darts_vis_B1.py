#!/usr/bin/env python3
"""Benchmark-specific visualization for B1_CDR."""

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
# Script lives under alg/; default results root is the repo root (matches today's cwd=repo Path("results")).
RESULTS_DIR = _HERE.parent / "results"

BENCHMARK_ID = "B1_CDR"
STRESS_LEVELS = ["light", "medium", "heavy"]
N_SEEDS = 3
METHODS = ["darts", "darts_nocert", "cea_edf", "sadcher"]
# NOTE: max_rt_ms intentionally excluded — the existing B1 csvs predate
# the R_v=C_v−τ(v) reformulation (paper v6 Theorem 1 Part (2)) and
# contain A_v in this column. RT figures are carried by B2_LML where
# the post-fix sweep produces R_v-correct values.
METRICS = ["task_miss_rate", "rho_hat", "task_time"]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-specific visualization for B1_CDR.")
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


if __name__ == "__main__":
    main()
