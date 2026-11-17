#!/usr/bin/env python3
"""Empirical CDFs of assignment-convergence wall time.

There is no shared style helper in this repo. Conventions below are copied
from scripts/fig_exp2_response_cdf.py and scripts/fig_tb1_tradeoff.py: figure
width 7.16 in and height 2.40 in, typography (title 8.5, label 8, tick 7,
legend 7.5), line geometry, y-grid, savefig pad, and legend-above-panels.
Resolved source paths are printed at startup.

Deliberate ordinal-ramp choice: sequential single-hue shading (light to dark)
is used because the plotted family is ordered by task count, the same
rationale as the published response CDF figure's density ladder, unlike a
categorical method comparison.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.cbba_analysis import (  # noqa: E402
    EXECUTION_NOTE,
    DEFAULT_ROOT,
    median,
    partition_rows,
)

STYLE_SOURCES = (
    "scripts/fig_exp2_response_cdf.py",
    "scripts/fig_tb1_tradeoff.py",
)

FIGSIZE = (7.16, 2.4)
LINEWIDTH = 1.4
GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.5)
TITLE_FS = 8.5
LABEL_FS = 8
TICK_FS = 7
LEGEND_FS = 7.5
SAVEFIG_KW = dict(bbox_inches=None, pad_inches=0.02)
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    frameon=False,
    fontsize=LEGEND_FS,
    ncol=4,
    handlelength=1.8,
    columnspacing=1.4,
    handletextpad=0.4,
)

# Light to dark by increasing task count (ordinal ramp, not categorical).
TASK_COLORS: Sequence[str] = ("#b7b3df", "#8e88c8", "#6560ab", "#403990")
TASK_COUNTS: Sequence[int] = (4, 8, 16, 32)
# Paper binds m as Algorithm 2's outer-iteration index and n as agent count, so a bare "m" in a panel title collides; labels are worded and symbol-free.
# Allocation unit is "job" (M3: one subround per ready job); with four agents and 4/8/16/32 jobs, jobs per agent is 1/2/4/8 exactly, so the ceiling phrasing is unnecessary.
LEGEND_LABELS: Tuple[str, str, str, str] = (
    "4 jobs",
    "8 jobs",
    "16 jobs",
    "32 jobs",
)
XLABEL = "convergence wall time over local loopback (milliseconds)"
YLABEL = "fraction of trials"
PANEL_LEFT = "bundle length one"
PANEL_RIGHT = "bundle length equal to jobs per agent"

DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "figures", "FIG_exp3_tconv_cdf.pdf")
DEFAULT_CAPTION = os.path.join(
    _PROJECT_ROOT, "figures", "FIG_exp3_tconv_cdf_caption.txt"
)


def rendered_strings() -> str:
    """Paper-facing strings only (caption is written separately)."""
    return "\n".join(
        [XLABEL, YLABEL, PANEL_LEFT, PANEL_RIGHT, *LEGEND_LABELS]
    )


def _print_style_provenance() -> None:
    print("[style] no shared helper; conventions copied from:")
    for rel in STYLE_SOURCES:
        print(f"  {str((Path(_PROJECT_ROOT) / rel).resolve())}")
    print(
        "[style] deviation: sequential single-hue shading for an ordinal "
        "task-count ladder (not a categorical method comparison)"
    )


def _cdf_xy(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=float))
    if xs.size == 0:
        return xs, xs
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def _is_ceil_bundle(cid: str) -> bool:
    return cid.endswith("ltceil")


def load_recorded(root: str) -> List[Dict[str, str]]:
    path = os.path.join(root, "cbba_log.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _nulls, _warm, recorded = partition_rows(rows)
    return recorded


def series_for_panel(
    recorded: Sequence[Dict[str, str]], *, ceil_bundle: bool
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in TASK_COUNTS:
        vals = [
            float(r["t_total_ms"])
            for r in recorded
            if int(r["m"]) == m and _is_ceil_bundle(r["config_id"]) == ceil_bundle
        ]
        out.append({"m": m, "values": vals, "median": median(vals) if vals else float("nan")})
    return out


def _draw_panel(ax: Any, series: Sequence[Dict[str, Any]], title: str, show_y: bool) -> None:
    for i, curve in enumerate(series):
        xs, ys = _cdf_xy(curve["values"])
        if xs.size == 0:
            continue
        ax.plot(
            xs,
            ys,
            color=TASK_COLORS[i],
            linestyle="-",
            linewidth=LINEWIDTH,
            zorder=3,
        )
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    if not show_y:
        ax.tick_params(labelleft=False)


def _legend_handles() -> Tuple[List[Line2D], List[str]]:
    handles = [
        Line2D([0], [0], color=TASK_COLORS[i], linestyle="-", linewidth=LINEWIDTH)
        for i in range(len(TASK_COUNTS))
    ]
    return handles, list(LEGEND_LABELS)


def build_caption(
    floor_ms: float,
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
) -> str:
    def _med_line(series: Sequence[Dict[str, Any]], panel: str) -> str:
        bits = [
            f"{int(c['m'])} jobs {_fmt_med(c['median'])} ms" for c in series
        ]
        return f"{panel}: " + "; ".join(bits) + "."

    def _fmt_med(v: float) -> str:
        return f"{v:.3f}"

    parts = [
        EXECUTION_NOTE,
        f"The median null-trial runner overhead was {floor_ms:.4f} milliseconds.",
        _med_line(left, PANEL_LEFT),
        _med_line(right, PANEL_RIGHT),
    ]
    return " ".join(parts)


def draw(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    out_path: str,
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        1, 2, figure=fig, wspace=0.22,
        left=0.08, right=0.995, bottom=0.18, top=0.78,
    )
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_l)
    _draw_panel(ax_l, left, PANEL_LEFT, show_y=True)
    _draw_panel(ax_r, right, PANEL_RIGHT, show_y=False)
    ax_l.set_ylabel(YLABEL, fontsize=LABEL_FS)
    fig.supxlabel(XLABEL, fontsize=LABEL_FS, y=0.02)
    handles, labels = _legend_handles()
    fig.legend(handles, labels, **LEGEND_KW)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, **SAVEFIG_KW)
    w, h = fig.get_size_inches()
    print(f"[size] {w:.2f} x {h:.2f} in")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--caption_out", default=DEFAULT_CAPTION)
    args = ap.parse_args(argv)
    _print_style_provenance()
    recorded = load_recorded(args.root)
    with open(os.path.join(args.root, "cbba_log.csv"), newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    nulls, _w, _r = partition_rows(all_rows)
    floor = median([float(r["t_total_ms"]) for r in nulls])
    left = series_for_panel(recorded, ceil_bundle=False)
    right = series_for_panel(recorded, ceil_bundle=True)
    draw(left, right, args.out)
    caption = build_caption(floor, left, right)
    Path(args.caption_out).write_text(caption + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.caption_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
