#!/usr/bin/env python3
"""Empirical CDFs of per-release response time divided by relative deadline.

There is no shared style helper in this repo. Conventions below are copied
from scripts/fig_tb1_tradeoff.py and scripts/fig_tb3_f3.py: figure width
7.16 in and height 2.40 in, typography, line geometry, savefig, y-grid, and
legend-above-panels from those two (and the published-figure sources they
name). Resolved source paths are printed at startup.

Deliberate deviation: sequential single-hue shading is used because the
plotted family is ordinal (a density ladder), unlike the categorical method
comparisons in the source figures.
"""
from __future__ import annotations

import argparse
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

import scripts.thm2a_analysis as tha  # noqa: E402

STYLE_SOURCES = (
    "scripts/fig_tb1_tradeoff.py",
    "scripts/fig_tb3_f3.py",
    "scripts/fig0_headline_bar.py",
    "scripts/fig1_b1_grid.py",
    "scripts/fig2_generalization_scaling.py",
)

FIGSIZE = (7.16, 2.4)
LINEWIDTH_ADMITTED = 1.4
LINEWIDTH_OUTSIDE = 1.4
LINEWIDTH_SECOND = 0.9
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
    ncol=3,
    handlelength=1.8,
    columnspacing=1.4,
    handletextpad=0.4,
)

# Light to dark by increasing admitted density. Second-darkest = #6560ab.
ADMITTED_COLORS: Sequence[str] = ("#b7b3df", "#8e88c8", "#6560ab", "#403990")
ADMITTED_LEGEND_COLOR = "#6560ab"
OUTSIDE_COLOR = "#F46F43"
SECOND_COLOR = "#403990"

MAIN_X_LO = 0.0
MAIN_X_HI = 0.72
STRIP_X_LO = 0.96
STRIP_X_HI = 1.04
DEADLINE_X = 1.0
PANEL_ORDER = ("B3_IQI", "B4_DEM")
PANEL_TITLES: Tuple[str, str] = ("B3", "B4")
PANEL_TITLE_BY_WORKLOAD = {
    "B3_IQI": "B3",
    "B4_DEM": "B4",
}
BREAK_DIAG = 0.015

LEGEND_LABELS: Tuple[str, str, str] = (
    "admitted (density increases left to right)",
    "outside the certified region",
    "second evaluation seed",
)

DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "figures", "FIG_exp2_response_cdf.pdf")
DEFAULT_CAPTION = os.path.join(
    _PROJECT_ROOT, "figures", "FIG_exp2_response_cdf_caption.txt"
)

CAPTION = (
    "Empirical CDFs of per-release response time divided by relative deadline "
    "for B3 (left panel) and B4 (right panel). Solid curves are "
    "the admitted configurations; density increases left to right with darker "
    "shade (achieved factored bottleneck densities 0.2997, 0.4990, 0.6967, "
    "0.8448 on B3 and 0.2998, 0.4989, 0.6986, 0.8462 on B4). The dashed curve "
    "is the configuration outside the certified region (density 1.0730 on B3, "
    "1.0726 on B4). The dotted curve is the densest admitted configuration at "
    "a second evaluation seed. Each curve pools 120 job releases (4 classes "
    "times 30 releases). The vertical line marks the relative deadline "
    "(ratio 1). The horizontal axis is compressed between 0.72 and 0.96; no "
    "plotted job release falls in the omitted interval."
)


def rendered_strings() -> str:
    """Paper-facing strings only (caption, legend, axis labels, panel titles)."""
    labels = [
        CAPTION,
        "relative deadline",
        "per-release response time divided by relative deadline",
        "fraction of job releases",
        *PANEL_TITLES,
        *LEGEND_LABELS,
    ]
    return "\n".join(labels)


def _print_style_provenance() -> None:
    print("[style] no shared helper; conventions copied from:")
    for rel in STYLE_SOURCES:
        print(f"  {str((Path(_PROJECT_ROOT) / rel).resolve())}")
    print(
        "[style] deviation: sequential single-hue shading for an ordinal "
        "density ladder (not a categorical method comparison)"
    )


def assert_ratios_fit_break(curves: Sequence[Dict[str, Any]]) -> None:
    """Refuse to draw if any plotted ratio would hide in the omitted strip."""
    pooled: List[float] = []
    for curve in curves:
        for raw in curve.get("values") or []:
            v = float(raw)
            if MAIN_X_HI < v < STRIP_X_LO or v >= STRIP_X_LO:
                raise ValueError(
                    f"ratio {v} lies in omitted interval ({MAIN_X_HI}, {STRIP_X_LO}) "
                    f"or at/right of the deadline strip (>= {STRIP_X_LO})"
                )
            pooled.append(v)
    if pooled:
        gmax = float(max(pooled))
        if not (gmax < MAIN_X_HI):
            raise ValueError(
                f"global max ratio {gmax} is not < {MAIN_X_HI}; broken axis would clip"
            )


def curves_from_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All plotted series, including the seed-200 densest-admitted overlay."""
    out: List[Dict[str, Any]] = []
    for r in records:
        seed = int(r.get("eval_seed_base", 100))
        out.append({
            "workload": r["workload"],
            "density": float(r.get("delta_sum_bottleneck") or float("nan")),
            "outside": bool(r.get("outside_certified_region")),
            "second_eval": seed == 200,
            "values": list(r.get("ratios") or []),
        })
    return out


def _cdf_xy(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(values, dtype=float))
    if xs.size == 0:
        return xs, xs
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def _break_marks(ax_main: Any, ax_strip: Any) -> None:
    kwargs = dict(transform=ax_main.transAxes, color="k", clip_on=False, linewidth=0.8)
    d = BREAK_DIAG
    ax_main.plot((1.0 - d, 1.0 + d), (-d, +d), **kwargs)
    ax_main.plot((1.0 - d, 1.0 + d), (1.0 - d, 1.0 + d), **kwargs)
    kwargs2 = dict(transform=ax_strip.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_strip.plot((-d, +d), (-d, +d), **kwargs2)
    ax_strip.plot((-d, +d), (1.0 - d, 1.0 + d), **kwargs2)


def _plot_curve(ax: Any, curve: Dict[str, Any], **plot_kw: Any) -> None:
    xs, ys = _cdf_xy(curve["values"])
    if xs.size == 0:
        return
    ax.plot(xs, ys, **plot_kw)


def _legend_handles() -> Tuple[List[Line2D], List[str]]:
    handles = [
        Line2D(
            [0], [0], color=ADMITTED_LEGEND_COLOR, linestyle="-",
            linewidth=LINEWIDTH_ADMITTED,
        ),
        Line2D(
            [0], [0], color=OUTSIDE_COLOR, linestyle="--",
            linewidth=LINEWIDTH_OUTSIDE,
        ),
        Line2D(
            [0], [0], color=SECOND_COLOR, linestyle=":",
            linewidth=LINEWIDTH_SECOND,
        ),
    ]
    return handles, list(LEGEND_LABELS)


def _draw_panel(
    ax_main: Any,
    ax_strip: Any,
    group: Sequence[Dict[str, Any]],
    *,
    show_y_labels: bool,
) -> int:
    admitted = sorted(
        [c for c in group if not c["outside"] and not c.get("second_eval")],
        key=lambda c: c["density"],
    )
    outside = [c for c in group if c["outside"]]
    seconds = [c for c in group if c.get("second_eval")]

    n_drawn = 0
    for curve in seconds:
        _plot_curve(
            ax_main, curve,
            color=SECOND_COLOR,
            linestyle=":",
            linewidth=LINEWIDTH_SECOND,
            zorder=2,
        )
        n_drawn += 1
    for i, curve in enumerate(admitted):
        _plot_curve(
            ax_main, curve,
            color=ADMITTED_COLORS[min(i, len(ADMITTED_COLORS) - 1)],
            linestyle="-",
            linewidth=LINEWIDTH_ADMITTED,
            zorder=3,
        )
        n_drawn += 1
    for curve in outside:
        _plot_curve(
            ax_main, curve,
            color=OUTSIDE_COLOR,
            linestyle="--",
            linewidth=LINEWIDTH_OUTSIDE,
            zorder=3,
        )
        n_drawn += 1

    ax_strip.axvline(
        DEADLINE_X, color="k", linestyle=":", linewidth=1.0, zorder=1,
    )
    ax_strip.text(
        DEADLINE_X, 0.50, "relative deadline",
        rotation=90, va="center", ha="right",
        fontsize=TICK_FS, color="0.25",
        clip_on=False,
    )

    ax_main.set_xlim(MAIN_X_LO, MAIN_X_HI)
    ax_strip.set_xlim(STRIP_X_LO, STRIP_X_HI)
    for ax in (ax_main, ax_strip):
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        ax.legend_ = None
    ax_main.spines["right"].set_visible(False)
    ax_strip.spines["left"].set_visible(False)
    ax_strip.tick_params(left=False, labelleft=False)
    ax_main.tick_params(right=False)
    if not show_y_labels:
        ax_main.tick_params(labelleft=False)
    ax_strip.set_xticks([DEADLINE_X])
    ax_strip.set_xticklabels(["1.0"])
    _break_marks(ax_main, ax_strip)
    ax_main.set_title(
        PANEL_TITLE_BY_WORKLOAD[group[0]["workload"]], fontsize=TITLE_FS
    )
    return n_drawn


def _report_legend_vs_panels(fig: Any, panel_axes: Sequence[Any], leg: Any) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    leg_bb = leg.get_window_extent(renderer=renderer).transformed(inv)
    panel_top = max(ax.get_position().y1 for ax in panel_axes)
    inside = False
    for ax in panel_axes:
        pb = ax.get_position()
        overlap_x = not (leg_bb.x1 <= pb.x0 or leg_bb.x0 >= pb.x1)
        overlap_y = not (leg_bb.y1 <= pb.y0 or leg_bb.y0 >= pb.y1)
        if overlap_x and overlap_y:
            inside = True
            break
    print(
        f"[legend] fig-y [{leg_bb.y0:.4f}, {leg_bb.y1:.4f}]; "
        f"panel top y={panel_top:.4f}; inside_panels={inside}"
    )
    if inside:
        raise ValueError("legend bbox intersects a panel axes bounding box")
    if leg_bb.y0 < panel_top:
        raise ValueError(
            f"legend bottom y={leg_bb.y0:.4f} is below panel top y={panel_top:.4f}"
        )


def draw(curves: Sequence[Dict[str, Any]], out_path: str) -> List[int]:
    assert_ratios_fit_break(curves)
    print(
        "[self-check] all plotted ratios in [0, 0.72); omitted interval empty"
    )
    panels = []
    for key in PANEL_ORDER:
        group = [c for c in curves if c["workload"] == key]
        if group:
            panels.append(group)
    if not panels:
        raise ValueError("no curves to draw")

    fig = plt.figure(figsize=FIGSIZE)
    outer = GridSpec(
        1, len(panels), figure=fig, wspace=0.22,
        left=0.08, right=0.995, bottom=0.18, top=0.78,
    )
    counts: List[int] = []
    first_main = None
    panel_axes: List[Any] = []
    for i, group in enumerate(panels):
        inner = outer[i].subgridspec(1, 2, width_ratios=[7, 1], wspace=0.04)
        sharey = first_main
        ax_main = fig.add_subplot(inner[0, 0], sharey=sharey)
        ax_strip = fig.add_subplot(inner[0, 1], sharey=ax_main)
        panel_axes.extend([ax_main, ax_strip])
        if first_main is None:
            first_main = ax_main
        n = _draw_panel(ax_main, ax_strip, group, show_y_labels=(i == 0))
        counts.append(n)
        print(
            f"[panel] {PANEL_TITLE_BY_WORKLOAD[group[0]['workload']]}: {n} curves"
        )
    if first_main is not None:
        first_main.set_ylabel("fraction of job releases", fontsize=LABEL_FS)
    fig.supxlabel(
        "per-release response time divided by relative deadline",
        fontsize=LABEL_FS,
        y=0.02,
    )
    handles, labels = _legend_handles()
    leg = fig.legend(handles, labels, **LEGEND_KW)
    _report_legend_vs_panels(fig, panel_axes, leg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, **SAVEFIG_KW)
    w, h = fig.get_size_inches()
    print(f"[size] {w:.2f} x {h:.2f} in")
    plt.close(fig)
    return counts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--table", default=tha.DEFAULT_TABLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--caption_out", default=DEFAULT_CAPTION)
    ap.add_argument("--repo", default=_PROJECT_ROOT)
    return ap


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _print_style_provenance()
    repo = os.path.abspath(args.repo)
    table = tha.load_table(args.table)
    root = (
        Path(args.results_root)
        if args.results_root
        else tha.discover_results_root(repo)
    )
    records = tha.analyze_sweep(table, root, repo)
    curves = curves_from_records(records)
    draw(curves, args.out)
    Path(args.caption_out).write_text(CAPTION + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.caption_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
