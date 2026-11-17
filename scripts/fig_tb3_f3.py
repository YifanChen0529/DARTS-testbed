#!/usr/bin/env python3
"""TB-3 offline acceptance-ratio small-multiples (paper Fig. F3 draft).

There is no shared style helper in this repo. Conventions below are copied
from the published-figure generators (same STYLE_SOURCES list as
scripts/fig_tb1_tradeoff.py): palette / marker / linestyle from
fig1_b1_grid.py and fig2_generalization_scaling.py; typography, line
geometry, and savefig from fig1_b1_grid.py; legend-above-panels from
fig2_generalization_scaling.py; y-grid from fig0_headline_bar.py /
fig2_generalization_scaling.py. Resolved source paths are printed at
startup.

Internal test codes (S0/S1) do not appear in figure text; the mapping is
path-sum test = published workspace-diameter path-sum, per-node travel
refinement = farthest-corner eccentricity (Lemma).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.offline_test_harness import GRID_CELLS_TB3  # noqa: E402

STYLE_SOURCES = (
    "scripts/fig0_headline_bar.py",
    "scripts/fig1_b1_grid.py",
    "scripts/fig2_generalization_scaling.py",
    "scripts/fig3_t3_bound_tightness.py",
    "scripts/fig4_d1_heatmap.py",
    "scripts/fig3_energy_utilization.py",
)

FIGSIZE = (7.16, 1.7)
LINEWIDTH = 1.2
MARKERSIZE = 2.8
MARKEREDGEWIDTH = 0.5
GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.5)
TITLE_FS = 8.5
LABEL_FS = 8
TICK_FS = 7
LEGEND_FS = 7.5
LEGEND_KW = dict(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    frameon=False,
    fontsize=LEGEND_FS,
    handlelength=1.8,
    columnspacing=1.4,
    handletextpad=0.4,
)
SAVEFIG_KW = dict(bbox_inches=None, pad_inches=0.02)

PANEL_TITLES = {
    "B1_CDR:medium": "B1 medium",
    "B1_CDR:heavy": "B1 heavy",
    "B2_LML:medium": "B2",
    "B3_IQI:medium": "B3",
    "B4_DEM:medium": "B4",
}

LABEL_PATH_SUM = "path-sum test"
LABEL_REFINEMENT = "per-node travel refinement"
LABEL_EMPIRICAL = "empirical feasibility (all-clean)"

PATH_SUM_STYLE = {
    "color": "#403990",
    "ls": "-",
    "marker": "o",
    "markerfacecolor": "#403990",
    "markeredgecolor": "white",
}
REFINEMENT_STYLE = {
    "color": "#F46F43",
    "ls": "--",
    "marker": "o",
    "markerfacecolor": "none",
    "markeredgecolor": "#F46F43",
}

X_LIM = (0.5, 2.4)
Y_LIM = (0.0, 1.0)
X_TICKS = [0.5, 1.0, 1.5, 2.0, 2.4]
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]

DEFAULT_CURVES = REPO / "docs" / "tb3" / "tb3_acceptance_curves.csv"
DEFAULT_OUT = REPO / "figures" / "FIG_tb3_f3_draft.pdf"

CAPTION = (
    "Offline acceptance-ratio curves over M=200 randomized mission instances "
    "per cell (per-job WCETs x0.8-1.2, task locations resampled in the "
    "workspace, initial batteries within profile bounds), deadlines uniformly "
    "scaled by the x-axis factor; the dotted line marks deployed deadlines. "
    "The refinement replaces the workspace-diameter travel envelope with a "
    "per-node farthest-corner bound (Lemma)."
)
CAPTION_ADDENDUM = (
    " Black points show the empirically feasible fraction among instances "
    "shared with the offline pool under frozen-policy replay (3 checkpoints "
    "x 5 episodes, all-clean); the deployed-deadline point aggregates 50 "
    "instances and the two relaxed points 10 each; bars are exact 95% "
    "Clopper-Pearson intervals. Accepted instances that are not all-clean "
    "fail only through energy-unreached jobs; no certified job completes "
    "late, no runtime bound is violated, and no episode overruns the mission "
    "deadline with all jobs individually on time. The response-time test "
    "contains no energy term; energy admission is owned by the runtime "
    "energy predicate and reserve sizing."
)
XLABEL = "deadline scale"
YLABEL = "fraction of instances"
EMPIRICAL_CELLS = ("B1_CDR:medium", "B1_CDR:heavy")


def _print_style_provenance() -> None:
    print("[style] no shared helper; conventions copied from:")
    for rel in STYLE_SOURCES:
        print(f"  {str((REPO / rel).resolve())}")


def _load_curves(path: Path) -> Dict[Tuple[str, str], Tuple[List[float], List[float]]]:
    """Return {(cell, test): (betas, accept_frac_e2e)} sorted by beta."""
    by: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["cell"], row["test"])
            by.setdefault(key, []).append(
                (float(row["beta"]), float(row["accept_frac_e2e"]))
            )
    out: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    for key, pairs in by.items():
        pairs.sort(key=lambda p: p[0])
        out[key] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def _load_empirical(
    path: Path,
) -> Dict[str, Tuple[List[float], List[float], List[float], List[float]]]:
    """Return {cell: (beta, feasible_frac, ci_lo, ci_hi)} sorted by beta.

    ``ci_lo`` / ``ci_hi`` are the primary interval (Clopper-Pearson on the
    p7 ceiling CSV; bootstrap on the n=10 P6 file).
    """
    by: Dict[str, List[Tuple[float, float, float, float]]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row["cell"]
            by.setdefault(cell, []).append(
                (
                    float(row["beta"]),
                    float(row["feasible_frac"]),
                    float(row["ci_lo"]),
                    float(row["ci_hi"]),
                )
            )
    out: Dict[str, Tuple[List[float], List[float], List[float], List[float]]] = {}
    for cell, rows in by.items():
        rows.sort(key=lambda p: p[0])
        out[cell] = (
            [p[0] for p in rows],
            [p[1] for p in rows],
            [p[2] for p in rows],
            [p[3] for p in rows],
        )
    return out


def _plot_offline(ax, xs: List[float], ys: List[float], style: dict) -> None:
    ax.plot(
        xs,
        ys,
        color=style["color"],
        ls=style["ls"],
        marker=style["marker"],
        markersize=MARKERSIZE,
        markerfacecolor=style["markerfacecolor"],
        markeredgecolor=style["markeredgecolor"],
        markeredgewidth=MARKEREDGEWIDTH,
        linewidth=LINEWIDTH,
        zorder=3,
    )


def _legend_handles(*, empirical: bool = False) -> Tuple[List[Line2D], List[str]]:
    h_s0 = Line2D(
        [0],
        [0],
        color=PATH_SUM_STYLE["color"],
        ls=PATH_SUM_STYLE["ls"],
        marker=PATH_SUM_STYLE["marker"],
        markersize=MARKERSIZE,
        markerfacecolor=PATH_SUM_STYLE["markerfacecolor"],
        markeredgecolor=PATH_SUM_STYLE["markeredgecolor"],
        markeredgewidth=MARKEREDGEWIDTH,
        linewidth=LINEWIDTH,
    )
    h_s1 = Line2D(
        [0],
        [0],
        color=REFINEMENT_STYLE["color"],
        ls=REFINEMENT_STYLE["ls"],
        marker=REFINEMENT_STYLE["marker"],
        markersize=MARKERSIZE,
        markerfacecolor=REFINEMENT_STYLE["markerfacecolor"],
        markeredgecolor=REFINEMENT_STYLE["markeredgecolor"],
        markeredgewidth=MARKEREDGEWIDTH,
        linewidth=LINEWIDTH,
    )
    handles = [h_s0, h_s1]
    labels = [LABEL_PATH_SUM, LABEL_REFINEMENT]
    if empirical:
        h_emp = Line2D(
            [0],
            [0],
            color="k",
            ls="none",
            marker="o",
            markersize=MARKERSIZE,
            markerfacecolor="k",
            markeredgecolor="k",
            markeredgewidth=MARKEREDGEWIDTH,
        )
        handles.append(h_emp)
        labels.append(LABEL_EMPIRICAL)
    return handles, labels


def rendered_strings(*, empirical: bool = False) -> str:
    """Paper-facing strings only (caption, legend, axis labels, panel titles)."""
    caption = CAPTION + (CAPTION_ADDENDUM if empirical else "")
    labels = [LABEL_PATH_SUM, LABEL_REFINEMENT]
    if empirical:
        labels.append(LABEL_EMPIRICAL)
    parts = [caption, XLABEL, YLABEL, *labels, *PANEL_TITLES.values()]
    return "\n".join(parts)


def render(
    curves_path: Path,
    out_path: Path,
    empirical_path: Optional[Path] = None,
) -> Path:
    series = _load_curves(curves_path)
    empirical = _load_empirical(empirical_path) if empirical_path else None

    fig, axes = plt.subplots(1, len(GRID_CELLS_TB3), figsize=FIGSIZE, sharey=True)
    for col, cell in enumerate(GRID_CELLS_TB3):
        ax = axes[col]
        xs0, ys0 = series[(cell, "S0")]
        xs1, ys1 = series[(cell, "S1")]
        _plot_offline(ax, xs0, ys0, PATH_SUM_STYLE)
        _plot_offline(ax, xs1, ys1, REFINEMENT_STYLE)
        if empirical is not None and cell in EMPIRICAL_CELLS and cell in empirical:
            ex, ey, elo, ehi = empirical[cell]
            yerr = (
                [y - lo for y, lo in zip(ey, elo)],
                [hi - y for y, hi in zip(ey, ehi)],
            )
            ax.errorbar(
                ex,
                ey,
                yerr=yerr,
                fmt="ko",
                markersize=MARKERSIZE,
                linewidth=LINEWIDTH,
                capsize=1.5,
                zorder=5,
            )
        ax.axvline(1.0, color="k", linestyle=":", linewidth=1.0, zorder=1)
        ax.set_title(PANEL_TITLES[cell], fontsize=TITLE_FS)
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.set_xticks(X_TICKS)
        ax.set_yticks(Y_TICKS)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, axis="y", **GRID_KW)
        ax.set_axisbelow(True)
        ax.set_xlabel(XLABEL, fontsize=LABEL_FS)
        if col == 0:
            ax.set_ylabel(YLABEL, fontsize=LABEL_FS)

    use_emp = empirical is not None
    handles, labels = _legend_handles(empirical=use_emp)
    for lab in labels:
        print(f"[legend] {lab}")
    fig.legend(handles, labels, ncol=len(handles), **LEGEND_KW)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86], h_pad=0.40, w_pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Wrote {out_path}")
    caption_path = out_path.with_name("FIG_tb3_f3_caption.txt")
    caption_text = CAPTION + (CAPTION_ADDENDUM if use_emp else "")
    caption_path.write_text(caption_text + "\n", encoding="utf-8")
    print(f"Wrote {caption_path}")
    return out_path


def main() -> int:
    _print_style_provenance()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", default=str(DEFAULT_CURVES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--empirical",
        default=None,
        help="Optional empirical CSV (cell, beta, n_instances, n_feasible, "
        "feasible_frac, ci_lo, ci_hi). Draft run omits this.",
    )
    args = ap.parse_args()
    emp = Path(args.empirical) if args.empirical else None
    render(Path(args.curves), Path(args.out), empirical_path=emp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
