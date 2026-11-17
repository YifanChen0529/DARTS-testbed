#!/usr/bin/env python3
"""TB-1 envelope-scaling tradeoff small-multiples (paper Fig. F1).

There is no shared style helper in this repo. Conventions below are copied
verbatim from the DATA.md §6 published-figure generators (and
fig3_energy_utilization.py): palette / marker / linestyle from fig1_b1_grid.py
and fig2_generalization_scaling.py; 2×N grid typography, line geometry, and
savefig from fig1_b1_grid.py; legend-above-panels from fig2_generalization_scaling.py;
y-grid from fig0_headline_bar.py / fig2_generalization_scaling.py. Resolved source
paths are printed at startup.

Always 4 columns. Utilization U is not drawn: per-cell max |U(ζ)-U(1)| is
printed and written to a caption fragment. U remains in tb1_per_cell.csv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.tb1_analysis import (  # noqa: E402
    _mean,
    collect_cell_series,
    discover_kappas,
    iter_cell_runs,
)

STYLE_SOURCES = (
    "scripts/fig0_headline_bar.py",
    "scripts/fig1_b1_grid.py",
    "scripts/fig2_generalization_scaling.py",
    "scripts/fig3_t3_bound_tightness.py",
    "scripts/fig4_d1_heatmap.py",
    "scripts/fig3_energy_utilization.py",
)

FIGSIZE = (7.16, 2.4)
LINEWIDTH = 1.2
MARKERSIZE = 2.8
MARKEREDGEWIDTH = 0.5
GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.5)
TITLE_FS = 8.5
LABEL_FS = 8
TICK_FS = 7  # fig1_b1_grid.py tick_params(labelsize=7)
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
BAND_ALPHA = 0.18
UNSOUND_FACE = "0.92"
UNSOUND_EDGE = "0.55"
UNSOUND_HATCH = "///"
UNSOUND_ALPHA = 1.0
UNSOUND_ZORDER = 0  # below data lines (3) and seed min–max bands (2)
UNSOUND_PATCH_KW = dict(
    facecolor=UNSOUND_FACE,
    edgecolor=UNSOUND_EDGE,
    hatch=UNSOUND_HATCH,
    alpha=UNSOUND_ALPHA,
)

SERIES_STYLE = {
    "bid_accept": {"color": "#403990", "ls": "-", "marker": "o"},
    "rho_hat": {"color": "#D9A441", "ls": ":", "marker": "D"},
    "U_team": {"color": "#58A14C", "ls": "-", "marker": "v"},
    "task_miss_rate": {"color": "#F46F43", "ls": "--", "marker": "s"},
    "certified_miss": {"color": "#6C91C2", "ls": "-.", "marker": "^"},
}
SPLIT_UNREACHED_COLOR = "#B5A4D6"

TOP_SERIES: Sequence[Tuple[str, str]] = (
    ("bid_accept", "admissible-bid fraction"),
    ("rho_hat", r"$\hat{\rho}$"),
)
BOTTOM_SERIES: Sequence[Tuple[str, str]] = (
    ("task_miss_rate", "task miss rate"),
    ("certified_miss", "full-fidelity miss rate"),
)
SPLIT_LABELS = (
    "full-fidelity miss (late)",
    "full-fidelity miss (unreached)",
)
UNSOUND_LABEL = "unsound region (ζ<1)"

COLUMNS: Sequence[Tuple[str, str, str, float]] = (
    ("B1_CDR", "medium", "B1 medium", 1.0),
    ("B1_CDR", "heavy", "B1 heavy", 1.0),
    ("B2_LML", "medium", "B2 medium", 1.0),
    ("B1_CDR", "heavy", "B1 heavy (deadlines ×0.7)", 0.7),
)
ZETA_TICKS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
ZETA_TICK_LABELS = ["0.25", "0.5", "0.75", "1", "1.5", "2", "3", "4"]
N_DEPLOYED = 3  # first three COLUMNS: published deadlines (β=1.0)
TOP_YLIM = (0.0, 1.02)
TOP_YTICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
PANEL4_BOT_YLIM = (0.0, 1.0)
PANEL4_BOT_YTICKS = [0.0, 0.5, 1.0]

DEFAULT_RESULTS = REPO / "results_tb1_p0"
DEFAULT_OUT = REPO / "figures" / "FIG_tb1_tradeoff.pdf"


def _print_style_provenance() -> None:
    print("[style] no shared helper; conventions copied from:")
    for rel in STYLE_SOURCES:
        print(f"  {str((REPO / rel).resolve())}")


def _seeds_for(results_dir: str, bench: str, stress: str, beta: float) -> List[int]:
    return sorted(
        {
            seed
            for _name, _k, seed, b in iter_cell_runs(results_dir, bench, stress)
            if b == float(beta)
        }
    )


def zeta_to_ordinal(z: float) -> float:
    """Map a ζ value onto equal-interval index positions of ZETA_TICKS.

    Labeled ticks sit at 0..n-1. Values between ticks (e.g. 1.25 between 1 and 1.5)
    interpolate; values outside the labeled set extrapolate on the end interval.
    """
    ticks = ZETA_TICKS
    n = len(ticks)
    if z <= ticks[0]:
        span = ticks[1] - ticks[0]
        return (z - ticks[0]) / span
    if z >= ticks[-1]:
        span = ticks[-1] - ticks[-2]
        return (n - 1) + (z - ticks[-1]) / span
    for i in range(n - 1):
        lo, hi = ticks[i], ticks[i + 1]
        if lo <= z <= hi:
            return i + (z - lo) / (hi - lo)
    raise AssertionError(f"unmapped ζ={z}")


def _xy(
    series: Dict[str, Dict[float, List[float]]], metric: str, kappas: Sequence[float]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    means: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    for k in kappas:
        vals = [v for v in series[metric][k] if not math.isnan(v)]
        if not vals:
            continue
        xs.append(zeta_to_ordinal(k))
        means.append(_mean(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    return xs, means, lo, hi


def _plot_series(
    ax,
    series: Dict[str, Dict[float, List[float]]],
    metric: str,
    label: str,
    kappas: Sequence[float],
    *,
    band: bool = True,
) -> None:
    st = SERIES_STYLE[metric]
    xs, means, lo, hi = _xy(series, metric, kappas)
    if not xs:
        return
    ax.plot(
        xs,
        means,
        color=st["color"],
        ls=st["ls"],
        marker=st["marker"],
        markersize=MARKERSIZE,
        markevery=1,
        markeredgewidth=MARKEREDGEWIDTH,
        markeredgecolor="white",
        linewidth=LINEWIDTH,
        label=label,
        zorder=3,
    )
    if band:
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=BAND_ALPHA, linewidth=0, zorder=2)


def _apply_shared_x(ax) -> None:
    n = len(ZETA_TICKS)
    ax.set_xlim(-0.35, n - 1 + 0.35)
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(ZETA_TICK_LABELS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.axvline(zeta_to_ordinal(1.0), color="k", linestyle=":", linewidth=1.0, zorder=1)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y", **GRID_KW)
    ax.set_axisbelow(True)


def _bottom_ylim(series: Dict[str, Dict[float, List[float]]], kappas: Sequence[float], split: bool) -> Tuple[float, float]:
    hi = 0.0
    metrics = ["task_miss_rate", "certified_miss"]
    if split:
        metrics.extend(["certified_miss_late", "certified_miss_unreached"])
    for metric in metrics:
        for k in kappas:
            for v in series.get(metric, {}).get(k, []):
                if not math.isnan(v):
                    hi = max(hi, v)
    if hi <= 0:
        hi = 0.05
    return (0.0, hi * 1.08)


def _plot_split_stack(
    ax,
    series: Dict[str, Dict[float, List[float]]],
    kappas: Sequence[float],
) -> None:
    xs: List[float] = []
    late: List[float] = []
    unr: List[float] = []
    for k in kappas:
        lv = [v for v in series["certified_miss_late"][k] if not math.isnan(v)]
        uv = [v for v in series["certified_miss_unreached"][k] if not math.isnan(v)]
        if not lv and not uv:
            continue
        xs.append(zeta_to_ordinal(k))
        late.append(_mean(lv) if lv else 0.0)
        unr.append(_mean(uv) if uv else 0.0)
    if not xs:
        return
    ax.stackplot(
        xs,
        late,
        unr,
        colors=[SERIES_STYLE["certified_miss"]["color"], SPLIT_UNREACHED_COLOR],
        alpha=0.35,
        labels=list(SPLIT_LABELS),
        zorder=1,
    )


def _max_abs_du(series: Dict[str, Dict[float, List[float]]], kappas: Sequence[float]) -> float:
    u1_vals = [v for v in series.get("U_team", {}).get(1.0, []) if not math.isnan(v)]
    if not u1_vals:
        return float("nan")
    u1 = _mean(u1_vals)
    worst = 0.0
    for k in kappas:
        vals = [v for v in series["U_team"][k] if not math.isnan(v)]
        if not vals:
            continue
        worst = max(worst, abs(_mean(vals) - u1))
    return worst


def _unsound_span(
    series: Dict[str, Dict[float, List[float]]], kappas: Sequence[float]
) -> Optional[Tuple[float, float]]:
    base_vals = [v for v in series.get("certified_miss", {}).get(1.0, []) if not math.isnan(v)]
    if not base_vals:
        return None
    base = _mean(base_vals)
    bad = [
        k for k in kappas
        if k < 1.0
        and any(not math.isnan(v) for v in series["certified_miss"][k])
        and _mean([v for v in series["certified_miss"][k] if not math.isnan(v)]) > base
    ]
    if not bad:
        return None
    return (zeta_to_ordinal(min(bad)), zeta_to_ordinal(1.0))


def _load_column(
    results_dir: str, bench: str, stress: str, beta: float
) -> Tuple[Dict[str, Dict[float, List[float]]], List[float]]:
    kappas = discover_kappas(results_dir, bench, stress, beta)
    seeds = _seeds_for(results_dir, bench, stress, beta)
    series = collect_cell_series(results_dir, bench, stress, seeds, kappas, beta)
    return series, kappas


def caption_text(deployed_bound: float) -> str:
    if math.isnan(deployed_bound):
        head = "System utilization U could not be compared to U(ζ=1) (no ζ=1.0 rows)"
    else:
        head = (
            f"System utilization U stays within {deployed_bound:.4f} of U(ζ=1) "
            "across the sweep, ζ sampled at the labeled values (ordinal spacing)"
        )
    return (
        f"{head}. Dotted line: deployed calibration ζ=1. "
        "In the tightened-deadline probe, utilization falls with admission for ζ≥3. "
        "Shaded bands: seed min–max across the three frozen policies; "
        "hatched area: certificate evaluated below calibration (unsound).\n"
    )


def _write_caption(path: Path, deployed_bound: float) -> None:
    text = caption_text(deployed_bound)
    path.write_text(text)
    print(f"[caption] {path}: {text.strip()}")


def _legend_handles() -> Tuple[List, List[str]]:
    handles: List = []
    labels: List[str] = []
    for metric, label in list(TOP_SERIES) + list(BOTTOM_SERIES):
        st = SERIES_STYLE[metric]
        handles.append(
            Line2D(
                [0], [0],
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"],
                markersize=MARKERSIZE,
                markeredgewidth=MARKEREDGEWIDTH,
                markeredgecolor="white",
                linewidth=LINEWIDTH,
            )
        )
        labels.append(label)
    handles.append(Patch(**UNSOUND_PATCH_KW))
    labels.append(UNSOUND_LABEL)
    return handles, labels


def render(
    results_dir: str,
    out_path: Path,
    *,
    split: bool,
) -> Path:
    loaded: List[Tuple[Dict[str, Dict[float, List[float]]], List[float]]] = []
    du_values: List[float] = []
    for bench, stress, title, col_beta in COLUMNS:
        series, kappas = _load_column(results_dir, bench, stress, col_beta)
        loaded.append((series, kappas))
        du = _max_abs_du(series, kappas)
        du_values.append(du)

    deployed = du_values[:N_DEPLOYED]
    probe = du_values[N_DEPLOYED:]
    for (bench, stress, title, _cb), du in zip(COLUMNS[:N_DEPLOYED], deployed):
        print(f"[U] deployed  {title:<32} max |U(ζ)-U(1)| = {_fmt_du(du)}")
    bound = max((v for v in deployed if not math.isnan(v)), default=float("nan"))
    print(f"[U] deployed-deadline bound                 max |U(ζ)-U(1)| = {_fmt_du(bound)}")
    for (bench, stress, title, _cb), du in zip(COLUMNS[N_DEPLOYED:], probe):
        print(f"[U] probe     {title:<32} max |U(ζ)-U(1)| = {_fmt_du(du)}")

    caption_path = out_path.with_name("FIG_tb1_tradeoff_caption.txt")
    _write_caption(caption_path, bound)

    fig, axes = plt.subplots(2, len(COLUMNS), figsize=FIGSIZE, sharex=True)

    for col, ((bench, stress, title, _cb), (series, kappas)) in enumerate(zip(COLUMNS, loaded)):
        ax_t = axes[0, col]
        ax_b = axes[1, col]
        for metric, label in TOP_SERIES:
            _plot_series(ax_t, series, metric, label, kappas)
        for metric, label in BOTTOM_SERIES:
            _plot_series(ax_b, series, metric, label, kappas)
        if split:
            _plot_split_stack(ax_b, series, kappas)

        if col == len(COLUMNS) - 1:
            span = _unsound_span(series, kappas)
            if span is not None:
                ax_b.axvspan(
                    span[0], span[1],
                    zorder=UNSOUND_ZORDER,
                    **UNSOUND_PATCH_KW,
                )

        ax_t.set_title(title, fontsize=TITLE_FS)
        ax_t.set_ylim(*TOP_YLIM)
        ax_t.set_yticks(TOP_YTICKS)
        if col == len(COLUMNS) - 1:
            ax_b.set_ylim(*PANEL4_BOT_YLIM)
            ax_b.set_yticks(PANEL4_BOT_YTICKS)
        else:
            ax_b.set_ylim(*_bottom_ylim(series, kappas, split))
            ax_b.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax_b.set_xlabel(r"envelope-scaling factor $\zeta$", fontsize=LABEL_FS)
        _apply_shared_x(ax_t)
        _apply_shared_x(ax_b)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, ncol=len(handles), **LEGEND_KW)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.86], h_pad=0.40, w_pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **SAVEFIG_KW)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def _fmt_du(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def main() -> int:
    _print_style_provenance()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--split",
        action="store_true",
        help="Stack full-fidelity miss late / unreached under the miss series.",
    )
    args = ap.parse_args()
    render(args.results_dir, Path(args.out), split=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
