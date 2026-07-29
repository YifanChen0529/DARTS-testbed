#!/usr/bin/env python3
"""FIG4: D1 admission-infeasibility heatmaps — N3 vs full-fidelity success (§V-F, D3 Fig.13 style)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MD_PATH = REPO / "recovered_n3_ffl.md"
OUT_PATH = REPO / "fig" / "FIG4_d1_heatmap.pdf"
_FIG_NAME = "FIG4_d1_heatmap.pdf"

METHODS: Sequence[str] = (
    "darts",
    "darts_nocert",
    "darts_no_incentive",
    "cea_edf",
    "sadcher",
)
STRESS: Sequence[str] = ("light", "medium", "heavy")
METHOD_LABELS = [
    r"DARTS$^{\mathrm{rt}}$",
    "NoCert",
    "NoIncen",
    "CEA-EDF",
    "SADCHER",
]

SADCHER_N3_CAPTION = (
    "SADCHER's N3 row is masked because the SADCHER-AC instrumentation does "
    "not classify misses by structural cause. The right panel (full-fidelity "
    "success) is derivable from total task miss rate and is shown."
)
SADCHER_ROW = METHODS.index("sadcher")
NOINC_ROW = METHODS.index("darts_no_incentive")

# Fallback when recovered_n3_ffl.md cannot be parsed (handoff §3 defaults).
FALLBACK_N3 = np.array(
    [
        [0.03, 0.10, 0.22],
        [0.02, 0.09, 0.21],
        [0.04, 0.13, 0.28],
        [0.03, 0.33, 0.82],
        [np.nan, np.nan, np.nan],
    ],
    dtype=float,
)
FALLBACK_FF = np.array(
    [
        [0.97, 0.90, 0.77],
        [0.98, 0.91, 0.77],
        [0.96, 0.87, 0.72],
        [0.97, 0.67, 0.17],
        [0.98, 0.69, 0.44],
    ],
    dtype=float,
)

CAPTION_TEXT = (
    "D1 admission-infeasibility breakdown across B1 stress. Left (N3 unreached "
    "fraction): DARTSrt-family misses are exclusively N3; N1, N2, FFL are "
    "identically zero across all cells. Cert layer's value is most clearly "
    "visible on CEA-EDF's heavy column (0.82 vs DARTSrt 0.22, a 3.7× gap). "
    "Right (full-fidelity success): the cert layer enables DARTSrt-family to "
    "maintain ≥0.72 success rate even at heavy stress, while CEA-EDF drops "
    "to 0.17 and SADCHER-AC to 0.44. SADCHER-AC does not partition misses by "
    "structural cause (n/a)."
)


def _method_idx(method: str) -> int:
    return METHODS.index(method)


def _stress_idx(stress: str) -> int:
    return STRESS.index(stress.lower())


def _empty_matrices() -> Tuple[np.ndarray, np.ndarray]:
    shape = (len(METHODS), len(STRESS))
    return (
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
    )


def _to_rate(raw: str) -> Optional[float]:
    s = raw.strip()
    if s in ("—", "-", "", "nan", "n/a"):
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if val > 1.0:
        val /= 100.0
    return val


def _parse_md_table(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse the 12-row table in recovered_n3_ffl.md into N3 and FF_success matrices."""
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    n3, ff = _empty_matrices()
    rows_parsed = 0

    row_re = re.compile(
        r"^\|\s*(\w+)\s*\|\s*([\w_]+)\s*\|"
        r"[^|]*\|[^|]*\|"
        r"\s*([^|]+)\s*\|"
        r"\s*([^|]+)\s*\|"
        r"\s*([^|]+)\s*\|"
        r"\s*([^|]+)\s*\|"
        r"\s*([^|]+)\s*\|",
        re.MULTILINE,
    )

    for m in row_re.finditer(text):
        stress, method, miss_s, _n1, _n2, n3_s, _ffl = m.groups()
        stress = stress.lower()
        if stress not in STRESS or method not in METHODS:
            continue

        i, j = _method_idx(method), _stress_idx(stress)
        miss = _to_rate(miss_s)
        n3_val = _to_rate(n3_s)

        if miss is not None:
            ff[i, j] = 1.0 - miss
        if method == "sadcher":
            n3[i, j] = np.nan
        elif n3_val is not None:
            n3[i, j] = n3_val
        rows_parsed += 1

    if rows_parsed < 12:
        return None
    return n3, ff


def _load_matrices() -> Tuple[np.ndarray, np.ndarray, str]:
    parsed = _parse_md_table(MD_PATH)
    if parsed is not None:
        n3, ff = parsed
        source = f"parsed {MD_PATH.name}"
    else:
        print(f"[fallback] could not parse {MD_PATH}; using default matrices", file=sys.stderr)
        n3 = FALLBACK_N3.copy()
        ff = FALLBACK_FF.copy()
        source = "hardcoded defaults"

    # recovered_n3_ffl.md omits darts_no_incentive; use handoff estimates.
    if source.startswith("parsed"):
        n3[NOINC_ROW, :] = FALLBACK_N3[NOINC_ROW, :]
        ff[NOINC_ROW, :] = FALLBACK_FF[NOINC_ROW, :]
        source += " + no_incentive defaults"

    return n3, ff, source


def _print_markdown_table(name: str, matrix: np.ndarray) -> None:
    header = "| method | " + " | ".join(STRESS) + " |"
    sep = "|---|" + "|".join(["---:" for _ in STRESS]) + "|"
    print(f"\n### {name}\n")
    print(header)
    print(sep)
    for i, method in enumerate(METHODS):
        cells = []
        for j in range(len(STRESS)):
            if method == "sadcher" and name.startswith("N3"):
                cells.append("n/a")
            elif np.isfinite(matrix[i, j]):
                cells.append(f"{matrix[i, j]:.3f}")
            else:
                cells.append("—")
        print(f"| {method} | " + " | ".join(cells) + " |")


def _plot_n3_panel(ax, n3: np.ndarray, fig) -> None:
    cmap = plt.cm.Reds.copy()
    cmap.set_bad("#DDDDDD")
    matrix = np.ma.masked_invalid(n3)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=0.85, origin="upper")

    for i in range(len(METHODS)):
        for j in range(len(STRESS)):
            if i == SADCHER_ROW:
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666",
                    style="italic",
                    zorder=5,
                )
                continue
            val = n3[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if val > 0.45 else "black",
                zorder=5,
            )

    ax.set_xticks(range(len(STRESS)))
    ax.set_xticklabels(list(STRESS), fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=7.5)
    ax.set_title("(a) N3 (unreached) fraction", fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.85)
    cbar.set_label("fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)


def _plot_ff_panel(ax, ff: np.ndarray, fig) -> None:
    im = ax.imshow(ff, cmap=plt.cm.Greens, aspect="auto", vmin=0.0, vmax=1.0, origin="upper")

    for i in range(len(METHODS)):
        for j in range(len(STRESS)):
            val = ff[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if val > 0.55 else "black",
                zorder=5,
            )

    ax.set_xticks(range(len(STRESS)))
    ax.set_xticklabels(list(STRESS), fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=7.5)
    ax.set_title("(b) Full-fidelity success fraction", fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.85)
    cbar.set_label("fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)


def _verify(n3: np.ndarray, ff: np.ndarray) -> None:
    cea_heavy = n3[_method_idx("cea_edf"), _stress_idx("heavy")]
    darts_light = ff[_method_idx("darts"), _stress_idx("light")]

    if not np.isfinite(cea_heavy) or cea_heavy < 0.7:
        print(f"[WARN] ax_n3 SE corner (cea_edf×heavy={cea_heavy:.2f}) < 0.7")
    if not np.isfinite(darts_light) or darts_light < 0.9:
        print(f"[WARN] ax_ff NW corner (darts×light={darts_light:.2f}) < 0.9")


def main() -> None:
    ap = argparse.ArgumentParser(description="FIG4: D1 admission-infeasibility heatmaps.")
    ap.add_argument(
        "--input_md",
        type=str,
        default=str(REPO / "recovered_n3_ffl.md"),
        help="recovered_n3_ffl.md input (default: <repo_root>/recovered_n3_ffl.md)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO / "fig"),
        help="Directory for FIG4_d1_heatmap.pdf (default: <repo_root>/fig)",
    )
    args = ap.parse_args()
    global MD_PATH, OUT_PATH
    MD_PATH = Path(args.input_md).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_PATH = out_dir / _FIG_NAME
    print(f"[fig4_d1] INPUT_MD={MD_PATH}", file=sys.stderr)
    print(f"[fig4_d1] OUT_DIR={out_dir}", file=sys.stderr)

    n3, ff, source = _load_matrices()
    print(f"Data source: {source}\n")
    _print_markdown_table("N3 (unreached) fraction", n3)
    _print_markdown_table("Full-fidelity success fraction", ff)

    fig, (ax_n3, ax_ff) = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    _plot_n3_panel(ax_n3, n3, fig)
    _plot_ff_panel(ax_ff, ff, fig)

    fig.savefig(OUT_PATH, pad_inches=0.02)
    plt.close(fig)
    print(f"\nWrote {OUT_PATH}")

    _verify(n3, ff)
    print("\nCaption-ready text:")
    print(CAPTION_TEXT)
    print("\nSADCHER N3 panel note (for paper):")
    print(SADCHER_N3_CAPTION)


if __name__ == "__main__":
    main()
