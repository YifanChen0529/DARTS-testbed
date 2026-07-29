#!/usr/bin/env python3
"""FIG3 (T3): Theorem 2 graph-response bound tightness across DBench-4 (§V-G)."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MD_PATH = REPO / "vg_t3_combined.md"
OUT_PATH = REPO / "fig" / "FIG3_t3_tightness.pdf"
_FIG_NAME = "FIG3_t3_tightness.pdf"

BENCH_LABEL = {
    "B1_CDR": "B1",
    "B2_LML": "B2",
    "B3_IQI": "B3",
    "B4_DEM": "B4",
}

CELL_ORDER: Sequence[Tuple[str, str]] = (
    ("B1_CDR", "light"),
    ("B1_CDR", "medium"),
    ("B1_CDR", "heavy"),
    ("B2_LML", "medium"),
    ("B3_IQI", "medium"),
    ("B4_DEM", "medium"),
)

FALLBACK_CELLS = [
    ("B1 / light", 240250, 109678, 2.19),
    ("B1 / medium", 240610, 103382, 2.33),
    ("B1 / heavy", 240250, 100540, 2.39),
    ("B2 / medium", 255753, 94403, 2.71),
    ("B3 / medium", 255753, 121693, 2.10),
    ("B4 / medium", 75050, 57059, 1.32),
]

CAPTION_TEXT = (
    "Theorem 2 graph-response bound is empirically tight across DBench-4. B4 "
    "parallel topology achieves 1.32× bound tightness (near-tight, theoretical "
    "limit due to single-edge structure). Diamond and chain topologies show "
    "2.0–2.7× tightness; the loosening on B2/B3 stems from the workspace-"
    "diameter $\\tau$ lifting (Theorem 2(a)), with per-edge $\\tau$ tightening "
    "deferred to the journal extension."
)


@dataclass
class T3Cell:
    label: str
    r_off: float
    r_measured: float
    ratio: float

    @property
    def flag(self) -> str:
        if self.ratio < 1.5:
            return "NEAR-TIGHT"
        if self.ratio < 2.5:
            return "STANDARD"
        return "LOOSE (workspace-diameter τ̄ lifting)"


def _parse_ratio(s: str) -> Optional[float]:
    m = re.search(r"([\d.]+)", s.replace("×", "").replace("x", ""))
    return float(m.group(1)) if m else None


def _parse_measured(s: str) -> Optional[float]:
    s = s.strip()
    if s in ("—", "-", ""):
        return None
    head = s.split("±")[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def _parse_md(path: Path) -> Optional[List[T3Cell]]:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    by_key: dict[Tuple[str, str], T3Cell] = {}
    row_re = re.compile(
        r"^\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*[^|]+\|\s*([^|]+)\|",
        re.MULTILINE,
    )

    for m in row_re.finditer(text):
        bench, stress, method, r_off_s, r_meas_s, ratio_s = m.groups()
        if method != "darts":
            continue
        key = (bench, stress)
        if key not in CELL_ORDER:
            continue
        r_off = _parse_measured(r_off_s)
        r_meas = _parse_measured(r_meas_s)
        ratio = _parse_ratio(ratio_s)
        if r_off is None or r_meas is None or ratio is None:
            continue
        label = f"{BENCH_LABEL[bench]} / {stress}"
        by_key[key] = T3Cell(label, r_off, r_meas, ratio)

    if len(by_key) < len(CELL_ORDER):
        return None
    return [by_key[key] for key in CELL_ORDER]


def _load_cells() -> Tuple[List[T3Cell], str]:
    parsed = _parse_md(MD_PATH)
    if parsed is not None:
        return parsed, f"parsed {MD_PATH.name}"

    print("[FALLBACK from handoff §3]", file=sys.stderr)
    cells = [
        T3Cell(label, r_off, r_meas, ratio)
        for label, r_off, r_meas, ratio in FALLBACK_CELLS
    ]
    return cells, "hardcoded defaults"


def _print_table(cells: Sequence[T3Cell]) -> None:
    header = f"{'cell':<14} {'R_off':>10} {'R_measured':>12} {'ratio':>8} {'flag'}"
    print(header)
    print("-" * len(header))
    for c in cells:
        print(f"{c.label:<14} {c.r_off:10.0f} {c.r_measured:12.0f} {c.ratio:8.2f} {c.flag}")


def _verify(cells: Sequence[T3Cell]) -> None:
    b4 = next(c for c in cells if c.label.startswith("B4"))
    b1_med = next(c for c in cells if c.label == "B1 / medium")

    if b4.ratio >= 1.5:
        print(f"[WARN] B4 ratio={b4.ratio:.2f} is not near-tight (expected < 1.5)")

    for c in cells:
        if c.r_off <= c.r_measured:
            print(
                f"[ERROR] bound violated at {c.label}: "
                f"R_off={c.r_off:.0f} ≤ R_measured={c.r_measured:.0f}"
            )

    if not (2.0 <= b1_med.ratio <= 2.5):
        print(
            f"[WARN] B1/medium ratio={b1_med.ratio:.2f} outside advisor fork-join band [2.0, 2.5]"
        )


def _plot(cells: Sequence[T3Cell]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.4))

    y_centers = np.arange(len(cells) - 1, -1, -1, dtype=float)
    bar_h = 0.32
    y_off = 0.18
    min_inbar_ms = 25000

    for i, (cell, yc) in enumerate(zip(cells, y_centers)):
        ax.barh(
            yc + y_off,
            cell.r_off,
            height=bar_h,
            color="#5D6066",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.9,
            label="Analytical bound $\\bar{R}^{\\mathrm{off}}_G$" if i == 0 else None,
        )
        ax.barh(
            yc - y_off,
            cell.r_measured,
            height=bar_h,
            color="#403990",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.9,
            label="DARTS$^{\\mathrm{rt}}$ measured $R_G$" if i == 0 else None,
        )

        if cell.r_off > min_inbar_ms:
            ax.text(
                cell.r_off * 0.5,
                yc + y_off,
                f"{cell.r_off / 1000:.0f}k",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=3,
            )
        if cell.r_measured > min_inbar_ms:
            ax.text(
                cell.r_measured * 0.5,
                yc - y_off,
                f"{cell.r_measured / 1000:.0f}k",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        is_b4 = cell.label.startswith("B4")
        ratio_color = "#1B7E3D" if is_b4 else "#222"
        ratio_weight = "bold" if is_b4 else "normal"
        ratio_size = 8 if is_b4 else 7.5
        y_ax = (5 - i) / 5.5
        ax.text(
            1.02,
            y_ax,
            f"{cell.ratio:.2f}×",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=ratio_size,
            fontweight=ratio_weight,
            color=ratio_color,
            clip_on=False,
        )
        if is_b4:
            ax.text(
                1.02,
                y_ax - 0.04,
                "★ near-tight",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                fontweight="bold",
                fontstyle="italic",
                color="#1B7E3D",
                clip_on=False,
            )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([c.label for c in cells], fontsize=7.5)
    ax.set_xlabel(r"Graph response time $R_G$ [ms]", fontsize=8)
    ax.set_xlim(0, 280000)
    ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000])
    ax.set_xticklabels(["0", "50k", "100k", "150k", "200k", "250k"], fontsize=7)
    ax.set_title("Theorem 2 bound tightness across DBench-4", fontsize=8.5, pad=6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.24, right=0.78, top=0.90, bottom=0.22)

    fig.savefig(OUT_PATH, bbox_inches=None, pad_inches=0.04)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"FIG 3 tight bbox: {bbox} (figsize was {fig.get_size_inches()})")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="FIG3: Theorem-2 graph-response bound tightness.")
    ap.add_argument(
        "--input_md",
        type=str,
        default=str(REPO / "vg_t3_combined.md"),
        help="vg_t3_combined.md input (default: <repo_root>/vg_t3_combined.md)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO / "fig"),
        help="Directory for FIG3_t3_tightness.pdf (default: <repo_root>/fig)",
    )
    args = ap.parse_args()
    global MD_PATH, OUT_PATH
    MD_PATH = Path(args.input_md).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_PATH = out_dir / _FIG_NAME
    print(f"[fig3_t3] INPUT_MD={MD_PATH}", file=sys.stderr)
    print(f"[fig3_t3] OUT_DIR={out_dir}", file=sys.stderr)

    cells, source = _load_cells()
    print(f"Data source: {source}\n")
    _print_table(cells)
    _plot(cells)
    print(f"\nWrote {OUT_PATH}")
    _verify(cells)
    print("\nCaption-ready text:")
    print(CAPTION_TEXT)


if __name__ == "__main__":
    main()
