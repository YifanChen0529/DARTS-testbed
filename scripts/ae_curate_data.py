#!/usr/bin/env python3
"""AE T7 — assemble curated behavioral data package under data/.

Three destination trees (paths match Makefile BEH_A1 / BEH_LEG / BEH_D1):

  A. data/raw/behavioral/sweep_a1a2b1/   ← results_a1a2b1_20260521 (CSV-only)
  B. data/raw/behavioral/legacy_medium/ ← results/ legacy B2–B4 dirs (CSV-only)
  C. data/raw/behavioral/sweep_d1/      ← merged Table III input (333 files)

Header cleaning: strip whitespace on FIRST line fields only; data rows unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent

SRC_A1 = REPO / "results_a1a2b1_20260521"
SRC_RESULTS = REPO / "results"

METHODS = (
    "darts",
    "darts_nocert",
    "darts_no_incentive",
    "cea_edf",
    "sadcher",
)
A1_STRESSES = (
    "light",
    "medium",
    "heavy",
    "medium_n2",
    "medium_n6",
    "medium_n8",
    "energy_b02",
    "energy_b03",
    "energy_b04",
)

LEGACY_DIR_RE = re.compile(
    r"^(darts|darts_nocert|darts_no_incentive|cea_edf|sadcher)"
    r"_(B2_LML|B3_IQI|B4_DEM)"
    r"_(light|medium|heavy)_seed([0-9]+)$"
)

# Exact Table III selection for sweep_d1. Total MUST equal 333.
# Each entry: (src_root_key, exp_lo, exp_hi, bench, stress, methods)
# src_root_key ∈ {"a1", "results_exp"}; legacy handled separately.
SELECTION: List[Tuple[str, int, int, str, str, Tuple[str, ...]]] = [
    # 1. From results_a1a2b1_20260521
    ("a1", 1, 10, "B1", "light", METHODS),
    ("a1", 1, 10, "B1", "medium", METHODS),
    ("a1", 1, 10, "B1", "heavy", METHODS),
    ("a1", 1, 10, "B2", "medium", ("darts_no_incentive",)),
    # 2. From results/ exp layout
    ("results_exp", 1, 10, "B2", "medium", ("darts", "darts_nocert", "cea_edf", "sadcher")),
    ("results_exp", 1, 10, "B2", "heavy", ("darts", "darts_nocert", "cea_edf", "sadcher")),
    ("results_exp", 1, 3, "B2", "light", ("darts_nocert",)),
    ("results_exp", 1, 10, "B3", "medium", ("darts", "cea_edf", "sadcher")),
    ("results_exp", 1, 10, "B4", "medium", ("darts", "cea_edf", "sadcher")),
]

LEGACY_B2_LIGHT_METHODS = ("darts", "cea_edf", "sadcher")
LEGACY_B2_LIGHT_SEEDS = tuple(range(100, 1001, 100))  # 100..1000

SWEEP_D1_EXPECTED_TOTAL = 333

CopyPlan = List[Tuple[Path, Path]]  # (src, dest)


def _src_root(key: str) -> Path:
    if key == "a1":
        return SRC_A1
    if key == "results_exp":
        return SRC_RESULTS
    raise ValueError(key)


def _looks_like_header(line: str) -> bool:
    fields = line.rstrip("\n\r").split(",")
    if not fields:
        return False
    return any(any(c.isalpha() for c in f) for f in fields)


def clean_header_copy(src: Path, dest: Path) -> Tuple[int, int]:
    """Copy src→dest stripping whitespace on first-line fields only.

    Returns (n_rows including header, n_cols). Data rows are byte-identical
    to the source bytes after the first newline.
    """
    raw = src.read_bytes()
    if b"\n" in raw:
        first, rest = raw.split(b"\n", 1)
    else:
        first, rest = raw, b""
    first_text = first.decode("utf-8", errors="replace").rstrip("\r")
    if not _looks_like_header(first_text):
        raise SystemExit(
            f"ERROR: first line does not look like a CSV header: {src}\n"
            f"  first_line={first_text[:200]!r}"
        )
    cleaned = ",".join(f.strip() for f in first_text.split(","))
    n_cols = len(cleaned.split(","))
    body = rest.decode("utf-8", errors="replace")
    data_lines = [ln for ln in body.splitlines() if ln.strip() != ""]
    n_rows = 1 + len(data_lines)

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = cleaned.encode("utf-8") + b"\n"
    if rest:
        out += rest
        if not rest.endswith(b"\n"):
            out += b"\n"
    dest.write_bytes(out)
    return n_rows, n_cols


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_sweep_a1a2b1(dest_root: Path) -> CopyPlan:
    plan: CopyPlan = []
    if not SRC_A1.is_dir():
        raise SystemExit(f"ERROR: missing source tree {SRC_A1}")
    # Skip top-level backup trees (e.g. _partial_noinc_backup_*).
    for exp_dir in sorted(SRC_A1.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        b1 = exp_dir / "B1"
        if not b1.is_dir():
            continue
        for stress in A1_STRESSES:
            sdir = b1 / stress
            if not sdir.is_dir():
                continue
            for csv in sorted(sdir.glob("*_log.csv")):
                rel = csv.relative_to(SRC_A1)
                plan.append((csv, dest_root / rel))
    return plan


def _legacy_log_src(dir_path: Path, method: str) -> Path:
    """Resolve the behavioral CSV inside a legacy run directory.

    Most methods write ``{method}_log.csv``. Ablation variants
    ``darts_nocert`` / ``darts_no_incentive`` historically wrote ``darts_log.csv``;
    accept that fallback so the curated tree is complete. Destination always
    uses ``{method}_log.csv`` so Makefile consumers (fig2/vg_*) can glob it.
    """
    preferred = dir_path / f"{method}_log.csv"
    if preferred.is_file():
        return preferred
    fallback = dir_path / "darts_log.csv"
    if method.startswith("darts") and fallback.is_file():
        return fallback
    raise SystemExit(
        f"ERROR: legacy dir missing {method}_log.csv"
        + (f" (and no darts_log.csv fallback)" if method.startswith("darts") else "")
        + f": {dir_path}"
    )


def plan_legacy_medium(dest_root: Path) -> CopyPlan:
    plan: CopyPlan = []
    if not SRC_RESULTS.is_dir():
        raise SystemExit(f"ERROR: missing source tree {SRC_RESULTS}")
    for d in sorted(SRC_RESULTS.iterdir()):
        if not d.is_dir():
            continue
        m = LEGACY_DIR_RE.match(d.name)
        if not m:
            continue
        method = m.group(1)
        csv = _legacy_log_src(d, method)
        plan.append((csv, dest_root / d.name / f"{method}_log.csv"))
    return plan


def _collect_exp_cell(
    root: Path,
    exp_lo: int,
    exp_hi: int,
    bench: str,
    stress: str,
    method: str,
) -> List[Tuple[int, Path]]:
    """Return (exp_idx, path) for existing files; abort if extras beyond exp_hi."""
    found: List[Tuple[int, Path]] = []
    extras: List[Path] = []
    for exp_dir in sorted(root.glob("exp_*")):
        m = re.match(r"^exp_(\d+)$", exp_dir.name)
        if not m:
            continue
        idx = int(m.group(1))
        csv = exp_dir / bench / stress / f"{method}_log.csv"
        if not csv.is_file():
            continue
        if exp_lo <= idx <= exp_hi:
            found.append((idx, csv))
        elif idx > exp_hi:
            extras.append(csv)
    if extras:
        raise SystemExit(
            f"ERROR: cell {bench}/{stress}/{method} has files beyond exp_{exp_hi}:\n"
            + "\n".join(f"  {p}" for p in extras)
        )
    expected = exp_hi - exp_lo + 1
    if len(found) != expected:
        have = sorted(i for i, _ in found)
        raise SystemExit(
            f"ERROR: cell {bench}/{stress}/{method} under {root}: "
            f"expected exp_{exp_lo}..exp_{exp_hi} ({expected} files), "
            f"found {len(found)}: {have}"
        )
    return found


def plan_sweep_d1(dest_root: Path) -> Tuple[CopyPlan, Dict[Tuple[str, str, str], int]]:
    plan: CopyPlan = []
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for src_key, exp_lo, exp_hi, bench, stress, methods in SELECTION:
        root = _src_root(src_key)
        for method in methods:
            found = _collect_exp_cell(root, exp_lo, exp_hi, bench, stress, method)
            for idx, csv in found:
                dest = dest_root / f"exp_{idx}" / bench / stress / f"{method}_log.csv"
                plan.append((csv, dest))
                counts[(bench, stress, method)] += 1

    # 3. Legacy B2 light → unified exp layout
    for method in LEGACY_B2_LIGHT_METHODS:
        for seed in LEGACY_B2_LIGHT_SEEDS:
            dname = f"{method}_B2_LML_light_seed{seed}"
            csv = SRC_RESULTS / dname / f"{method}_log.csv"
            if not csv.is_file():
                raise SystemExit(f"ERROR: missing legacy source {csv}")
            exp_idx = seed // 100
            dest = dest_root / f"exp_{exp_idx}" / "B2" / "light" / f"{method}_log.csv"
            plan.append((csv, dest))
            counts[("B2", "light", method)] += 1

    if len(plan) != SWEEP_D1_EXPECTED_TOTAL:
        lines = ["bench\tstress\tmethod\tn"]
        for (b, s, m), n in sorted(counts.items()):
            lines.append(f"{b}\t{s}\t{m}\t{n}")
        raise SystemExit(
            f"ERROR: sweep_d1 plan has {len(plan)} files, expected "
            f"{SWEEP_D1_EXPECTED_TOTAL}.\nPer-cell counts:\n" + "\n".join(lines)
        )
    return plan, dict(counts)


def _dest_nonempty(dest: Path) -> bool:
    if not dest.exists():
        return False
    return any(dest.rglob("*"))


def execute_plan(
    plan: CopyPlan,
    dry_run: bool,
) -> List[Tuple[Path, Path, str, int, int]]:
    """Return manifest rows: dest, src, sha256, n_rows, n_cols (sha empty on dry_run)."""
    rows: List[Tuple[Path, Path, str, int, int]] = []
    for src, dest in plan:
        if dry_run:
            rows.append((dest, src, "", 0, 0))
            continue
        n_rows, n_cols = clean_header_copy(src, dest)
        digest = sha256_file(dest)
        rows.append((dest, src, digest, n_rows, n_cols))
    return rows


def write_manifest(
    path: Path,
    dest_root: Path,
    rows: Sequence[Tuple[Path, Path, str, int, int]],
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["dest_rel_path\tsrc_abs_path\tsha256_of_dest\tn_rows\tn_cols"]
    for dest, src, digest, n_rows, n_cols in rows:
        try:
            rel = dest.relative_to(dest_root)
        except ValueError:
            rel = dest
        if dry_run:
            lines.append(f"{rel}\t{src.resolve()}\t(dry_run)\t\t")
        else:
            lines.append(f"{rel}\t{src.resolve()}\t{digest}\t{n_rows}\t{n_cols}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seedcounts_block(
    title: str,
    dest_root: Path,
    plan: CopyPlan,
) -> List[str]:
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for _src, dest in plan:
        # Parse either exp_*/B*/stress/method_log.csv or legacy dirname
        parts = dest.relative_to(dest_root).parts
        if len(parts) >= 4 and parts[0].startswith("exp_"):
            bench, stress, fname = parts[1], parts[2], parts[3]
            method = fname[: -len("_log.csv")] if fname.endswith("_log.csv") else fname
            counts[(bench, stress, method)] += 1
        elif len(parts) >= 2:
            m = LEGACY_DIR_RE.match(parts[0])
            if m:
                method, wl, stress, _seed = m.group(1), m.group(2), m.group(3), m.group(4)
                bench = {"B2_LML": "B2", "B3_IQI": "B3", "B4_DEM": "B4"}[wl]
                counts[(bench, stress, method)] += 1
    lines = [f"## {title}", "", "| benchmark | stress | method | n |", "|---|---|---|---:|"]
    for (b, s, m), n in sorted(counts.items()):
        lines.append(f"| {b} | {s} | {m} | {n} |")
    lines.append("")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=str, default="data", help="Package root (default: data)")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print full copy plan and per-tree counts; write nothing",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Required to overwrite an existing non-empty dest tree",
    )
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = (REPO / dest).resolve()
    else:
        dest = dest.resolve()

    tree_a = dest / "raw" / "behavioral" / "sweep_a1a2b1"
    tree_b = dest / "raw" / "behavioral" / "legacy_medium"
    tree_c = dest / "raw" / "behavioral" / "sweep_d1"
    man_dir = dest / "manifests"

    plan_a = plan_sweep_a1a2b1(tree_a)
    plan_b = plan_legacy_medium(tree_b)
    plan_c, d1_counts = plan_sweep_d1(tree_c)

    print("# ae_curate_data copy plan")
    print(f"- dest: {dest}")
    print(f"- dry_run: {args.dry_run}")
    print()
    print(f"## A. sweep_a1a2b1  → {tree_a.relative_to(dest)}")
    print(f"   files: {len(plan_a)}")
    print(f"## B. legacy_medium → {tree_b.relative_to(dest)}")
    print(f"   files: {len(plan_b)}")
    print(f"## C. sweep_d1      → {tree_c.relative_to(dest)}")
    print(f"   files: {len(plan_c)} (assert == {SWEEP_D1_EXPECTED_TOTAL})")
    print()
    print("### sweep_d1 per-cell counts")
    print(f"{'bench':<4}  {'stress':<8}  {'method':<22}  {'n':>3}")
    for (b, s, m), n in sorted(d1_counts.items()):
        print(f"{b:<4}  {s:<8}  {m:<22}  {n:3d}")
    print(f"{'TOTAL':<4}  {'':<8}  {'':<22}  {len(plan_c):3d}")
    print()

    if args.dry_run:
        print("DRY RUN — no files written.")
        return 0

    if _dest_nonempty(dest) and not args.force:
        raise SystemExit(
            f"ERROR: dest {dest} is non-empty; pass --force to overwrite "
            f"(or choose a fresh --dest)."
        )
    if args.force and dest.exists():
        # Remove only behavioral + manifests under dest to avoid nuking unrelated files
        for p in (tree_a, tree_b, tree_c, man_dir):
            if p.exists():
                shutil.rmtree(p)

    rows_a = execute_plan(plan_a, dry_run=False)
    rows_b = execute_plan(plan_b, dry_run=False)
    rows_c = execute_plan(plan_c, dry_run=False)

    write_manifest(man_dir / "MANIFEST_sweep_a1a2b1.tsv", tree_a, rows_a, dry_run=False)
    write_manifest(man_dir / "MANIFEST_legacy_medium.tsv", tree_b, rows_b, dry_run=False)
    write_manifest(man_dir / "MANIFEST_sweep_d1.tsv", tree_c, rows_c, dry_run=False)

    seed_lines = [
        "# SEEDCOUNTS — curated behavioral trees",
        "",
        f"Generated by `scripts/ae_curate_data.py` into `{dest}`.",
        "",
    ]
    seed_lines += seedcounts_block("sweep_a1a2b1", tree_a, plan_a)
    seed_lines += seedcounts_block("legacy_medium", tree_b, plan_b)
    seed_lines += seedcounts_block("sweep_d1", tree_c, plan_c)
    (man_dir / "SEEDCOUNTS.md").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")

    def tree_bytes(rows: Sequence[Tuple[Path, Path, str, int, int]]) -> int:
        return sum(r[0].stat().st_size for r in rows)

    ba, bb, bc = tree_bytes(rows_a), tree_bytes(rows_b), tree_bytes(rows_c)
    print("## Final summary")
    print(f"  sweep_a1a2b1: {len(rows_a)} files, {ba} bytes")
    print(f"  legacy_medium: {len(rows_b)} files, {bb} bytes")
    print(f"  sweep_d1: {len(rows_c)} files, {bc} bytes")
    print(f"  TOTAL: {len(rows_a)+len(rows_b)+len(rows_c)} files, {ba+bb+bc} bytes")
    print()
    du = subprocess.run(
        ["du", "-sh", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"du -sh {dest}: {du.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
