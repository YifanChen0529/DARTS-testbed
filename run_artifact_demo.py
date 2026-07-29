#!/usr/bin/env python3
"""Artifact-evaluation demo: reduced-scale end-to-end run in simulator mode.

Simulator mode (config_darts.py default deployment_mode=\"simulator\") is the same
mode that produced ALL behavioral results in the paper: all four agents in one
Python process with synthetic message latencies. No robot, Jetson, or network
access is required.

This script is an orchestration wrapper around the live
run_all_experiments_{light,medium,heavy}.py code paths (per-method entry scripts
under alg/ + stress-profile CLI overrides). It demonstrates trends at reduced
scale; the paper's exact numbers regenerate from the shipped raw logs via the
Makefile.

Every flag is independently variable (AE \"extensible\" criterion):
  --benchmark     which of B1–B4 (CDR / LML / IQI / DEM) to run
  --stress        stress profile (light / medium / heavy; see valid matrix)
  --method        algorithm / baseline key (darts family, cea_edf, sadcher)
  --episodes      training / run length
  --seed          RNG seed passed to the entry script
  --eval_interval CSV row cadence (must divide episodes)
  --eval_batch    evaluation episodes averaged into each CSV row
  --results_dir   directory under which <exp_name>/ workspace logs are found
  --exp_name      workspace folder name under results_dir
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from alg.stress_profiles import get_stress_profile

ROOT = Path(__file__).resolve().parent

BENCHMARK_MAP = {
    "B1": "B1_CDR",
    "B2": "B2_LML",
    "B3": "B3_IQI",
    "B4": "B4_DEM",
}

# Evaluator-facing valid matrix (paper behavioral cells).
VALID_STRESS_MATRIX = {
    "B1_CDR": ["light", "medium", "heavy"],
    "B2_LML": ["light", "medium", "heavy"],
    "B3_IQI": ["medium"],
    "B4_DEM": ["medium"],
}

# Mirrored from run_all_experiments_medium.py (keep in sync).
METHODS = {
    "darts": ROOT / "alg" / "train_darts.py",
    "darts_nocert": ROOT / "alg" / "train_darts.py",
    "darts_no_incentive": ROOT / "alg" / "train_darts.py",
    "cea_edf": ROOT / "alg" / "baseline_cea_edf.py",
    "sadcher": ROOT / "alg" / "baseline_sadcher.py",
}

NOCERT_BENCHMARKS = ["B1_CDR", "B2_LML"]
_METHOD_BENCHMARK_GATING: dict[str, set[str]] = {
    "darts_nocert": set(NOCERT_BENCHMARKS),
}

_WORKSPACE_LOG_NAMES = {
    "darts": "darts_log.csv",
    "darts_nocert": "darts_log.csv",
    "darts_no_incentive": "darts_log.csv",
    "cea_edf": "cea_edf_log.csv",
    "sadcher": "sadcher_log.csv",
}

DARTS_FAMILY = {m for m, p in METHODS.items() if p.name == "train_darts.py"}


def _fmt_value(v: object) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def _print_valid_matrix() -> None:
    print("Valid (benchmark, stress) combinations for this demo:")
    print("  B1 (B1_CDR): light, medium, heavy")
    print("  B2 (B2_LML): light, medium, heavy")
    print("  B3 (B3_IQI): medium only")
    print("  B4 (B4_DEM): medium only")


def _build_cmd(
    method: str,
    full_id: str,
    seed: int,
    profile: dict,
    num_episodes: int,
    exp_name: str,
) -> list[str]:
    """Replicate run_all_experiments_medium.py::_build_cmd (must stay in sync).

    Differences vs the live runner: STRESS_LEVEL is not baked into exp_name here
    (caller supplies --exp_name); stress is already encoded in ``profile``.
    """
    # Cross-ref: run_all_experiments_medium.py::_build_cmd — keep method-specific
    # extras and stress-profile formatting identical to the live runners.
    cmd = [
        sys.executable,
        str(METHODS[method]),
        "--workload",
        full_id,
        "--random_seed",
        str(seed),
        "--num_episodes",
        str(num_episodes),
    ]
    if method == "darts_nocert":
        cmd.append("--disable_feasibility_mask")
    if method in ("darts", "darts_nocert", "darts_no_incentive"):
        cmd.extend(["--bilevel_mode", "fixed_eta"])
    if method == "darts_no_incentive":
        cmd.append("--no_incentive")
    if method in ("darts", "darts_nocert", "darts_no_incentive") and num_episodes < 500:
        cmd.extend(["--save_interval", "10", "--eval_episodes", "5"])
    for key, value in profile.items():
        cmd.extend([f"--{key}", _fmt_value(value)])
    cmd.extend(["--exp_name", exp_name])
    return cmd


def _append_eval_cadence(cmd: list[str], method: str, eval_interval: int, eval_batch: int) -> None:
    """Wrapper-only additions on top of the live _build_cmd."""
    if method in DARTS_FAMILY:
        cmd.extend(
            [
                "--save_interval",
                str(eval_interval),
                "--eval_episodes",
                str(eval_batch),
                "--deployment_mode",
                "simulator",
            ]
        )
    elif method == "cea_edf":
        cmd.extend(
            [
                "--eval_interval",
                str(eval_interval),
                "--eval_batch",
                str(eval_batch),
            ]
        )
    elif method == "sadcher":
        # --eval_every is PART B (optional); without it, baseline_sadcher still
        # writes a final-episode row via `or episode == num_run`.
        cmd.extend(["--eval_batch", str(eval_batch), "--eval_every", str(eval_interval)])


def _fmt_metric(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{float(v):.6g}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AE demo: reduced-scale simulator-mode experiment (paper behavioral mode)."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARK_MAP.keys()),
        help="Short benchmark id: B1 B2 B3 B4",
    )
    parser.add_argument(
        "--stress",
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Stress profile level",
    )
    parser.add_argument(
        "--method",
        default="darts",
        choices=list(METHODS.keys()),
        help="Method key (exact live METHODS spelling)",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="CSV row cadence (episodes %% eval_interval must be 0)",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=5,
        help="Evaluation episodes averaged per CSV row",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Workspace name under results_dir (default: demo_<method>_<B#>_<stress>_seed<seed>)",
    )
    args = parser.parse_args()

    full_id = BENCHMARK_MAP[args.benchmark]
    method = args.method
    stress = args.stress
    seed = args.seed
    episodes = args.episodes
    eval_interval = args.eval_interval
    eval_batch = args.eval_batch
    results_dir = args.results_dir
    exp_name = args.exp_name or f"demo_{method}_{args.benchmark}_{stress}_seed{seed}"

    try:
        profile = get_stress_profile(full_id, stress)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        _print_valid_matrix()
        sys.exit(2)

    if profile is None:
        print(
            f"ERROR: unsupported combination benchmark={args.benchmark} ({full_id}) "
            f"stress={stress}.",
            file=sys.stderr,
        )
        _print_valid_matrix()
        sys.exit(2)

    allowed = _METHOD_BENCHMARK_GATING.get(method)
    if allowed is not None and full_id not in allowed:
        allowed_short = sorted({b.split("_", 1)[0] for b in allowed})
        print(
            f"ERROR: method={method} is gated out for benchmark={args.benchmark} ({full_id}). "
            f"Allowed benchmarks: {allowed_short} ({sorted(allowed)}).",
            file=sys.stderr,
        )
        sys.exit(2)

    if episodes % eval_interval != 0:
        print(
            f"ERROR: --episodes ({episodes}) must be divisible by --eval_interval "
            f"({eval_interval}); got remainder {episodes % eval_interval}.",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = _build_cmd(method, full_id, seed, profile, episodes, exp_name)
    _append_eval_cadence(cmd, method, eval_interval, eval_batch)

    print("CLI:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    wall_s = time.perf_counter() - t0
    if rc != 0:
        sys.exit(rc)

    csv_path = (ROOT / results_dir / exp_name / _WORKSPACE_LOG_NAMES[method]).resolve()
    if not csv_path.is_file():
        print(f"ERROR: workspace CSV missing: {csv_path}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.replace("N/A", float("nan"))
    if len(df) == 0:
        print(f"ERROR: workspace CSV has zero data rows: {csv_path}", file=sys.stderr)
        sys.exit(3)

    for col in (
        "episode",
        "task_miss_rate",
        "rt_bound_violations",
        "max_rt_ms",
        "A1_crash_rate",
        "A2_crash_rate",
        "A3_crash_rate",
        "A4_crash_rate",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    last = df.iloc[-1]
    n = len(df)
    k = min(3, n)
    miss_last = last["task_miss_rate"] if "task_miss_rate" in df.columns else float("nan")
    miss_mean = (
        df["task_miss_rate"].iloc[-k:].mean() if "task_miss_rate" in df.columns else float("nan")
    )
    rt_viol = last["rt_bound_violations"] if "rt_bound_violations" in df.columns else float("nan")
    max_rt = last["max_rt_ms"] if "max_rt_ms" in df.columns else float("nan")
    crashes = [
        last[c] if c in df.columns else float("nan")
        for c in ("A1_crash_rate", "A2_crash_rate", "A3_crash_rate", "A4_crash_rate")
    ]
    final_ep = int(last["episode"]) if "episode" in df.columns and pd.notna(last["episode"]) else -1

    rt_ok = pd.notna(rt_viol) and float(rt_viol) == 0.0
    rt_tag = "[OK]" if rt_ok else "[WARN]"
    print()
    print(
        f"DEMO SUMMARY  {method} {args.benchmark} ({full_id}) {stress} "
        f"seed={seed} episodes={episodes}"
    )
    print(f"csv: {csv_path}   rows: {n}   final_episode: {final_ep}")
    print(
        f"task_miss_rate(last): {_fmt_metric(miss_last)}   "
        f"task_miss_rate(mean, last {k} rows): {_fmt_metric(miss_mean)}"
    )
    print(
        f"rt_bound_violations(last): {_fmt_metric(rt_viol)}  {rt_tag}  "
        "(Theorem-1-backed methods / darts family expected 0; "
        "baselines may legitimately be nonzero — WARN does not change exit code)"
    )
    print(
        f"max_rt_ms(last): {_fmt_metric(max_rt)}   "
        f"crash A1..A4 (last): {_fmt_metric(crashes[0])}, {_fmt_metric(crashes[1])}, "
        f"{_fmt_metric(crashes[2])}, {_fmt_metric(crashes[3])}"
    )
    print(f"wall_time_s: {wall_s:.1f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
