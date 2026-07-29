#!/usr/bin/env python3
"""
Sweep experiment driver: (method, benchmark, stress_level, seed) matrix.

Stress levels tune deadline / battery / horizon (not DAG topology). Benchmarks select
the workload registry ID passed as --workload to each training script.

Output layout (default):
    result/exp_1/
      master_run_log.csv
      B1/darts_light_log.csv
      B1/cea_edf_light_log.csv
      B1/sadcher_light_log.csv
      ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent

METHOD_SCRIPTS: Dict[str, List[str]] = {
    "darts": [sys.executable, str(ROOT / "alg" / "train_darts.py")],
    "darts_no_incentive": [sys.executable, str(ROOT / "alg" / "train_darts.py")],
    "cea_edf": [sys.executable, str(ROOT / "alg" / "baseline_cea_edf.py")],
    "sadcher": [sys.executable, str(ROOT / "alg" / "baseline_sadcher.py")],
}

BENCHMARKS = ["B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"]

# Deadline / battery / horizon tightness (orthogonal to benchmark DAG).
STRESS_LEVELS: Dict[str, Dict[str, Any]] = {
    "light": {
        "global_deadline_ms": 180000.0,
        "battery_capacities_wh": "0.60,0.90,1.20,1.50",
        "max_episode_steps": 250,
        "max_work_ms": 200000.0,
        "lr_eta": 3e-4,
        "lr_phi": 1e-4,
        "lambda_rho": 2.0,
        "gradient_clip_norm": 1.0,
        "lipschitz_rho_estimate": 10.0,
    },
    "medium": {
        "global_deadline_ms": 150000.0,
        "battery_capacities_wh": "0.50,0.75,1.00,1.25",
        "max_episode_steps": 200,
        "max_work_ms": 142500.0,
        "lr_eta": 3e-4,
        "lr_phi": 1e-4,
        "lambda_rho": 1.0,
        "gradient_clip_norm": 1.0,
        "lipschitz_rho_estimate": 10.0,
    },
    "heavy": {
        "global_deadline_ms": 120000.0,
        "battery_capacities_wh": "0.40,0.60,0.80,1.00",
        "max_episode_steps": 180,
        "max_work_ms": 120000.0,
        "lr_eta": 1e-4,
        "lr_phi": 5e-5,
        "lambda_rho": 1.0,
        "gradient_clip_norm": 1.0,
        "lipschitz_rho_estimate": 10.0,
    },
}

MASTER_LOG_HEADER = [
    "method",
    "benchmark",
    "stress_level",
    "seed",
    "exp_name",
    "start_time",
    "end_time",
    "duration_s",
    "exit_code",
    "rho_hat_final_100_mean",
    "rho_hat_final_100_std",
    "n_nondegradable_n1_total",
    "n_nondegradable_n2_total",
]


def _aggregate_last_100_episodes(log_csv: Path) -> Tuple[float, float, float, float]:
    """Mean/std of rho_hat and summed N1/N2 counts over the last 100 CSV rows (eval checkpoints)."""
    if not log_csv.is_file():
        return float("nan"), float("nan"), float("nan"), float("nan")
    with open(log_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tail = rows[-100:]
    rhos: List[float] = []
    n1_sum = 0.0
    n2_sum = 0.0
    for r in tail:
        try:
            rhos.append(float(r.get("rho_hat", "nan")))
        except ValueError:
            continue
        try:
            n1_sum += float(r.get("n_nondegradable_n1", 0) or 0)
            n2_sum += float(r.get("n_nondegradable_n2", 0) or 0)
        except ValueError:
            pass
    rho_mean = float(statistics.mean(rhos)) if rhos else float("nan")
    rho_std = float(statistics.stdev(rhos)) if len(rhos) > 1 else 0.0
    return rho_mean, rho_std, n1_sum, n2_sum


def _safe_round_int(x: float) -> int:
    if x != x or math.isinf(x):
        return 0
    return int(round(x))


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _benchmark_dir_name(benchmark: str) -> str:
    # B1_CDR -> B1, B2_LML -> B2, etc.
    return benchmark.split("_", 1)[0]


def _build_cli(
    method: str,
    benchmark: str,
    stress: str,
    seed: int,
    exp_tag: str,
    num_episodes: int,
) -> List[str]:
    p = STRESS_LEVELS[stress]
    cmd = list(METHOD_SCRIPTS[method])
    cmd += [
        "--exp_name",
        exp_tag,
        "--workload",
        benchmark,
        "--global_deadline_ms",
        str(p["global_deadline_ms"]),
        "--battery_capacities_wh",
        str(p["battery_capacities_wh"]),
        "--max_episode_steps",
        str(p["max_episode_steps"]),
        "--max_work_ms",
        str(p["max_work_ms"]),
        "--random_seed",
        str(seed),
        "--lr_eta",
        str(p["lr_eta"]),
        "--lr_phi",
        str(p["lr_phi"]),
        "--lambda_rho",
        str(p["lambda_rho"]),
        "--gradient_clip_norm",
        str(p["gradient_clip_norm"]),
        "--lipschitz_rho_estimate",
        str(p["lipschitz_rho_estimate"]),
    ]
    if method == "darts_no_incentive":
        cmd.append("--no_incentive")
    if method in ("darts", "darts_no_incentive"):
        cmd.extend(["--bilevel_mode", "fixed_eta"])
    if method in ("darts", "darts_no_incentive") and num_episodes < 500:
        cmd.extend(["--save_interval", "10", "--eval_episodes", "5"])
    if method in ("darts", "darts_no_incentive", "cea_edf", "sadcher"):
        cmd += ["--num_episodes", str(num_episodes)]
    return cmd


def _find_log_csv(log_dir: Path) -> Optional[Path]:
    for name in ("darts_log.csv", "cea_edf_log.csv", "sadcher_log.csv"):
        p = log_dir / name
        if p.is_file():
            return p
    return None


def _find_config_json(log_dir: Path) -> Optional[Path]:
    for name in ("config.json", "cea_edf_config.json", "sadcher_config.json"):
        p = log_dir / name
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run (method × benchmark × stress × seed) experiment matrix.")
    parser.add_argument(
        "--methods",
        type=str,
        default="darts,darts_no_incentive,cea_edf,sadcher",
        help="Comma-separated subset of: darts, darts_no_incentive, cea_edf, sadcher",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS),
        help=f"Comma-separated subset of: {','.join(BENCHMARKS)}",
    )
    parser.add_argument(
        "--stress-levels",
        type=str,
        default="light,medium,heavy",
        help="Comma-separated subset of stress keys in STRESS_LEVELS.",
    )
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Project-relative results root (matches DARTS_CONFIG.results_dir).",
    )
    parser.add_argument(
        "--sweep_out",
        type=str,
        default="result/exp_1",
        help=(
            "Root directory to store sweep artifacts in benchmark subfolders. "
            "Example: result/exp_1/B1/darts_light_log.csv"
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5000,
        help="Episodes per run (DARTS train loop and baseline scripts).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    methods = _parse_csv_list(args.methods)
    benchmarks = [b for b in _parse_csv_list(args.benchmarks) if b in BENCHMARKS]
    stress_levels = [s for s in _parse_csv_list(args.stress_levels) if s in STRESS_LEVELS]
    seeds = [int(x) for x in _parse_csv_list(args.seeds)]
    include_seed_in_name = len(seeds) > 1

    for m in methods:
        if m not in METHOD_SCRIPTS:
            raise SystemExit(f"Unknown method {m!r}; choose from {list(METHOD_SCRIPTS)}")

    sweep_root = ROOT / args.sweep_out.replace("/", os.sep)
    sweep_root.mkdir(parents=True, exist_ok=True)
    master_path = sweep_root / "master_run_log.csv"
    master_exists = master_path.is_file()
    with open(master_path, "a", newline="") as mf:
        mw = csv.writer(mf)
        if not master_exists:
            mw.writerow(MASTER_LOG_HEADER)

        for method in methods:
            for benchmark in benchmarks:
                for stress in stress_levels:
                    for seed in seeds:
                        exp_tag = f"{method}_{benchmark}_{stress}_seed{seed}".replace(" ", "_")
                        cmd = _build_cli(method, benchmark, stress, seed, exp_tag, args.num_episodes)
                        log_dir = ROOT / args.results_dir / exp_tag

                        if args.dry_run:
                            print("DRY:", " ".join(cmd))
                            continue

                        env = os.environ.copy()
                        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                        t_start = time.time()
                        start_iso = datetime.now(timezone.utc).isoformat()
                        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
                        end_iso = datetime.now(timezone.utc).isoformat()
                        duration_s = time.time() - t_start

                        log_src = _find_log_csv(log_dir) if log_dir.is_dir() else None
                        cfg_src = _find_config_json(log_dir) if log_dir.is_dir() else None

                        bench_dir = _benchmark_dir_name(benchmark)
                        target_dir = sweep_root / bench_dir
                        target_dir.mkdir(parents=True, exist_ok=True)
                        seed_suffix = f"_seed{seed}" if include_seed_in_name else ""
                        log_name = f"{method}_{stress}{seed_suffix}_log.csv"
                        cfg_name = f"{method}_{stress}{seed_suffix}_config.json"
                        log_dst = target_dir / log_name
                        cfg_dst = target_dir / cfg_name

                        if log_src is not None:
                            try:
                                shutil.copy2(log_src, log_dst)
                            except shutil.SameFileError:
                                pass  # src and dst resolve to same inode (v8 leftover link)
                        else:
                            # Keep deterministic output structure even when a short debug run
                            # finishes before scripts emit their first eval CSV row.
                            with open(log_dst, "w", newline="") as f:
                                f.write("")
                        if cfg_src is not None:
                            try:
                                shutil.copy2(cfg_src, cfg_dst)
                            except shutil.SameFileError:
                                pass  # src and dst resolve to same inode (v8 leftover link)
                        else:
                            with open(cfg_dst, "w") as f:
                                f.write("{}\n")

                        agg_path = log_dst if log_src is not None else ROOT / "__missing_log__.csv"
                        rho_m, rho_s, n1_tot, n2_tot = _aggregate_last_100_episodes(agg_path)
                        mw.writerow(
                            [
                                method,
                                benchmark,
                                stress,
                                seed,
                                exp_tag,
                                start_iso,
                                end_iso,
                                f"{duration_s:.3f}",
                                rc,
                                rho_m if rho_m == rho_m else "N/A",
                                rho_s if rho_s == rho_s else "N/A",
                                _safe_round_int(n1_tot),
                                _safe_round_int(n2_tot),
                            ]
                        )


if __name__ == "__main__":
    main()
