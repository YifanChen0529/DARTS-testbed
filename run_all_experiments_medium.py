#!/usr/bin/env python3
"""Run medium-stress experiment matrix with benchmark-specific stress axes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from alg.stress_profiles import get_stress_profile

ROOT = Path(__file__).resolve().parent
STRESS_LEVEL = "medium"
BENCHMARKS = ["B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"]
SEEDS_PER_BENCHMARK = {"B1_CDR": 10, "B2_LML": 10, "B3_IQI": 5, "B4_DEM": 5}
METHODS = {
    "darts": ROOT / "alg" / "train_darts.py",
    "darts_nocert": ROOT / "alg" / "train_darts.py",
    "darts_no_incentive": ROOT / "alg" / "train_darts.py",
    "cea_edf": ROOT / "alg" / "baseline_cea_edf.py",
    "sadcher": ROOT / "alg" / "baseline_sadcher.py",
}

# darts_nocert is the cert-masking ablation; admitted only on these benchmarks
# (B1 + B2) so bidding-time overhead can be measured on B2_LML. B3/B4 are
# excluded (no multi-stress sweep — degenerate ablation there).
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
_AGGREGATE_LOG_NAMES = {
    "darts": "darts_log.csv",
    "darts_nocert": "darts_nocert_log.csv",
    "darts_no_incentive": "darts_no_incentive_log.csv",
    "cea_edf": "cea_edf_log.csv",
    "sadcher": "sadcher_log.csv",
}
_WORKSPACE_CONFIG_NAMES = {
    "darts": "config.json",
    "darts_nocert": "config.json",
    "darts_no_incentive": "config.json",
    "cea_edf": "cea_edf_config.json",
    "sadcher": "sadcher_config.json",
}


def _seed_list(n: int) -> list[int]:
    return [100 * (i + 1) for i in range(n)]


def _fmt_value(v: object) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def _build_cmd(method: str, benchmark: str, seed: int, profile: dict, num_episodes: int) -> list[str]:
    cmd = [
        sys.executable,
        str(METHODS[method]),
        "--workload",
        benchmark,
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
    exp_name = f"{method}_{benchmark}_{STRESS_LEVEL}_seed{seed}"
    cmd.extend(["--exp_name", exp_name])
    return cmd


def _find_method_log(exp_dir: Path, method: str) -> Path:
    return exp_dir / _WORKSPACE_LOG_NAMES[method]


def _find_method_config(exp_dir: Path, method: str) -> Path:
    return exp_dir / _WORKSPACE_CONFIG_NAMES[method]


def _benchmark_dir_name(benchmark: str) -> str:
    return benchmark.split("_", 1)[0]


def _result_log_dst(results_dir: Path, run_idx: int, benchmark: str, method: str) -> Path:
    bench_dir = _benchmark_dir_name(benchmark)
    target_dir = results_dir / f"exp_{run_idx}" / bench_dir / STRESS_LEVEL
    return target_dir / _AGGREGATE_LOG_NAMES[method]


def _expected_total_runs(method_list: list[str]) -> int:
    total = 0
    for method in method_list:
        gated = _METHOD_BENCHMARK_GATING.get(method)
        for benchmark in BENCHMARKS:
            if gated is not None and benchmark not in gated:
                continue
            if get_stress_profile(benchmark, STRESS_LEVEL) is None:
                continue
            total += SEEDS_PER_BENCHMARK[benchmark]
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark-specific medium stress matrix.")
    parser.add_argument("--methods", default="darts,darts_nocert,darts_no_incentive,cea_edf,sadcher")
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated subset of BENCHMARKS to iterate (default: all).",
    )
    args = parser.parse_args()

    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.benchmarks:
        bench_loop = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        for b in bench_loop:
            if b not in BENCHMARKS:
                raise SystemExit(f"Unknown benchmark in --benchmarks: {b!r}")
    else:
        bench_loop = list(BENCHMARKS)

    expected = _expected_total_runs(method_list)
    print(f"[info] expected total runs for stress={STRESS_LEVEL}: {expected}")

    emitted = 0
    for benchmark in bench_loop:
        profile = get_stress_profile(benchmark, STRESS_LEVEL)
        if profile is None:
            print(f"[skip] {benchmark} not evaluated at stress={STRESS_LEVEL} (single-medium-only by design)")
            continue
        for run_idx, seed in enumerate(_seed_list(SEEDS_PER_BENCHMARK[benchmark]), start=1):
            for method in method_list:
                if method not in METHODS:
                    raise SystemExit(f"Unknown method: {method}")
                allowed_benchmarks = _METHOD_BENCHMARK_GATING.get(method)
                if allowed_benchmarks is not None and benchmark not in allowed_benchmarks:
                    print(
                        f"[GATE-SKIP] benchmark={_benchmark_dir_name(benchmark)} "
                        f"stress={STRESS_LEVEL} method={method} seed={seed} "
                        f"(method restricted to {sorted(allowed_benchmarks)})"
                    )
                    continue
                log_dst = _result_log_dst(ROOT / args.results_dir, run_idx, benchmark, method)
                if args.resume and log_dst.is_file():
                    print(
                        f"[RESUME-SKIP] benchmark={_benchmark_dir_name(benchmark)} "
                        f"stress={STRESS_LEVEL} method={method} seed={seed}"
                    )
                    continue
                emitted += 1
                cmd = _build_cmd(method, benchmark, seed, profile, args.num_episodes)
                print("CLI:", " ".join(cmd))
                if args.dry_run:
                    continue
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
                if rc != 0:
                    raise SystemExit(rc)
                exp_name = f"{method}_{benchmark}_{STRESS_LEVEL}_seed{seed}"
                exp_dir = ROOT / "results" / exp_name
                log_src = _find_method_log(exp_dir, method)
                cfg_src = _find_method_config(exp_dir, method)
                bench_dir = _benchmark_dir_name(benchmark)
                target_dir = ROOT / args.results_dir / f"exp_{run_idx}" / bench_dir / STRESS_LEVEL
                target_dir.mkdir(parents=True, exist_ok=True)
                log_dst = _result_log_dst(ROOT / args.results_dir, run_idx, benchmark, method)
                cfg_dst = target_dir / f"{method}_config.json"
                if log_src.is_file():
                    try:
                        shutil.copy2(log_src, log_dst)
                    except shutil.SameFileError:
                        pass  # src and dst resolve to same inode (v8 leftover link)
                if cfg_src.is_file():
                    try:
                        shutil.copy2(cfg_src, cfg_dst)
                    except shutil.SameFileError:
                        pass  # src and dst resolve to same inode (v8 leftover link)
    print(f"[info] emitted commands: {emitted}")


if __name__ == "__main__":
    main()
