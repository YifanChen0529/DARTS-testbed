"""
Run a small (benchmark, stress_level, seed) evaluation matrix by invoking ``alg/coordinator.py``
once per tuple (one episode per coordinator run).

Requires four ``agent_node`` processes (or SSH-launched Jetsons) already able to reach the
coordinator gRPC endpoint.

Example:
    PYTHONPATH=. python alg/run_eval_matrix.py --exp_prefix rtss_matrix
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _default_matrix():
    return [
        ("B1_CDR", "S0", 1001),
        ("B2_LML", "S0", 1002),
        ("B3_IQI", "S0", 1003),
        ("B4_DEM", "S0", 1004),
    ]


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description="Run coordinator once per matrix tuple (RTSS-style sweep).")
    p.add_argument("--exp_prefix", type=str, default="rtss_matrix")
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to invoke coordinator.py.",
    )
    args = p.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = root if not env.get("PYTHONPATH") else root + os.pathsep + env["PYTHONPATH"]

    for bench, stress, seed in _default_matrix():
        exp_name = f"{args.exp_prefix}_{bench}_{stress}_seed{seed}"
        cmd = [
            args.python,
            os.path.join(root, "alg", "coordinator.py"),
            "--workload",
            bench,
            "--exp_name",
            exp_name,
            "--num_episodes",
            "1",
            "--random_seed",
            str(int(seed)),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=root, env=env)


if __name__ == "__main__":
    main()
