#!/usr/bin/env python3
"""TB-3 P6 post-processing: partition identity, adjudication, over-rejection, ceiling.

Counter semantics (eval-only aggregated row in ``alg/train_darts.py``
``evaluate_agents``):

* ``task_miss_rate`` (:1321) is a **per-episode mean** of a binary end-to-end
  miss (``task_time`` is None or ``> global_deadline_ms``). Range [0, 1].
* The four miss counters (:1369-1376) are **per-episode means of job counts**,
  not run totals. Per-episode construction is at :904-922 (4 jobs / episode on
  B1). FFL uses the per-job ``absolute_deadline_ms``, not the mission deadline.
* ``rt_bound_violations`` (:1274, :1344) is a **run total** (sum across
  episodes).

Pre-registered partition identity (per run, this convention):

    residual := task_miss_rate * 4 - (FFL + N1 + N2 + N3)
    tolerance 1e-9

Any residual > tol is flagged (prompt: the "completed after the mission
deadline but within its per-job deadline" channel / T0 H6). The identity is
implemented exactly; whatever the 180 rows produce is the result.

Violation rule (per accepted instance, per test, per (cell, s-point)):

    timing_violation := accept_e2e AND (any of its 3 runs has FFL > 0
        OR rt_bound_violations > 0 OR partition_residual > tol)
    energy_scope_miss := accept_e2e AND NOT feasible_e2e_all AND NOT
        timing_violation AND miss mass exclusively N1+N2+N3

Bootstrap: resample the 10 instance ids with replacement, B=10000,
``np.random.default_rng(20260825)`` (fresh generator per (cell, s-point[,
test]) so loop order cannot scramble CIs). Over-rejection is undefined on
resamples with 0 feasible instances: those draws are dropped from that CI
and counted in ``n_undefined``.

TB3-6b amendment (original rule and ``tb3_soundness_adjudication.csv`` kept):
R1 locks episode-binary semantics: ``|residual - 3 * task_miss_rate| < 1e-9``
on every run (one missed job per miss episode). R2 re-verdicts accepted
instances: TIMING_VIOLATION only on FFL, rt_bound_violations, or R1 failure;
otherwise accept-and-not-all-clean is energy_scope_miss. See
``tb3_soundness_adjudication_v2.csv``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.csv_schema_tb import (  # noqa: E402
    TB3_ADJUDICATION_HEADER,
    TB3_ADJUDICATION_P7_HEADER,
    TB3_ADJUDICATION_V2_HEADER,
    TB3_CEILING_HEADER,
    TB3_CEILING_P7_HEADER,
    TB3_FRONTIER_LB_P7_HEADER,
    TB3_OVERREJ_HEADER,
    TB3_OVERREJ_P7_HEADER,
)
from scripts.sweep_tb3 import TB3_EMP_PLAN  # noqa: E402

N_JOBS_PER_EPISODE = 4
N_EVAL_EPISODES = 5
PARTITION_TOL = 1e-9
R1_TOL = 1e-9
EXPECTED_ADJ_V1_SHA256 = (
    "4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9"
)
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 10000
N_INSTANCES = 10
INSTANCE_SEEDS = tuple(range(N_INSTANCES))
TESTS = ("S0", "S1")
ACCEPT_COL = {"S0": "s0_accept_e2e", "S1": "s1_accept_e2e"}
EXPECTED_GRID_SHA256 = (
    "0db098e9563a72f6961e380b40448a66d15bb1ba312e5088919606b116d6c42e"
)
DEFAULT_GRID = (
    Path(_PROJECT_ROOT) / "results_tb3_offline" / "grid" / "tb3_offline_grid.csv"
)

# Prompt 2.2 point estimates (n_rej_feas / n_feasible_e2e_all).
EXPECTED_OVER_REJECTION: Dict[Tuple[str, str, float], Tuple[int, int]] = {
    ("B1_CDR:medium", "S0", 1.0): (3, 3),
    ("B1_CDR:medium", "S0", 1.4): (1, 3),
    ("B1_CDR:medium", "S0", 1.6): (0, 3),
    ("B1_CDR:medium", "S1", 1.0): (3, 3),
    ("B1_CDR:medium", "S1", 1.4): (2, 3),
    ("B1_CDR:medium", "S1", 1.6): (0, 3),
    ("B1_CDR:heavy", "S0", 1.0): (1, 1),
    ("B1_CDR:heavy", "S0", 1.8): (1, 3),
    ("B1_CDR:heavy", "S0", 2.0): (0, 3),
    ("B1_CDR:heavy", "S1", 1.0): (1, 1),
    ("B1_CDR:heavy", "S1", 1.8): (2, 3),
    ("B1_CDR:heavy", "S1", 2.0): (0, 3),
}


class AnalysisError(RuntimeError):
    """Hard failure: schema, SHA, or join mismatch."""


def fmt_float(x: float) -> str:
    return format(float(x), ".17g")


def fmt_beta(x: float) -> str:
    """Plan betas are one-decimal; avoid 1.6000000000000001 in committed CSVs."""
    return f"{float(x):.1f}"


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    p = min(1.0, max(0.0, float(p)))
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    logp = math.log(p)
    log1p = math.log(1.0 - p)
    acc = 0.0
    for i in range(k + 1):
        logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        acc += math.exp(logc + i * logp + (n - i) * log1p)
    return min(1.0, max(0.0, acc))


def _bisect_root(fn, lo: float, hi: float, *, n_iter: int = 80) -> float:
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if fn(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _clopper_pearson_bisect(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Invert binomial CDF so the interval matches the beta.ppf definition."""
    if k == 0:
        lo = 0.0
    else:
        target = alpha / 2.0

        def f_lo(p: float) -> float:
            return (1.0 - _binom_cdf(k - 1, n, p)) - target

        lo = _bisect_root(f_lo, 0.0, 1.0)
    if k == n:
        hi = 1.0
    else:
        target = alpha / 2.0

        def f_hi(p: float) -> float:
            # Binomial CDF decreases in p; negate so the bisect sees an
            # increasing residual (root: P(X <= k) = alpha/2).
            return target - _binom_cdf(k, n, p)

        hi = _bisect_root(f_hi, 0.0, 1.0)
    return float(lo), float(hi)


def clopper_pearson(
    k: int, n: int, alpha: float = 0.05, *, numpy_only: bool = False
) -> Tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for k successes in n trials."""
    if n <= 0:
        raise ValueError("n must be positive")
    k = int(k)
    n = int(n)
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside 0..{n}")
    if not numpy_only:
        try:
            from scipy.stats import beta as beta_dist

            lo = (
                0.0
                if k == 0
                else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
            )
            hi = (
                1.0
                if k == n
                else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
            )
            return lo, hi
        except ImportError:
            pass
    return _clopper_pearson_bisect(k, n, alpha)


def _as_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, float) and math.isnan(v):
        return float("nan")
    if isinstance(v, str):
        s = v.strip()
        if not s or s.upper() in ("N/A", "NAN", "NONE"):
            return float("nan")
        return float(s)
    return float(v)


def _as_int(v: Any) -> int:
    return int(round(_as_float(v)))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_grid_sha(path: Path, expected: str = EXPECTED_GRID_SHA256) -> str:
    digest = sha256_file(path)
    if digest != expected:
        raise AnalysisError(
            f"STOP: grid SHA-256 mismatch\n  path={path}\n  got={digest}\n"
            f"  expected={expected}"
        )
    return digest


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def partition_residual(
    task_miss_rate: Any,
    ffl: Any,
    n1: Any,
    n2: Any,
    n3: Any,
    n_jobs: int = N_JOBS_PER_EPISODE,
) -> float:
    """lhs - rhs for the pre-registered identity (per-episode-mean convention)."""
    return _as_float(task_miss_rate) * float(n_jobs) - (
        _as_float(ffl) + _as_float(n1) + _as_float(n2) + _as_float(n3)
    )


def residual_of_run(row: Dict[str, Any]) -> float:
    return partition_residual(
        row["task_miss_rate"],
        row["n_miss_full_fidelity_late"],
        row["n_miss_n1_drop"],
        row["n_miss_n2_fallback_insufficient"],
        row["n_miss_n3_unreached"],
    )


def residual_census(
    runs: Sequence[Dict[str, Any]], *, tol: float = PARTITION_TOL
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Per-run residual rows, max residual, count of residuals > tol."""
    out: List[Dict[str, Any]] = []
    n_nz = 0
    max_r = 0.0
    for row in runs:
        r = residual_of_run(row)
        rec = {
            "cell": row["cell"],
            "beta": _as_float(row["beta"]),
            "instance_seed": _as_int(row["instance_seed"]),
            "ckpt_seed": _as_int(row["ckpt_seed"]),
            "task_miss_rate": _as_float(row["task_miss_rate"]),
            "ffl": _as_float(row["n_miss_full_fidelity_late"]),
            "n1": _as_float(row["n_miss_n1_drop"]),
            "n2": _as_float(row["n_miss_n2_fallback_insufficient"]),
            "n3": _as_float(row["n_miss_n3_unreached"]),
            "rt_bound_violations": _as_float(row["rt_bound_violations"]),
            "partition_residual": r,
        }
        out.append(rec)
        if r > tol:
            n_nz += 1
        if abs(r) > abs(max_r):
            max_r = r
    return out, max_r, n_nz


def _runs_by_instance(
    runs: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, float, int], List[Dict[str, Any]]]:
    by: Dict[Tuple[str, float, int], List[Dict[str, Any]]] = {}
    for row in runs:
        key = (row["cell"], _as_float(row["beta"]), _as_int(row["instance_seed"]))
        by.setdefault(key, []).append(row)
    return by


def adjudicate_instance(
    accept_e2e: int,
    feasible_e2e_all: int,
    inst_runs: Sequence[Dict[str, Any]],
    *,
    tol: float = PARTITION_TOL,
) -> Dict[str, Any]:
    residuals = [residual_of_run(r) for r in inst_runs]
    ffl = [_as_float(r["n_miss_full_fidelity_late"]) for r in inst_runs]
    n1 = [_as_float(r["n_miss_n1_drop"]) for r in inst_runs]
    n2 = [_as_float(r["n_miss_n2_fallback_insufficient"]) for r in inst_runs]
    n3 = [_as_float(r["n_miss_n3_unreached"]) for r in inst_runs]
    rt = [_as_float(r["rt_bound_violations"]) for r in inst_runs]
    ffl_sum = float(sum(ffl))
    n1_sum = float(sum(n1))
    n2_sum = float(sum(n2))
    n3_sum = float(sum(n3))
    resid_max = max(residuals) if residuals else 0.0
    any_ffl = any(v > 0.0 for v in ffl)
    any_rt = any(v > 0.0 for v in rt)
    any_resid = any(v > tol for v in residuals)
    timing_violation = bool(accept_e2e) and (any_ffl or any_rt or any_resid)
    exclusive_n123 = (ffl_sum == 0.0) and ((n1_sum + n2_sum + n3_sum) > 0.0)
    energy_scope_miss = (
        bool(accept_e2e)
        and (not bool(feasible_e2e_all))
        and (not timing_violation)
        and exclusive_n123
    )
    if timing_violation:
        verdict = "TIMING_VIOLATION"
    elif energy_scope_miss:
        verdict = "energy_scope_miss"
    else:
        verdict = "consistent"
    return {
        "ffl_sum": ffl_sum,
        "n1_sum": n1_sum,
        "n2_sum": n2_sum,
        "n3_sum": n3_sum,
        "partition_residual_max": resid_max,
        "verdict": verdict,
    }


def build_adjudication(
    runs: Sequence[Dict[str, Any]],
    feas: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by = _runs_by_instance(runs)
    rows: List[Dict[str, Any]] = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            for test in TESTS:
                col = ACCEPT_COL[test]
                for seed in INSTANCE_SEEDS:
                    matches = [
                        f
                        for f in feas
                        if f["cell"] == cell
                        and abs(_as_float(f["beta"]) - float(beta)) < 1e-12
                        and _as_int(f["instance_seed"]) == seed
                    ]
                    if len(matches) != 1:
                        raise AnalysisError(
                            f"feas join {cell} beta={beta} seed={seed}: "
                            f"{len(matches)} rows"
                        )
                    f = matches[0]
                    inst_runs = by.get((cell, float(beta), seed), [])
                    if len(inst_runs) != 3:
                        raise AnalysisError(
                            f"expected 3 runs for {cell} beta={beta} seed={seed}, "
                            f"got {len(inst_runs)}"
                        )
                    acc = _as_int(f[col])
                    fea = _as_int(f["feasible_e2e_all"])
                    adj = adjudicate_instance(acc, fea, inst_runs)
                    rows.append(
                        {
                            "cell": cell,
                            "test": test,
                            "beta": fmt_beta(float(beta)),
                            "instance_seed": seed,
                            "accept_e2e": acc,
                            "feasible_e2e_all": fea,
                            "ffl_sum": fmt_float(adj["ffl_sum"]),
                            "n1_sum": fmt_float(adj["n1_sum"]),
                            "n2_sum": fmt_float(adj["n2_sum"]),
                            "n3_sum": fmt_float(adj["n3_sum"]),
                            "partition_residual_max": fmt_float(
                                adj["partition_residual_max"]
                            ),
                            "verdict": adj["verdict"],
                        }
                    )
    return rows


def _aligned_bits(
    feas: Sequence[Dict[str, Any]], cell: str, beta: float, flag: str
) -> np.ndarray:
    bits = np.zeros(N_INSTANCES, dtype=np.int64)
    for f in feas:
        if f["cell"] != cell or abs(_as_float(f["beta"]) - beta) >= 1e-12:
            continue
        bits[_as_int(f["instance_seed"])] = _as_int(f[flag])
    return bits


def bootstrap_over_rejection(
    feas_bits: np.ndarray,
    accept_bits: np.ndarray,
    rng: np.random.Generator,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Tuple[float, float, int]:
    """Percentile 95% CI of n_rej_feas / n_feasible; drop undefined draws."""
    hats: List[float] = []
    n_undefined = 0
    n = len(feas_bits)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        n_feas = int(feas_bits[idx].sum())
        if n_feas == 0:
            n_undefined += 1
            continue
        n_rej_feas = int(((1 - accept_bits[idx]) * feas_bits[idx]).sum())
        hats.append(n_rej_feas / n_feas)
    if not hats:
        return float("nan"), float("nan"), n_undefined
    lo, hi = np.percentile(np.asarray(hats, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi), n_undefined


def bootstrap_feasible_frac(
    feas_bits: np.ndarray,
    rng: np.random.Generator,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Tuple[float, float]:
    fracs = np.empty(n_bootstrap, dtype=np.float64)
    n = len(feas_bits)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        fracs[i] = float(feas_bits[idx].mean())
    lo, hi = np.percentile(fracs, [2.5, 97.5])
    return float(lo), float(hi)


def _seeds_at(
    feas: Sequence[Dict[str, Any]], cell: str, beta: float
) -> List[int]:
    return sorted(
        {
            _as_int(f["instance_seed"])
            for f in feas
            if f["cell"] == cell and abs(_as_float(f["beta"]) - beta) < 1e-12
        }
    )


def bits_for_flag(
    feas: Sequence[Dict[str, Any]],
    cell: str,
    beta: float,
    flag: str,
    seeds: Sequence[int],
) -> np.ndarray:
    idx = {int(s): i for i, s in enumerate(seeds)}
    bits = np.zeros(len(seeds), dtype=np.int64)
    for f in feas:
        if f["cell"] != cell or abs(_as_float(f["beta"]) - beta) >= 1e-12:
            continue
        s = _as_int(f["instance_seed"])
        if s in idx:
            bits[idx[s]] = _as_int(f[flag])
    return bits


def build_ceiling_p7(
    feas: Sequence[Dict[str, Any]], *, flag: str = "feasible_e2e_all"
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            seeds = _seeds_at(feas, cell, float(beta))
            bits = bits_for_flag(feas, cell, float(beta), flag, seeds)
            n = len(seeds)
            n_feasible = int(bits.sum())
            frac = n_feasible / float(n) if n else float("nan")
            cp_lo, cp_hi = clopper_pearson(n_feasible, n)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boot_lo, boot_hi = bootstrap_feasible_frac(bits, rng)
            rows.append(
                {
                    "cell": cell,
                    "beta": fmt_beta(float(beta)),
                    "n_instances": n,
                    "n_feasible": n_feasible,
                    "feasible_frac": fmt_float(frac),
                    "ci_lo": fmt_float(cp_lo),
                    "ci_hi": fmt_float(cp_hi),
                    "boot_lo": fmt_float(boot_lo),
                    "boot_hi": fmt_float(boot_hi),
                }
            )
    return rows


def build_overrejection_p7(feas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            seeds = _seeds_at(feas, cell, float(beta))
            feas_bits = bits_for_flag(
                feas, cell, float(beta), "feasible_e2e_all", seeds
            )
            n = len(seeds)
            n_feasible = int(feas_bits.sum())
            for test in TESTS:
                accept_bits = bits_for_flag(
                    feas, cell, float(beta), ACCEPT_COL[test], seeds
                )
                n_rej_feas = int(((1 - accept_bits) * feas_bits).sum())
                hat = (n_rej_feas / n_feasible) if n_feasible else float("nan")
                cp_lo, cp_hi = (
                    clopper_pearson(n_rej_feas, n_feasible)
                    if n_feasible
                    else (float("nan"), float("nan"))
                )
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                boot_lo, boot_hi, n_undef = bootstrap_over_rejection(
                    feas_bits, accept_bits, rng
                )
                rows.append(
                    {
                        "cell": cell,
                        "test": test,
                        "beta": fmt_beta(float(beta)),
                        "n_instances": n,
                        "n_feasible_e2e_all": n_feasible,
                        "n_rej_feas": n_rej_feas,
                        "over_rejection_hat": fmt_float(hat),
                        "ci_lo": fmt_float(cp_lo),
                        "ci_hi": fmt_float(cp_hi),
                        "boot_lo": fmt_float(boot_lo),
                        "boot_hi": fmt_float(boot_hi),
                        "n_undefined": n_undef,
                        "n_bootstrap": N_BOOTSTRAP,
                    }
                )
    return rows


def build_adjudication_p7(
    runs: Sequence[Dict[str, Any]],
    feas: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by = _runs_by_instance(runs)
    rows: List[Dict[str, Any]] = []
    ordered = sorted(
        feas,
        key=lambda f: (f["cell"], _as_float(f["beta"]), _as_int(f["instance_seed"])),
    )
    for f in ordered:
        cell = f["cell"]
        beta = _as_float(f["beta"])
        seed = _as_int(f["instance_seed"])
        inst_runs = by.get((cell, beta, seed), [])
        if len(inst_runs) != 3:
            raise AnalysisError(
                f"p7 expected 3 runs for {cell} beta={beta} seed={seed}, "
                f"got {len(inst_runs)}"
            )
        for test in TESTS:
            extra = adjudicate_instance_v2(
                _as_int(f[ACCEPT_COL[test]]),
                _as_int(f["feasible_e2e_all"]),
                inst_runs,
            )
            adj_v1 = adjudicate_instance(
                _as_int(f[ACCEPT_COL[test]]),
                _as_int(f["feasible_e2e_all"]),
                inst_runs,
            )
            rows.append(
                {
                    "cell": cell,
                    "test": test,
                    "beta": fmt_beta(beta),
                    "instance_seed": seed,
                    "accept_e2e": _as_int(f[ACCEPT_COL[test]]),
                    "feasible_e2e_all": _as_int(f["feasible_e2e_all"]),
                    "ffl_sum": fmt_float(adj_v1["ffl_sum"]),
                    "n1_sum": fmt_float(adj_v1["n1_sum"]),
                    "n2_sum": fmt_float(adj_v1["n2_sum"]),
                    "n3_sum": fmt_float(adj_v1["n3_sum"]),
                    "partition_residual_max": fmt_float(
                        adj_v1["partition_residual_max"]
                    ),
                    "r1_holds": 1 if extra["r1_holds"] else 0,
                    "verdict_v2": extra["verdict_v2"],
                    "n_miss_episodes": extra["n_miss_episodes"],
                    "n3_jobs": extra["n3_jobs"],
                    "plan": f.get("plan", ""),
                }
            )
    return rows


def build_overrejection(feas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            feas_bits = _aligned_bits(feas, cell, float(beta), "feasible_e2e_all")
            n_feasible = int(feas_bits.sum())
            for test in TESTS:
                accept_bits = _aligned_bits(feas, cell, float(beta), ACCEPT_COL[test])
                n_rej_feas = int(((1 - accept_bits) * feas_bits).sum())
                hat = (n_rej_feas / n_feasible) if n_feasible else float("nan")
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                lo, hi, n_undef = bootstrap_over_rejection(
                    feas_bits, accept_bits, rng
                )
                rows.append(
                    {
                        "cell": cell,
                        "test": test,
                        "beta": fmt_beta(float(beta)),
                        "n_instances": N_INSTANCES,
                        "n_feasible_e2e_all": n_feasible,
                        "n_rej_feas": n_rej_feas,
                        "over_rejection_hat": fmt_float(hat),
                        "ci_lo": fmt_float(lo),
                        "ci_hi": fmt_float(hi),
                        "n_undefined": n_undef,
                        "n_bootstrap": N_BOOTSTRAP,
                    }
                )
    return rows


def build_ceiling(
    feas: Sequence[Dict[str, Any]], *, flag: str = "feasible_e2e_all"
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            bits = _aligned_bits(feas, cell, float(beta), flag)
            n_feasible = int(bits.sum())
            frac = n_feasible / float(N_INSTANCES)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            lo, hi = bootstrap_feasible_frac(bits, rng)
            rows.append(
                {
                    "cell": cell,
                    "beta": fmt_beta(float(beta)),
                    "n_instances": N_INSTANCES,
                    "n_feasible": n_feasible,
                    "feasible_frac": fmt_float(frac),
                    "ci_lo": fmt_float(lo),
                    "ci_hi": fmt_float(hi),
                }
            )
    return rows


def point_over_rejection_from_feas(
    feas: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str, float], Tuple[int, int, float]]:
    """(n_rej_feas, n_feasible, hat) per (cell, test, beta)."""
    out: Dict[Tuple[str, str, float], Tuple[int, int, float]] = {}
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            feas_bits = _aligned_bits(feas, cell, float(beta), "feasible_e2e_all")
            n_feasible = int(feas_bits.sum())
            for test in TESTS:
                accept_bits = _aligned_bits(feas, cell, float(beta), ACCEPT_COL[test])
                n_rej_feas = int(((1 - accept_bits) * feas_bits).sum())
                hat = (n_rej_feas / n_feasible) if n_feasible else float("nan")
                out[(cell, test, float(beta))] = (n_rej_feas, n_feasible, hat)
    return out


def global_timing_conditions(
    runs: Sequence[Dict[str, Any]],
    adjudication: Sequence[Dict[str, Any]],
    *,
    tol: float = PARTITION_TOL,
) -> Dict[str, Any]:
    """Three conditions under which the verdict column has no TIMING_VIOLATION.

    1. sum(FFL) == 0
    2. sum(rt_bound_violations) == 0
    3. no accepted instance has partition_residual_max > tol
    """
    ffl_sum = sum(_as_float(r["n_miss_full_fidelity_late"]) for r in runs)
    rt_sum = sum(_as_float(r["rt_bound_violations"]) for r in runs)
    acc_resid = False
    for row in adjudication:
        if _as_int(row["accept_e2e"]) and _as_float(row["partition_residual_max"]) > tol:
            acc_resid = True
            break
    has_tv = any(row["verdict"] == "TIMING_VIOLATION" for row in adjudication)
    all_hold = (ffl_sum == 0.0) and (rt_sum == 0.0) and (not acc_resid)
    return {
        "ffl_sum": ffl_sum,
        "rt_sum": rt_sum,
        "accepted_residual_gt_tol": acc_resid,
        "all_three_hold": all_hold,
        "has_timing_violation": has_tv,
    }


def load_sstar_test(grid_path: Path) -> Dict[Tuple[str, int, str], float]:
    """load_e2e at deployed deadlines (beta=1.0), keyed (cell, seed, test)."""
    out: Dict[Tuple[str, int, str], float] = {}
    for row in read_csv(grid_path):
        if abs(_as_float(row["beta"]) - 1.0) >= 1e-12:
            continue
        cell = f"{row['benchmark']}:{row['stress']}"
        key = (cell, _as_int(row["instance_seed"]), row["test"])
        out[key] = _as_float(row["load_e2e"])
    return out


def frontier_lower_bounds(
    feas: Sequence[Dict[str, Any]],
    sstar: Dict[Tuple[str, int, str], float],
) -> List[Dict[str, Any]]:
    """Per feasible instance: s*_emp left-censored at the smallest tested s."""
    by_inst: Dict[Tuple[str, int], List[Tuple[float, int]]] = {}
    for f in feas:
        key = (f["cell"], _as_int(f["instance_seed"]))
        by_inst.setdefault(key, []).append(
            (_as_float(f["beta"]), _as_int(f["feasible_e2e_all"]))
        )
    rows: List[Dict[str, Any]] = []
    for (cell, seed), pairs in sorted(by_inst.items()):
        pairs.sort(key=lambda p: p[0])
        first = next((b for b, bit in pairs if bit == 1), None)
        if first is None:
            continue
        for test in TESTS:
            s_test = sstar[(cell, seed, test)]
            ratio_lb = s_test / first
            rows.append(
                {
                    "cell": cell,
                    "instance_seed": seed,
                    "test": test,
                    "s_star_test": s_test,
                    "s_emp_ub": first,
                    "ratio_lb": ratio_lb,
                    "left_censored_at": first,
                }
            )
    return rows


def jobcount(row: Dict[str, Any]) -> float:
    return (
        _as_float(row["n_miss_full_fidelity_late"])
        + _as_float(row["n_miss_n1_drop"])
        + _as_float(row["n_miss_n2_fallback_insufficient"])
        + _as_float(row["n_miss_n3_unreached"])
    )


def r1_deviation(row: Dict[str, Any]) -> float:
    """tmr - (FFL+N1+N2+N3). Equals residual - 3*tmr (episode-binary / job-count).

    Detection limit: aggregate accounting cannot exclude a zero-channel
    episode cancelled by a multi-job episode within the same run; medium
    is structurally immune (d_v equals the mission deadline there, so such
    an episode would raise FFL, which is zero), and every accepted-instance
    run satisfies strict R1, where the claim has force.
    """
    return residual_of_run(row) - 3.0 * _as_float(row["task_miss_rate"])


DETECTION_LIMIT = (
    "aggregate accounting cannot exclude a zero-channel episode cancelled "
    "by a multi-job episode within the same run; medium is structurally "
    "immune (d_v equals the mission deadline there, so such an episode "
    "would raise FFL, which is zero), and every accepted-instance run "
    "satisfies strict R1, where the claim has force."
)


def g5_holds_run(row: Dict[str, Any], *, tol: float = R1_TOL) -> bool:
    """Directional identity: positive dev is an uncancelled zero-channel miss."""
    return r1_deviation(row) <= tol


def r1_holds_run(row: Dict[str, Any], *, tol: float = R1_TOL) -> bool:
    return abs(r1_deviation(row)) < tol


def r1_census(
    runs: Sequence[Dict[str, Any]], *, tol: float = R1_TOL
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """Return (n_pass, n_fail, max_|dev|, failing run records)."""
    n_pass = 0
    n_fail = 0
    max_abs = 0.0
    fails: List[Dict[str, Any]] = []
    for row in runs:
        dev = r1_deviation(row)
        ad = abs(dev)
        if ad > max_abs:
            max_abs = ad
        rec = {
            "cell": row["cell"],
            "beta": _as_float(row["beta"]),
            "instance_seed": _as_int(row["instance_seed"]),
            "ckpt_seed": _as_int(row["ckpt_seed"]),
            "task_miss_rate": _as_float(row["task_miss_rate"]),
            "partition_residual": residual_of_run(row),
            "r1_deviation": dev,
            "n3": _as_float(row["n_miss_n3_unreached"]),
            "ffl": _as_float(row["n_miss_full_fidelity_late"]),
            "n1": _as_float(row["n_miss_n1_drop"]),
            "n2": _as_float(row["n_miss_n2_fallback_insufficient"]),
        }
        if ad < tol:
            n_pass += 1
        else:
            n_fail += 1
            fails.append(rec)
    return n_pass, n_fail, max_abs, fails


def _truthy_accept(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    try:
        return int(float(v)) != 0
    except (TypeError, ValueError):
        return False


def accepted_instance_keys(
    feas: Sequence[Dict[str, Any]],
) -> set:
    """(cell, beta, instance_seed) with s0_accept_e2e OR s1_accept_e2e."""
    keys = set()
    for f in feas:
        if _truthy_accept(f.get("s0_accept_e2e")) or _truthy_accept(
            f.get("s1_accept_e2e")
        ):
            keys.add(
                (f["cell"], _as_float(f["beta"]), _as_int(f["instance_seed"]))
            )
    return keys


def g5_census(
    runs: Sequence[Dict[str, Any]], *, tol: float = R1_TOL
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """Return (n_positive, max signed dev, positive-dev records)."""
    pos: List[Dict[str, Any]] = []
    max_signed = 0.0
    for row in runs:
        dev = r1_deviation(row)
        if dev > max_signed:
            max_signed = dev
        if dev > tol:
            pos.append(
                {
                    "cell": row["cell"],
                    "beta": _as_float(row["beta"]),
                    "instance_seed": _as_int(row["instance_seed"]),
                    "ckpt_seed": _as_int(row["ckpt_seed"]),
                    "task_miss_rate": _as_float(row["task_miss_rate"]),
                    "n3": _as_float(row["n_miss_n3_unreached"]),
                    "r1_deviation": dev,
                }
            )
    return len(pos), max_signed, pos


def g6_census(
    runs: Sequence[Dict[str, Any]],
    feas: Sequence[Dict[str, Any]],
    *,
    tol: float = R1_TOL,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Strict |dev|<tol on runs whose instance is offline-accepted. (n_checked, n_fail, fails)."""
    acc = accepted_instance_keys(feas)
    n_checked = 0
    fails: List[Dict[str, Any]] = []
    for row in runs:
        key = (
            row["cell"],
            _as_float(row["beta"]),
            _as_int(row["instance_seed"]),
        )
        if key not in acc:
            continue
        n_checked += 1
        if not r1_holds_run(row, tol=tol):
            fails.append(
                {
                    "cell": row["cell"],
                    "beta": _as_float(row["beta"]),
                    "instance_seed": _as_int(row["instance_seed"]),
                    "ckpt_seed": _as_int(row["ckpt_seed"]),
                    "r1_deviation": r1_deviation(row),
                }
            )
    return n_checked, len(fails), fails


def _int_check(x: float, name: str, *, tol: float = R1_TOL) -> int:
    r = round(float(x))
    if abs(float(x) - r) > tol:
        raise AnalysisError(f"{name} is not an integer: {x!r}")
    return int(r)


def adjudicate_instance_v2(
    accept_e2e: int,
    feasible_e2e_all: int,
    inst_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Corrected verdict. R1 failure on any of the 3 runs keeps TIMING_VIOLATION."""
    r1 = all(r1_holds_run(r) for r in inst_runs)
    ffl_sum = float(sum(_as_float(r["n_miss_full_fidelity_late"]) for r in inst_runs))
    rt_sum = float(sum(_as_float(r["rt_bound_violations"]) for r in inst_runs))
    n1_sum = float(sum(_as_float(r["n_miss_n1_drop"]) for r in inst_runs))
    n2_sum = float(sum(_as_float(r["n_miss_n2_fallback_insufficient"]) for r in inst_runs))
    n3_sum = float(sum(_as_float(r["n_miss_n3_unreached"]) for r in inst_runs))
    timing_violation_v2 = bool(accept_e2e) and (
        ffl_sum > 0.0 or rt_sum > 0.0 or (not r1)
    )
    if timing_violation_v2:
        verdict_v2 = "TIMING_VIOLATION"
    elif bool(accept_e2e) and (not bool(feasible_e2e_all)):
        verdict_v2 = "energy_scope_miss"
    else:
        verdict_v2 = "consistent"
    n_miss_episodes = _int_check(
        sum(_as_float(r["task_miss_rate"]) * N_EVAL_EPISODES for r in inst_runs),
        "n_miss_episodes",
    )
    n3_jobs = _int_check(n3_sum * N_EVAL_EPISODES, "n3_jobs")
    return {
        "r1_holds": r1,
        "verdict_v2": verdict_v2,
        "n_miss_episodes": n_miss_episodes,
        "n3_jobs": n3_jobs,
        "ffl_sum": ffl_sum,
        "rt_sum": rt_sum,
        "n1_sum": n1_sum,
        "n2_sum": n2_sum,
        "n3_sum": n3_sum,
    }


def build_adjudication_v2(
    runs: Sequence[Dict[str, Any]],
    adj_v1: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by = _runs_by_instance(runs)
    rows: List[Dict[str, Any]] = []
    for row in adj_v1:
        cell = row["cell"]
        beta = _as_float(row["beta"])
        seed = _as_int(row["instance_seed"])
        inst_runs = by.get((cell, beta, seed), [])
        if len(inst_runs) != 3:
            raise AnalysisError(
                f"v2 expected 3 runs for {cell} beta={beta} seed={seed}, "
                f"got {len(inst_runs)}"
            )
        extra = adjudicate_instance_v2(
            _as_int(row["accept_e2e"]),
            _as_int(row["feasible_e2e_all"]),
            inst_runs,
        )
        out = dict(row)
        out["r1_holds"] = 1 if extra["r1_holds"] else 0
        out["verdict_v1"] = row["verdict"]
        out["verdict_v2"] = extra["verdict_v2"]
        out["n_miss_episodes"] = extra["n_miss_episodes"]
        out["n3_jobs"] = extra["n3_jobs"]
        rows.append(out)
    return rows


def _print_census(census: Sequence[Dict[str, Any]], max_r: float, n_nz: int) -> None:
    print(f"partition residual: n_nonzero={n_nz}/{len(census)}  max={max_r:.17g}")
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for rec in census:
        if rec["partition_residual"] > PARTITION_TOL:
            grouped.setdefault((rec["cell"], rec["beta"]), []).append(rec)
    for key in sorted(grouped, key=lambda k: (k[0], k[1])):
        recs = grouped[key]
        ids = ", ".join(
            f"i{r['instance_seed']}c{r['ckpt_seed']}={r['partition_residual']:.4g}"
            for r in sorted(recs, key=lambda r: (r["instance_seed"], r["ckpt_seed"]))
        )
        print(f"  {key[0]} s={key[1]:g} n={len(recs)}: {ids}")


def _main_p7(
    runs: List[Dict[str, str]],
    feas: List[Dict[str, str]],
    grid_path: Path,
    out_dir: Path,
) -> int:
    if len(runs) != 420:
        raise AnalysisError(f"p7 expected 420 runs, got {len(runs)}")
    if len(feas) != 140:
        raise AnalysisError(f"p7 expected 140 feas rows, got {len(feas)}")
    n_pass, n_fail, max_dev, fails = r1_census(runs)
    print(
        f"INFO strict-R1 census {n_pass}/420  n_fail={n_fail}  "
        f"max_|dev|={max_dev:.17g}  (not gated)"
    )
    n_pos, max_signed, pos = g5_census(runs)
    print(
        f"G5 directional identity n_pos={n_pos}/420  "
        f"max_signed_dev={max_signed:.17g}"
    )
    if n_pos:
        print("G5 POSITIVE-DEV RUNS (verbatim):")
        for rec in pos:
            print(
                f"  {rec['cell']} s={rec['beta']} i{rec['instance_seed']}"
                f"c{rec['ckpt_seed']} tmr={rec['task_miss_rate']:.17g} "
                f"N3={rec['n3']:.17g} dev={rec['r1_deviation']:.17g}"
            )
        raise AnalysisError("G5 failed on the merged 420; STOP")
    n_checked, n_g6, g6_fails = g6_census(runs, feas)
    print(
        f"G6 strict identity on accepted-instance runs "
        f"n_checked={n_checked} n_fail={n_g6}"
    )
    if n_g6:
        print("G6 FAILING RUNS (verbatim):")
        for rec in g6_fails:
            print(
                f"  {rec['cell']} s={rec['beta']} i{rec['instance_seed']}"
                f"c{rec['ckpt_seed']} dev={rec['r1_deviation']:.17g}"
            )
        raise AnalysisError("G6 failed; STOP")
    if fails:
        print("INFO strict-R1 failures (directional only; G5/G6 hold):")
        for rec in fails:
            print(
                f"  {rec['cell']} s={rec['beta']} i{rec['instance_seed']}"
                f"c{rec['ckpt_seed']} tmr={rec['task_miss_rate']:.17g} "
                f"N3={rec['n3']:.17g} dev={rec['r1_deviation']:.17g}"
            )
    adj = build_adjudication_p7(runs, feas)
    over = build_overrejection_p7(feas)
    ceiling = build_ceiling_p7(feas)
    sstar = load_sstar_test(grid_path)
    frontier = frontier_lower_bounds(feas, sstar)
    n_new_acc = 0
    n_new = 0
    for r in adj:
        if abs(_as_float(r["beta"]) - 1.0) > 1e-12:
            continue
        if _as_int(r["instance_seed"]) < 10:
            continue
        n_new += 1
        if _as_int(r["accept_e2e"]):
            n_new_acc += 1
        if _as_int(r["accept_e2e"]) and r["verdict_v2"] != "consistent":
            raise AnalysisError(
                f"new seed accept_e2e=1 at deployed: {r['cell']} "
                f"i{r['instance_seed']} {r['test']}"
            )
        if (not _as_int(r["accept_e2e"])) and r["verdict_v2"] != "consistent":
            raise AnalysisError(
                f"new rejected row not consistent: {r['cell']} "
                f"i{r['instance_seed']} {r['test']} {r['verdict_v2']}"
            )
    print(f"new deployed-deadline adjudication rows: {n_new}; accept_e2e=1: {n_new_acc}")
    if n_new_acc != 0:
        raise AnalysisError("grid said 0/200 accept at scale 1.0; STOP")
    uncertified = True
    for f in feas:
        if abs(_as_float(f["beta"]) - 1.0) > 1e-12:
            continue
        if _as_int(f["feasible_e2e_all"]) and (
            _as_int(f["s0_accept_e2e"]) or _as_int(f["s1_accept_e2e"])
        ):
            uncertified = False
            print(
                f"FEASIBLE AND ACCEPTED: {f['cell']} i{f['instance_seed']}"
            )
    print(
        "every empirically feasible instance at deployed deadlines is "
        f"uncertified: {uncertified}"
    )
    if not uncertified:
        raise AnalysisError("feasible instance accepted at deployed deadlines; STOP")
    write_csv(out_dir / "tb3_empirical_ceiling_p7.csv", TB3_CEILING_P7_HEADER, ceiling)
    write_csv(out_dir / "tb3_overrejection_p7.csv", TB3_OVERREJ_P7_HEADER, over)
    write_csv(
        out_dir / "tb3_soundness_adjudication_p7.csv", TB3_ADJUDICATION_P7_HEADER, adj
    )
    front_rows = []
    for r in frontier:
        if abs(float(r["s_emp_ub"]) - 1.0) > 1e-12:
            continue
        front_rows.append(
            {
                "cell": r["cell"],
                "instance_seed": r["instance_seed"],
                "test": r["test"],
                "s_star_test": fmt_float(r["s_star_test"]),
                "s_emp_ub": fmt_float(r["s_emp_ub"]),
                "ratio_lb": fmt_float(r["ratio_lb"]),
                "left_censored_at": fmt_float(r["left_censored_at"]),
            }
        )
    write_csv(
        out_dir / "tb3_frontier_lb_p7.csv", TB3_FRONTIER_LB_P7_HEADER, front_rows
    )
    print(f"Wrote p7 ceiling ({len(ceiling)}), overrej ({len(over)}), "
          f"adjudication ({len(adj)}), frontier_lb ({len(front_rows)})")
    print("verdict_v2:", dict(Counter(r["verdict_v2"] for r in adj)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        default=str(Path(_PROJECT_ROOT) / "docs" / "tb3" / "tb3_empirical_runs.csv"),
    )
    ap.add_argument(
        "--feas",
        default=str(
            Path(_PROJECT_ROOT) / "docs" / "tb3" / "tb3_empirical_feasibility.csv"
        ),
    )
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    ap.add_argument(
        "--out", default=str(Path(_PROJECT_ROOT) / "docs" / "tb3")
    )
    ap.add_argument(
        "--p7",
        action="store_true",
        default=False,
        help="Merged 420-run P7 pass: CP intervals, p7 CSVs, do not rewrite p5b/p6 files.",
    )
    args = ap.parse_args(argv)

    grid_path = Path(args.grid)
    digest = verify_grid_sha(grid_path)
    print(f"grid sha256 {digest}  OK")

    runs = read_csv(Path(args.runs))
    feas = read_csv(Path(args.feas))
    out_dir = Path(args.out)
    if args.p7:
        return _main_p7(runs, feas, grid_path, out_dir)
    if len(runs) != 180:
        raise AnalysisError(f"expected 180 runs, got {len(runs)}")
    if len(feas) != 60:
        raise AnalysisError(f"expected 60 feas rows, got {len(feas)}")

    census, max_r, n_nz = residual_census(runs)
    _print_census(census, max_r, n_nz)

    adj = build_adjudication(runs, feas)
    over = build_overrejection(feas)
    ceiling = build_ceiling(feas)
    ceiling_maj = build_ceiling(feas, flag="feasible_e2e_majority")
    cond = global_timing_conditions(runs, adj)
    print(
        "global conditions: "
        f"sum(FFL)={cond['ffl_sum']} sum(rt)={cond['rt_sum']} "
        f"accepted_residual_gt_tol={cond['accepted_residual_gt_tol']} "
        f"all_three_hold={cond['all_three_hold']} "
        f"has_TIMING_VIOLATION={cond['has_timing_violation']}"
    )
    print("verdict counts:", dict(Counter(r["verdict"] for r in adj)))
    tv = [r for r in adj if r["verdict"] == "TIMING_VIOLATION"]
    print(f"TIMING_VIOLATION rows: {len(tv)}")
    for r in tv:
        print(
            f"  {r['cell']} s={r['beta']} {r['test']} i{r['instance_seed']} "
            f"resid_max={r['partition_residual_max']} n3_sum={r['n3_sum']}"
        )

    sstar = load_sstar_test(grid_path)
    frontier = frontier_lower_bounds(feas, sstar)
    print("frontier lower bounds (s*_test / s_emp_ub; left-censored):")
    for r in frontier:
        print(
            f"  {r['cell']} i{r['instance_seed']} {r['test']} "
            f"s*_test={r['s_star_test']:.17g} s_emp_ub={r['s_emp_ub']} "
            f"ratio_lb={r['ratio_lb']:.17g}"
        )
    print("majority ceiling (report table only):")
    for r in ceiling_maj:
        print(
            f"  {r['cell']} s={r['beta']} n_feasible={r['n_feasible']} "
            f"frac={r['feasible_frac']} ci=[{r['ci_lo']}, {r['ci_hi']}]"
        )

    out_dir = Path(args.out)
    v1_path = out_dir / "tb3_soundness_adjudication.csv"
    if v1_path.is_file():
        v1_sha = sha256_file(v1_path)
        if v1_sha != EXPECTED_ADJ_V1_SHA256:
            raise AnalysisError(
                f"frozen v1 CSV SHA mismatch: {v1_sha} != {EXPECTED_ADJ_V1_SHA256}"
            )
        print(f"v1 csv sha256 {v1_sha}  frozen OK (not rewritten)")
    else:
        write_csv(v1_path, TB3_ADJUDICATION_HEADER, adj)
        print(f"Wrote {v1_path} ({len(adj)} rows)")

    n_pass, n_fail, max_dev, fails = r1_census(runs)
    print(f"R1: n_pass={n_pass}/180  n_fail={n_fail}  max_|dev|={max_dev:.17g}")
    if fails:
        print("R1 FAILING RUNS (verbatim; verdict stays TIMING_VIOLATION):")
        for rec in fails:
            print(
                f"  {rec['cell']} s={rec['beta']} i{rec['instance_seed']}"
                f"c{rec['ckpt_seed']} tmr={rec['task_miss_rate']:.17g} "
                f"resid={rec['partition_residual']:.17g} "
                f"dev={rec['r1_deviation']:.17g} n3={rec['n3']} "
                f"ffl={rec['ffl']} n1={rec['n1']} n2={rec['n2']}"
            )
    adj_v2 = build_adjudication_v2(runs, adj)
    v2_counts = Counter(r["verdict_v2"] for r in adj_v2)
    print("verdict_v2 counts:", dict(v2_counts))
    n_tv_v2 = sum(1 for r in adj_v2 if r["verdict_v2"] == "TIMING_VIOLATION")
    print(f"sum TIMING_VIOLATION v2: {n_tv_v2}")
    v1_tv = [r for r in adj_v2 if r["verdict_v1"] == "TIMING_VIOLATION"]
    print(f"v1 TIMING_VIOLATION rows ({len(v1_tv)}) -> v2:")
    for r in v1_tv:
        print(
            f"  {r['cell']} s={r['beta']} {r['test']} i{r['instance_seed']} "
            f"v1={r['verdict_v1']} v2={r['verdict_v2']} r1={r['r1_holds']} "
            f"n_miss_ep={r['n_miss_episodes']} n3_jobs={r['n3_jobs']}"
        )
    v1_cons = [r for r in adj_v2 if r["verdict_v1"] == "consistent"]
    n_cons_flip = sum(1 for r in v1_cons if r["verdict_v2"] != "consistent")
    print(
        f"v1 consistent rows: {len(v1_cons)}; "
        f"unchanged under v2: {len(v1_cons) - n_cons_flip}; flipped: {n_cons_flip}"
    )
    n1n2_bad = [
        r
        for r in adj_v2
        if _as_float(r["n1_sum"]) != 0.0 or _as_float(r["n2_sum"]) != 0.0
    ]
    print(f"rows with N1 or N2 != 0: {len(n1n2_bad)}")

    write_csv(out_dir / "tb3_soundness_adjudication_v2.csv", TB3_ADJUDICATION_V2_HEADER, adj_v2)
    print(
        f"Wrote {out_dir / 'tb3_soundness_adjudication_v2.csv'} ({len(adj_v2)} rows)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
