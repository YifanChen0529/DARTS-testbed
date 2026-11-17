"""Unit tests for the quarantined synchronous CBBA core (EXP3-P1)."""

from __future__ import annotations

import math
from pathlib import Path

from benchmarks.cbba_microbench.cbba_core import (
    generate_instance,
    run_sync_instance,
)


def test_t1_hand_checkable_optima():
    """n=2, L_t=1, explicit geometry; starts passed explicitly."""
    # Instance A: opposite corners prefer distinct nearby tasks.
    result_a = run_sync_instance(
        task_positions=[(1.0, 0.0), (9.0, 10.0)],
        starts=[(0.0, 0.0), (10.0, 10.0)],
        L_t=1,
    )
    assert result_a.assignment == {0: 0, 1: 1}
    assert result_a.valid is True
    assert len(set(result_a.per_round_digests[-1])) == 1

    # Instance B: initial conflict on task 0; agent 0 is closer and outbids.
    result_b = run_sync_instance(
        task_positions=[(1.0, 0.0), (9.0, 0.0)],
        starts=[(0.0, 0.0), (3.0, 0.0)],
        L_t=1,
    )
    assert result_b.assignment == {0: 0, 1: 1}
    assert result_b.valid is True
    assert result_b.n_rounds <= 1 * 2


def test_t2_determinism():
    tasks, starts = generate_instance(seed=12345, m=8, n=4)
    a = run_sync_instance(tasks, starts, L_t=2)
    b = run_sync_instance(tasks, starts, L_t=2)
    assert a.n_rounds == b.n_rounds
    assert a.per_round_digests == b.per_round_digests
    assert a.assignment == b.assignment


def test_t3_cap_and_validity():
    n = 4
    for seed in range(200):
        for m in (4, 8, 16):
            for L_t in (1, math.ceil(m / 4)):
                tasks, starts = generate_instance(seed, m, n=n)
                result = run_sync_instance(tasks, starts, L_t=L_t)
                assert result.valid is True
                assert result.n_rounds <= L_t * m
                if n * L_t >= m:
                    assert all(w != -1 for w in result.assignment.values())


def test_t4_convergence_agreement():
    for seed in range(20):
        tasks, starts = generate_instance(seed, m=8, n=4)
        result = run_sync_instance(tasks, starts, L_t=2)
        rounds = result.agent_convergence_round
        assert len(set(rounds)) == 1, (seed, rounds)


def test_t5_quarantine_tripwire():
    src = Path("benchmarks/cbba_microbench/cbba_core.py").read_text(encoding="utf-8")
    forbidden = (
        "import alg",
        "from alg",
        "import envs",
        "from envs",
        "import scripts",
        "from scripts",
        "import numpy",
        "import torch",
        "import grpc",
    )
    for token in forbidden:
        assert token not in src, token
