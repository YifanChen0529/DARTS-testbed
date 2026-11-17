"""Standalone synchronous CBBA microbenchmark core.

(1) Implements Choi, Brunet, How, "Consensus-Based Decentralized Auctions
    for Robust Task Allocation", IEEE T-RO 25(4), 2009 (bundle construction
    plus the Table I action table, run as a lockstep full-broadcast driver).

(2) Quarantine: this module imports only the Python standard library and
    nothing from the DARTS codebase, so the benchmark is open-sourceable
    alongside the technical report.

(3) Scope notes: travel time is straight-line Euclidean (obstacles are not
    modeled); task positions AND agent-start positions are sampled uniformly
    over the full workspace rectangle per trial (recorded adaptation below);
    latency is the only measured axis and no solution-quality comparison
    is made.

Recorded design adaptation (start positions; Yifan, 2026-08-29): the EXP-3
design doc assumed four fixed parking positions as agent starts. T0 R6(b)
established that no such four-tuple exists at HEAD. The evaluation convention
the paper's numbers were measured under samples agent starts uniformly: eval
constructors never pass init_positions (alg/train_darts.py:2141-2149,
alg/agent_node.py:131-138) and the env samples starts when None
(envs/turtlebot3_spatial_env.py:541-548, :559-565). The offline (0,0) corner
(scripts/r_off_g.py:13-14) is a bound-side parking convention, a single point,
and degenerate as a score landscape. Agent start positions are therefore
sampled uniformly per trial together with task positions, seeded, as part of
the instance. Obstacles are not modeled anywhere in this benchmark
(straight-line travel, full-rectangle sampling).
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

UNASSIGNED = -1

# source: envs/turtlebot3_spatial_env.py:55 @ 57187b7
WORKSPACE_RECT = (0.0, 0.0, 10.0, 10.0)

# source: envs/turtlebot3_spatial_env.py:45 @ 57187b7
SPEED_MPS = 0.22

# EXP-3 design constants, recorded values
LAMBDA_DISCOUNT = 0.95
TASK_REWARD = 100.0

Point = Tuple[float, float]


@dataclass
class AgentState:
    """Local CBBA state for one agent."""

    agent_id: int
    start: Point
    bundle: List[int]
    path: List[int]
    z: List[int]
    y: List[float]
    s: List[int]


@dataclass(frozen=True)
class Message:
    """Frozen (z, y, s) snapshot sent by ``sender_id`` in one round."""

    sender_id: int
    z: Tuple[int, ...]
    y: Tuple[float, ...]
    s: Tuple[int, ...]


@dataclass(frozen=True)
class SyncResult:
    """Outcome of ``run_sync_instance``."""

    n_rounds: int
    per_round_digests: Tuple[Tuple[str, ...], ...]
    assignment: Dict[int, int]
    valid: bool
    agent_convergence_round: Tuple[int, ...]


def fresh_state(agent_id: int, start: Point, m: int, n: int) -> AgentState:
    """Return an empty local state for ``agent_id`` on an m-task, n-agent instance."""
    return AgentState(
        agent_id=int(agent_id),
        start=(float(start[0]), float(start[1])),
        bundle=[],
        path=[],
        z=[UNASSIGNED] * m,
        y=[0.0] * m,
        s=[0] * n,
    )


def generate_instance(
    seed: int, m: int, n: int = 4
) -> Tuple[List[Point], List[Point]]:
    # Recorded adaptation (Yifan, 2026-08-29): no parking four-tuple exists at
    # HEAD (T0 R6(b)). Eval constructors never pass init_positions
    # (alg/train_darts.py:2141-2149, alg/agent_node.py:131-138); the env samples
    # starts when None (envs/turtlebot3_spatial_env.py:541-548, :559-565). The
    # offline (0,0) corner (scripts/r_off_g.py:13-14) is a single bound-side
    # point, degenerate as a score landscape. Task positions are drawn first,
    # then n agent starts, both uniform over WORKSPACE_RECT, in that fixed order
    # so instances are reproducible. The algorithm itself contains no randomness.
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = WORKSPACE_RECT
    task_positions = [
        (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(m)
    ]
    starts = [
        (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(n)
    ]
    return task_positions, starts


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_score(
    path: Sequence[int], start: Point, task_positions: Sequence[Point]
) -> float:
    """S(p; a): discounted-reward sum along straight-line execution of ``path``."""
    pos = start
    dist = 0.0
    total = 0.0
    for j in path:
        dist += _euclid(pos, task_positions[j])
        pos = task_positions[j]
        t_k = dist / SPEED_MPS
        total += (LAMBDA_DISCOUNT ** t_k) * TASK_REWARD
    return total


def _best_insertion(
    path: Sequence[int],
    start: Point,
    task_positions: Sequence[Point],
    j: int,
    score_without: float,
) -> Tuple[float, int]:
    """Return (max marginal, smallest insertion index achieving it)."""
    best_c = None
    best_idx = 0
    for idx in range(len(path) + 1):
        new_path = list(path[:idx]) + [j] + list(path[idx:])
        c = path_score(new_path, start, task_positions) - score_without
        if best_c is None or c > best_c:
            best_c = c
            best_idx = idx
    assert best_c is not None
    return best_c, best_idx


def build_bundle(
    state: AgentState, task_positions: Sequence[Point], L_t: int
) -> bool:
    """Greedy bundle construction (Choi et al. 2009, Algorithm 1).

    Bid ties are broken deterministically in favor of the smaller agent id,
    applied identically here and in consensus rule 1; scores are never
    perturbed (design rule: seeded id-based tie-breaking only).
    """
    m = len(task_positions)
    changed = False
    bundle_set = set(state.bundle)
    while len(state.bundle) < L_t:
        score_without = path_score(state.path, state.start, task_positions)
        best_j: Optional[int] = None
        best_c = 0.0
        best_idx = 0
        for j in range(m):
            if j in bundle_set:
                continue
            c_ij, ins_idx = _best_insertion(
                state.path, state.start, task_positions, j, score_without
            )
            # Eligible iff we strictly outbid the current winner, or we match
            # the bid and have the smaller agent id (same rule as consensus 1).
            if c_ij > state.y[j] or (
                c_ij == state.y[j] and state.agent_id < state.z[j]
            ):
                if best_j is None or c_ij > best_c:
                    best_j = j
                    best_c = c_ij
                    best_idx = ins_idx
        if best_j is None:
            break
        state.path.insert(best_idx, best_j)
        state.bundle.append(best_j)
        bundle_set.add(best_j)
        state.z[best_j] = state.agent_id
        state.y[best_j] = best_c
        changed = True
    return changed


def _action(
    i: int,
    k: int,
    z_i_j: int,
    z_k_j: int,
    y_i_j: float,
    y_k_j: float,
    s_i: Sequence[int],
    s_k: Sequence[int],
) -> str:
    """Return UPDATE / RESET / LEAVE for one (sender k, receiver i, task j)."""
    # Sender k believes the winner of j is k itself.
    if z_k_j == k:
        if z_i_j == i:
            # Choi et al. 2009, Table I, rule 1
            if y_k_j > y_i_j or (y_k_j == y_i_j and k < i):
                return "UPDATE"
            return "LEAVE"
        if z_i_j == k:
            # Choi et al. 2009, Table I, rule 2
            return "UPDATE"
        if z_i_j != UNASSIGNED:
            # Choi et al. 2009, Table I, rule 3
            m = z_i_j
            if s_k[m] > s_i[m] or y_k_j > y_i_j:
                return "UPDATE"
            return "LEAVE"
        # Choi et al. 2009, Table I, rule 4
        return "UPDATE"

    # Sender k believes the winner of j is i (the receiver).
    if z_k_j == i:
        if z_i_j == i:
            # Choi et al. 2009, Table I, rule 5
            return "LEAVE"
        if z_i_j == k:
            # Choi et al. 2009, Table I, rule 6
            return "RESET"
        if z_i_j != UNASSIGNED:
            # Choi et al. 2009, Table I, rule 7
            m = z_i_j
            if s_k[m] > s_i[m]:
                return "RESET"
            return "LEAVE"
        # Choi et al. 2009, Table I, rule 8
        return "LEAVE"

    # Sender k believes the winner of j is some m not in {i, k}.
    if z_k_j != UNASSIGNED:
        m = z_k_j
        if z_i_j == i:
            # Choi et al. 2009, Table I, rule 9
            if s_k[m] > s_i[m] and y_k_j > y_i_j:
                return "UPDATE"
            return "LEAVE"
        if z_i_j == k:
            # Choi et al. 2009, Table I, rule 10
            if s_k[m] > s_i[m]:
                return "UPDATE"
            return "RESET"
        if z_i_j == m:
            # Choi et al. 2009, Table I, rule 11
            if s_k[m] > s_i[m]:
                return "UPDATE"
            return "LEAVE"
        if z_i_j != UNASSIGNED:
            # Choi et al. 2009, Table I, rule 12
            n = z_i_j
            if s_k[m] > s_i[m] and s_k[n] > s_i[n]:
                return "UPDATE"
            if s_k[m] > s_i[m] and y_k_j > y_i_j:
                return "UPDATE"
            if s_k[n] > s_i[n] and s_i[m] > s_k[m]:
                return "RESET"
            return "LEAVE"
        # Choi et al. 2009, Table I, rule 13
        if s_k[m] > s_i[m]:
            return "UPDATE"
        return "LEAVE"

    # Sender k believes the winner of j is none.
    if z_i_j == i:
        # Choi et al. 2009, Table I, rule 14
        return "LEAVE"
    if z_i_j == k:
        # Choi et al. 2009, Table I, rule 15
        return "UPDATE"
    if z_i_j != UNASSIGNED:
        # Choi et al. 2009, Table I, rule 16
        m = z_i_j
        if s_k[m] > s_i[m]:
            return "UPDATE"
        return "LEAVE"
    # Choi et al. 2009, Table I, rule 17
    return "LEAVE"


def _apply_action(state: AgentState, msg: Message, j: int, action: str) -> None:
    if action == "UPDATE":
        state.z[j] = msg.z[j]
        state.y[j] = msg.y[j]
    elif action == "RESET":
        state.z[j] = UNASSIGNED
        state.y[j] = 0.0


def consensus_update(
    state: AgentState, incoming: Sequence[Message], round_idx: int
) -> bool:
    """Apply Table I to ``incoming`` (ascending sender_id), then bundle-release.

    Returns True iff (z, y) changed during this call (timestamp-only updates
    do not count).
    """
    z_before = list(state.z)
    y_before = list(state.y)
    i = state.agent_id
    m_tasks = len(state.z)
    ordered = sorted(incoming, key=lambda msg: msg.sender_id)
    for msg in ordered:
        k = msg.sender_id
        # Timestamp update before the action table for this sender.
        state.s[k] = round_idx
        for q in range(len(state.s)):
            if q == i or q == k:
                continue
            state.s[q] = max(state.s[q], msg.s[q])
        for j in range(m_tasks):
            action = _action(
                i, k, state.z[j], msg.z[j], state.y[j], msg.y[j], state.s, msg.s
            )
            _apply_action(state, msg, j, action)

    # Bundle-release: earliest bundle index whose winner is no longer us.
    l_star = None
    for idx, j in enumerate(state.bundle):
        if state.z[j] != i:
            l_star = idx
            break
    if l_star is not None:
        released = state.bundle[l_star:]
        state.bundle = state.bundle[:l_star]
        released_set = set(released)
        state.path = [t for t in state.path if t not in released_set]
        for offset, j in enumerate(released):
            if offset == 0:
                continue  # index l keeps whatever the action table wrote
            if state.z[j] == i:
                state.z[j] = UNASSIGNED
                state.y[j] = 0.0

    return state.z != z_before or state.y != y_before


def _digest(z: Sequence[int], y: Sequence[float]) -> str:
    payload = repr((tuple(z), tuple(y))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot(state: AgentState) -> Message:
    return Message(
        sender_id=state.agent_id,
        z=tuple(state.z),
        y=tuple(state.y),
        s=tuple(state.s),
    )


def run_sync_instance(
    task_positions: Sequence[Point],
    starts: Sequence[Point],
    L_t: int,
    max_rounds: Optional[int] = None,
) -> SyncResult:
    """In-process synchronous full-broadcast CBBA over ``len(starts)`` agents.

    Each round r = 1, 2, ...: (a) every agent ``build_bundle``; (b) snapshot
    (z, y, s); (c) every agent ``consensus_update`` on the n-1 peer snapshots;
    (d) record per-agent sha256 digests of (z, y).

    ``n_rounds`` is the last round in which any agent's (z, y) changed (the
    decision round whose outcome the following silent round confirms). A
    subsequent silent round is still executed to declare convergence; exceeding
    ``max_rounds`` without a silent round sets valid=False.
    """
    m = len(task_positions)
    n = len(starts)
    if max_rounds is None:
        max_rounds = 4 * L_t * m
    agents = [fresh_state(i, starts[i], m, n) for i in range(n)]
    agent_last_change = [0] * n
    per_round: List[Tuple[str, ...]] = []
    last_change_round = 0
    converged = False

    for r in range(1, max_rounds + 1):
        build_changed = [build_bundle(ag, task_positions, L_t) for ag in agents]
        snaps = [_snapshot(ag) for ag in agents]
        cons_changed = []
        for ag in agents:
            peers = [msg for msg in snaps if msg.sender_id != ag.agent_id]
            cons_changed.append(consensus_update(ag, peers, r))
        per_round.append(tuple(_digest(ag.z, ag.y) for ag in agents))
        any_change = False
        for i in range(n):
            if build_changed[i] or cons_changed[i]:
                agent_last_change[i] = r
                any_change = True
        if any_change:
            last_change_round = r
        else:
            converged = True
            break

    z0 = agents[0].z
    y0 = agents[0].y
    beliefs_identical = all(ag.z == z0 and ag.y == y0 for ag in agents)
    assignment = {j: z0[j] for j in range(m)}
    bundle_matches = True
    for ag in agents:
        won = {j for j in range(m) if ag.z[j] == ag.agent_id}
        if set(ag.bundle) != won:
            bundle_matches = False
            break
    all_assigned = True
    if n * L_t >= m:
        all_assigned = all(winner != UNASSIGNED for winner in z0)
    valid = (
        converged
        and beliefs_identical
        and bundle_matches
        and all_assigned
    )
    n_rounds = last_change_round if converged else max_rounds
    return SyncResult(
        n_rounds=n_rounds,
        per_round_digests=tuple(per_round),
        assignment=assignment,
        valid=valid,
        agent_convergence_round=tuple(agent_last_change),
    )
