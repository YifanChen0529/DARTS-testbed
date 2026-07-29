"""
Unit tests for CEA-EDF composite score S(i,v,t_k) = ε × σ/(σ+h).
Validates edge cases: full battery at task, far/low battery, two-agent trade-off, loaded vs empty queue.
Run: PYTHONPATH=. python -m pytest tests/test_cea_edf_score.py -v
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

pytest.importorskip("torch", reason="alg.darts_agent requires torch")

from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.dag_task import DAGGraph, TaskNode
from envs.workloads import get_workload
from alg.config_darts import DARTS_CONFIG
from alg.baseline_cea_edf import CentralizedDispatcher, CEAEDFAgent
from alg.darts_agent import DartsAgent


# B1_CDR task A location and a map corner far from it (matches env default workload layout)
_B1_LOC = get_workload("B1_CDR").task_locations_m
TASK_A_XY = _B1_LOC["A"]
FAR_FROM_TASK_A_XY = (0.5, 0.5)


def _make_env_and_agents(cfg: DARTS_CONFIG, num_agents: int = 2, init_positions=None):
    """Minimal env and CEAEDF agents for testing."""
    env = TurtleBot3Env(
        num_agents=num_agents,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=42,
    )
    env.reset(init_positions=init_positions)
    agents = [
        CEAEDFAgent(agent_id=f"agent_{i + 1}", cfg=cfg)
        for i in range(num_agents)
    ]
    return env, agents


def _job_a(cfg: DARTS_CONFIG) -> TaskNode:
    dag = DAGGraph.create_target_retrieval_dag(cfg)
    return dag.nodes["A"]


# ---------------------------------------------------------------------------
# Test 1: Agent at task, full battery, empty queue → score ≈ ε ≈ 1 (h=0)
# ---------------------------------------------------------------------------
def test_score_at_task_full_battery_empty_queue():
    cfg = DARTS_CONFIG()
    # Agent 0 at task A location; empty queue
    env, agents = _make_env_and_agents(
        cfg, num_agents=1,
        init_positions=[TASK_A_XY],
    )
    job = _job_a(cfg)
    d_v = job.relative_deadline or cfg.global_deadline_ms
    current_time_ms = 0.0
    # At task: dynamic_wcet ≈ job.wcet (no travel)
    dynamic_wcet_ms = job.wcet
    battery_wh = [cfg.battery_capacities_wh[0]]  # full
    dispatcher = CentralizedDispatcher(cfg, headroom_ms=0.0)  # h=0 so σ/(σ+h)=1 when σ>0
    score = dispatcher.compute_score(
        agents[0], job, current_time_ms,
        battery_wh[0], dynamic_wcet_ms, env,
    )
    # σ = d_v - (current_time + budget) = d_v - (0 + dynamic_wcet + overhead) = d_v - (wcet + 50)
    # With d_v = 180000, wcet = 15000, σ = 164950 > 0, σ/(σ+0) = 1.0, ε ≈ 1.0
    assert score >= 0.99, f"Expected score ≈ 1.0 (full battery, h=0), got {score}"


# ---------------------------------------------------------------------------
# Test 2: Agent far from task, low battery → score ≈ 0
# ---------------------------------------------------------------------------
def test_score_far_from_task_low_battery():
    cfg = DARTS_CONFIG()
    # Agent far from task A (low corner vs B1 A at center)
    env, agents = _make_env_and_agents(
        cfg, num_agents=1,
        init_positions=[FAR_FROM_TASK_A_XY],
    )
    job = _job_a(cfg)
    current_time_ms = 0.0
    # Far from task: estimate_dynamic_wcet_ms returns large (travel + wcet)
    dynamic_wcet_ms = env.estimate_dynamic_wcet_ms(0, TASK_A_XY, job.wcet)
    battery_wh_low = 0.1  # low
    dispatcher = CentralizedDispatcher(cfg, headroom_ms=1000.0)
    score = dispatcher.compute_score(
        agents[0], job, current_time_ms,
        battery_wh_low, dynamic_wcet_ms, env,
    )
    # ε low, σ small (predicted_finish large), σ/(σ+h) small → score near 0
    assert score < 0.2, f"Expected score near 0 (far + low battery), got {score}"


# ---------------------------------------------------------------------------
# Test 3: Two agents — close+low battery vs far+high battery; score trades off
# ---------------------------------------------------------------------------
def test_score_two_agents_tradeoff():
    cfg = DARTS_CONFIG()
    # Agent 0 at task A, agent 1 far from A
    env, agents = _make_env_and_agents(
        cfg, num_agents=2,
        init_positions=[TASK_A_XY, FAR_FROM_TASK_A_XY],
    )
    job = _job_a(cfg)
    current_time_ms = 1000.0
    d_v = job.relative_deadline or cfg.global_deadline_ms
    battery_wh = [0.3, 1.5]  # close=low battery, far=high battery
    dispatcher = CentralizedDispatcher(cfg, headroom_ms=500.0)
    dynamic_wcet_0 = env.estimate_dynamic_wcet_ms(0, TASK_A_XY, job.wcet)
    dynamic_wcet_1 = env.estimate_dynamic_wcet_ms(1, TASK_A_XY, job.wcet)
    score_0 = dispatcher.compute_score(
        agents[0], job, current_time_ms, battery_wh[0], dynamic_wcet_0, env,
    )
    score_1 = dispatcher.compute_score(
        agents[1], job, current_time_ms, battery_wh[1], dynamic_wcet_1, env,
    )
    # Close agent has larger σ (smaller predicted_finish); far agent has smaller σ.
    # Energy-rich distant agent should NOT always win: balance of ε and σ/(σ+h).
    assert score_0 >= 0.0 and score_1 >= 0.0
    # At least one should have non-trivial score; trade-off means we can have either win
    assert score_0 > 0.01 or score_1 > 0.01, "At least one agent should have positive score"


# ---------------------------------------------------------------------------
# Test 4: Loaded queue (σ small) vs empty queue (σ large) → loaded agent lower score
# ---------------------------------------------------------------------------
def test_score_loaded_vs_empty_queue():
    cfg = DARTS_CONFIG()
    # Both agents at task A so same travel; one has queue pre-loaded
    env, agents = _make_env_and_agents(
        cfg, num_agents=2,
        init_positions=[TASK_A_XY, TASK_A_XY],
    )
    job = _job_a(cfg)
    current_time_ms = 0.0
    d_v = job.relative_deadline or cfg.global_deadline_ms
    dynamic_wcet_ms = job.wcet + 100.0  # same for both (at task)
    battery_wh = [1.0, 1.0]

    # Agent 0: pre-load queue with one job (same deadline band) so predicted_finish increases
    from alg.darts_agent import CommittedJob, MAX_TRAVEL_TIME_MS
    delta_max = getattr(cfg, "delta_max_ms", 50.0)
    dummy_job = CommittedJob(
        job_id="dummy",
        task_name="Dummy",
        absolute_deadline_ms=d_v,
        remaining_work_ms=10000.0,
        wcet=10000.0,
        total_budget_ms=10000.0 + MAX_TRAVEL_TIME_MS + delta_max,
    )
    agents[0]._queue.append(dummy_job)
    agents[0]._queue.sort(key=lambda x: (x.absolute_deadline_ms, x.job_id))

    dispatcher = CentralizedDispatcher(cfg, headroom_ms=1000.0)
    score_loaded = dispatcher.compute_score(
        agents[0], job, current_time_ms, battery_wh[0], dynamic_wcet_ms, env,
    )
    score_empty = dispatcher.compute_score(
        agents[1], job, current_time_ms, battery_wh[1], dynamic_wcet_ms, env,
    )
    # Loaded agent: predicted_finish = current + (dummy_c_v) + (new job budget) → larger → σ smaller
    # Empty agent: predicted_finish = current + (new job budget) only → σ larger
    assert score_empty > score_loaded, (
        f"Empty-queue agent should have higher score than loaded; got empty={score_empty}, loaded={score_loaded}"
    )


def test_score_uses_release_relative_deadline_for_delayed_ready_job():
    cfg = DARTS_CONFIG()
    env, agents = _make_env_and_agents(cfg, num_agents=1, init_positions=[TASK_A_XY])
    job = _job_a(cfg)
    job.relative_deadline = 60000.0
    current_time_ms = 100000.0
    release_time_ms = 100000.0
    dispatcher = CentralizedDispatcher(cfg, headroom_ms=1000.0)

    score = dispatcher.compute_score(
        agents[0],
        job,
        current_time_ms,
        cfg.battery_capacities_wh[0],
        job.wcet,
        env,
        release_time_ms=release_time_ms,
    )

    assert score > 0.9, "delayed successors must be scored against tau(v) + relative_deadline"


def test_allocate_accepts_feasible_delayed_ready_job():
    cfg = DARTS_CONFIG()
    env, agents = _make_env_and_agents(cfg, num_agents=1, init_positions=[TASK_A_XY])
    job = _job_a(cfg)
    job.relative_deadline = 60000.0
    current_time_ms = 100000.0
    release_time_ms = 100000.0
    dispatcher = CentralizedDispatcher(cfg, headroom_ms=1000.0)

    winner_idx = dispatcher.allocate(
        job,
        agents,
        current_time_ms,
        [cfg.battery_capacities_wh[0]],
        env,
        [False],
        release_time_ms=release_time_ms,
    )

    assert winner_idx == 0


def test_darts_bid_certificate_uses_release_relative_deadline():
    cfg = DARTS_CONFIG()
    env, _ = _make_env_and_agents(cfg, num_agents=1, init_positions=[TASK_A_XY])
    job = _job_a(cfg)
    job.relative_deadline = 60000.0
    current_time_ms = 100000.0
    release_time_ms = 100000.0
    torch = pytest.importorskip("torch")
    agent = DartsAgent(agent_id="agent_1", cfg=cfg, device=torch.device("cpu"))

    is_feasible, _bid, _log_prob, _budget = agent.compute_bid(
        job,
        current_time_ms,
        current_battery_wh=cfg.battery_capacities_wh[0],
        explore=False,
        env=env,
        release_time_ms=release_time_ms,
    )

    assert is_feasible
