"""
DARTS Deep MARL Agent: Cyber layer (EDF Queue + Feasibility-Masked Bidding) + Physical layer (PyTorch).
LIO bilevel: η (bidding_net) is the only inner policy; φ (incentive_net) outer. Navigation is EDF-driven (no θ).
"""

import contextlib
import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alg.coverage import CoverageStats
    from envs.turtlebot3_spatial_env import TurtleBot3Env
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from envs.dag_task import TaskNode
except ImportError:
    from envs.dag_task import TaskNode

try:
    from envs.workloads import get_workload
except ImportError:
    from workloads import get_workload

try:
    from alg.config_darts import DARTS_CONFIG
except ImportError:
    from config_darts import DARTS_CONFIG


# Worst-case travel budget: 14.14 m diagonal of 10x10 map at 0.22 m/s (~64.3 s) + safety margin
MAX_TRAVEL_TIME_MS = 65000.0


def deadline_cost(t: float, d_v: float) -> float:
    """Eq. 20: DC contribution for one job: max(0, t - d_v)."""
    return max(0.0, t - d_v)


def energy_cost_wh(
    cfg: Any,
    is_moving: bool,
    step_dt_s: float,
    num_messages: int = 0,
    initial_capacity_wh: Optional[float] = None,
) -> float:
    """
    Eq. 21 + Eq. 22: Consumed energy in Wh, then normalized by initial battery (relative depletion).
    If initial_capacity_wh is given and > 0, returns consumed_wh / initial_capacity_wh; else returns consumed_wh.
    """
    dt_h = step_dt_s / 3600.0
    p = cfg.p_moving_w if is_moving else cfg.p_idle_w
    consumed_wh = p * dt_h + num_messages * getattr(cfg, "p_comm_w", 0.5) * dt_h
    if initial_capacity_wh is not None and initial_capacity_wh > 0:
        return consumed_wh / initial_capacity_wh
    return consumed_wh


@dataclass
class CommittedJob:
    """
    A job in the local EDF queue. remaining_work_ms = physical WCET (drained at target only).
    total_budget_ms = wcet + travel + delta_max (safe worst-case for feasibility).
    wcet = task WCET (ms) for Eq. 5 feasibility.
    """
    job_id: str
    task_name: str
    absolute_deadline_ms: float
    remaining_work_ms: float
    wcet: float = 0.0  # task WCET (ms) for Eq. 5 feasibility
    total_budget_ms: float = 0.0  # wcet + MAX_TRAVEL_TIME_MS + delta_max for EDF feasibility

    def __lt__(self, other: "CommittedJob") -> bool:
        if self.absolute_deadline_ms != other.absolute_deadline_ms:
            return self.absolute_deadline_ms < other.absolute_deadline_ms
        return self.job_id < other.job_id


def _feasibility_test(
    sorted_jobs: List[Tuple[str, float, float]],
    current_time_ms: float,
) -> bool:
    """Eq. (11): temporal feasibility (prefix form) iff ∀ℓ: Σ_{j=1..ℓ} c_{v_j} ≤ d_{v_ℓ} − t_k.

    Each ``c_v`` is the **chain-aware** sequential cost for that EDF position per Eq. (10)
    (travel from the previous task's map location, or from the agent for j = 1), not a fixed
    agent-rooted per-job constant.
    """
    assert len({j[0] for j in sorted_jobs}) == len(sorted_jobs), "duplicate job_ids in feasibility test"
    prefix_sum = 0.0
    for _jid, d_v, c_v in sorted_jobs:
        prefix_sum += c_v
        if prefix_sum > d_v - current_time_ms:
            return False
    return True


def _predict_finish_time(
    sorted_jobs: List[Tuple[str, float, float]],
    current_time_ms: float,
    deadline_of_interest: float,
) -> float:
    """Predicted finish time under EDF: t_k + W_i(t_k, d_v).

    Callers may supply agent-rooted ``c_v`` for the job under consideration (e.g. χ^h_i
    slack in ``compute_bid``); that conservative choice is intentional for bidding features
    and is separate from the chain-aware Eq. (10) costs used in ``_feasibility_test``.
    """
    total_work = 0.0
    for _jid, d_v, c_v in sorted_jobs:
        if d_v <= deadline_of_interest:
            total_work += c_v
        else:
            break
    return current_time_ms + total_work


def _edf_density(
    sorted_jobs: List[Tuple[str, float, float]],
    current_time_ms: float,
) -> float:
    """
    EDF density (processor demand): sum of c_v / (d_v - t) over jobs with d_v > t.
    Feasibility sufficient condition: density <= 1.0.
    """
    total = 0.0
    for _jid, d_v, c_v in sorted_jobs:
        slack = d_v - current_time_ms
        if slack <= 0:
            return float("inf")
        total += c_v / slack
    return total


# ---------------------------------------------------------------------------
# Neural networks (η and φ)
# ---------------------------------------------------------------------------

class BiddingNet(nn.Module):
    """
    bidding_net (η): Input χ^h_i = (Q_i, s_{i,v}, cert_{i,v}); dim = queue_state_dim + 2 (slack, cert).
    Output = bid fraction in (0, 1); scaled by slack in compute_bid for physical bid value.
    """
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        assert input_dim >= 1
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns bid fraction μ in (0, 1); scale by slack in compute_bid for actual bid."""
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class IncentiveNet(nn.Module):
    """
    incentive_net (φ): LIO incentive policy. Input = χ^p_s (Eq. 17): bids from C_v + P^max_s
    (num_agents + 1). Output = scalar payment ∈ (0, 1) per Eq. 16.
    """
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 16: payment ∈ (0, 1) (P^max_s = 1, same scale as bids)."""
        return torch.sigmoid(self.fc(x)).squeeze(-1)


def _functional_bidding_forward(x: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
    """
    Same computation as BiddingNet.forward but with explicit params (for LIO bilevel eta_next).
    params: list of 6 tensors [W1, b1, W2, b2, W3, b3]; returns bid mean in (0, 1), shape (batch,) or (batch,).
    """
    x = F.linear(x, params[0], params[1])
    x = F.relu(x)
    x = F.linear(x, params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[4], params[5])
    return torch.sigmoid(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DartsMARLAgent: Cyber + Physical
# ---------------------------------------------------------------------------

class DartsMARLAgent:
    """
    Deep MARL agent: Local EDF Queue (cyber) + Feasibility-Masked Bidding + PyTorch policy/bidding nets (physical).
    Shaped reward follows Eq. 20 (deadline penalty), Eq. 21 (energy cost), Eq. 22 (r̃_i).
    All parameters come from cfg (DARTS_CONFIG or dict with same attributes).
    """

    SPATIAL_STATE_DIM = 6   # default; overridden by spatial_state_dim when provided (e.g. env.single_observation_size)
    QUEUE_STATE_DIM = 2    # queue_length_norm, total_remaining_work_norm
    TASK_FEATURE_DIM = 3   # wcet_norm, deadline_norm, task_id_onehot or similar
    BID_SIGMA = 0.1        # Gaussian noise std for continuous bid exploration

    def __init__(
        self,
        agent_id: str,
        cfg: Optional[Any] = None,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
        spatial_state_dim: Optional[int] = None,
        exec_action_space_size: Optional[int] = None,
    ):
        """
        Initialize DARTS agent i with per-agent trainable parameters:
          η_i (self.bidding_net)   — Bidding policy, Eq. 14
          φ_i (self.incentive_net) — Incentive policy, Eq. 16
          ψ_i (self.psi_logits)   — Graph inference adjacency logits, Eq. 2
        Each agent i owns independent parameter copies (no sharing across agents).
        """
        self.agent_id = agent_id
        self.cfg = cfg if cfg is not None else DARTS_CONFIG()
        self._log = logger or logging.getLogger(f"darts_marl_agent.{agent_id}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._spatial_state_dim = spatial_state_dim if spatial_state_dim is not None else self.SPATIAL_STATE_DIM

        self._rng = random.Random(getattr(cfg, "random_seed", None) or seed)
        # EDF job queue: list of CommittedJob, kept sorted by (absolute_deadline_ms, job_id); head = active job
        self._queue: List[CommittedJob] = []

        # Bidding net (η): input χ^h = queue_state + slack + cert + inferred_pred + dist_norm (queue_state_dim + 4).
        self.bidding_net = BiddingNet(self.QUEUE_STATE_DIM + 4, hidden=64).to(self.device)
        self.optimizer_eta = torch.optim.Adam(
            self.bidding_net.parameters(), lr=getattr(cfg, "lr_eta", 3e-4)
        )
        # Eq. 16–17: incentive input χ^p_s = {bid_{j,v}}_{j in C_v} + P^max_s (num_agents bids + 1)
        incentive_input_dim = getattr(cfg, "num_agents", 4) + 1
        self.incentive_net = IncentiveNet(incentive_input_dim).to(self.device)
        self.optimizer_phi = torch.optim.Adam(
            self.incentive_net.parameters(), lr=getattr(cfg, "lr_phi", 1e-4)
        )

        workload_id = getattr(self.cfg, "workload_id", "B1_CDR")
        try:
            wl = get_workload(workload_id)
            self._graph_node_ids = tuple(wl.node_ids)
        except Exception:
            self._graph_node_ids = ("A", "B", "C", "D")
        self._graph_node_index = {nid: idx for idx, nid in enumerate(self._graph_node_ids)}

        # Graph inference module g_ψ: learnable adjacency logits (n tasks from workload graph).
        n_graph = max(1, len(self._graph_node_ids))
        self.psi_logits = nn.Parameter(torch.zeros(n_graph, n_graph, device=self.device))
        self.optimizer_psi = torch.optim.Adam([self.psi_logits], lr=getattr(cfg, "lr_psi", 1e-3))

        self.trajectory: List[Tuple[Any, ...]] = []  # on-policy episode data for REINFORCE (sequential)
        self.incentive_trajectory: List[torch.Tensor] = []  # incentive outputs for Eq. 25 (Phase B)
        # (payment, job_id, sponsor_idx) for Eq. (36) per-sponsored-job aggregation
        self.payments_phase_a: List[Tuple[float, str, int]] = []
        self.dag_tracker: Optional[Any] = None  # LocalDAGTracker(dag) for distributed causality; set by trainer
        self._return_baseline = 0.0  # EMA baseline for REINFORCE variance reduction

    @property
    def queue(self) -> List[CommittedJob]:
        return self._queue.copy()

    def _queue_state_vector(self) -> np.ndarray:
        """Queue state: [queue_length_norm, total_remaining_work_norm]."""
        n = len(self._queue)
        total_work = sum(j.remaining_work_ms for j in self._queue)
        max_q = getattr(self.cfg, "max_queue_size", 10)
        max_w = getattr(self.cfg, "max_work_ms", 2000.0)
        return np.array([
            n / max(1, max_q),
            min(1.0, total_work / max_w),
        ], dtype=np.float32)

    def _full_state(self, spatial_state: np.ndarray) -> np.ndarray:
        """Concatenate spatial (env.single_observation_size) + queue state (2) for state representation."""
        q = self._queue_state_vector()
        return np.concatenate([spatial_state, q], axis=-1).astype(np.float32)

    def _get_queue_state(self) -> np.ndarray:
        """Queue feature Q_i for χ^h_i (same as _queue_state_vector)."""
        return self._queue_state_vector()

    def reset_queue(self) -> None:
        self._queue.clear()

    def reset(self) -> None:
        """Clear agent state for a new episode (EDF queue, trajectory, incentive_trajectory, DAG tracker). payments_phase_a cleared in train loop after φ update."""
        self._queue.clear()
        self.trajectory.clear()
        self.incentive_trajectory.clear()
        if self.dag_tracker is not None and hasattr(self.dag_tracker, "reset"):
            self.dag_tracker.reset()

    def update_graph(self, verified_adj_matrix: torch.Tensor) -> float:
        """
        Update graph inference module g_ψ by minimizing cross-entropy consistency loss L_graph(ψ; τ_k)
        between neural-proposed dependencies (psi_logits) and physically verified Done sets (verified_adj_matrix).
        """
        target = verified_adj_matrix.to(device=self.device, dtype=torch.float32)
        if target.shape != self.psi_logits.shape:
            raise ValueError(f"verified_adj_matrix shape {target.shape} must match psi_logits {self.psi_logits.shape}")
        loss_graph = F.binary_cross_entropy_with_logits(self.psi_logits, target)
        self.optimizer_psi.zero_grad()
        loss_graph.backward()
        self.optimizer_psi.step()
        return loss_graph.item()

    def store_bid_event(self, log_prob: torch.Tensor) -> None:
        """Append the epoch's bid log_prob to trajectory (asynchronous bidding events only)."""
        self.trajectory.append(log_prob)

    def store_incentive_event(self, log_prob: torch.Tensor) -> None:
        """Append incentive net output (for Eq. 25 J^inc) to incentive_trajectory (Phase B)."""
        self.incentive_trajectory.append(log_prob)

    def store_payment_phase_a(self, payment: float, job_id: str, sponsor_idx: int) -> None:
        """Record payment from Phase A for penalty term in loss_phi (Eq. 25 / Eq. 36)."""
        self.payments_phase_a.append((float(payment), str(job_id), int(sponsor_idx)))

    def update_parameters(
        self,
        episode_return: Optional[float] = None,
        coverage_stats: Optional["CoverageStats"] = None,
    ) -> Optional[Dict[str, float]]:
        """
        REINFORCE on episodic bid events: trajectory is list of log_prob from store_bid_event.
        Uses advantage-style loss; with only episode return, baseline=0 keeps same gradient as original.
        When per-bid returns exist, use their mean as baseline for variance reduction.

        Eq. (35): REINFORCE target includes ``lambda_rho * rho_hat`` coverage term when
        ``coverage_stats`` is provided.
        """
        if len(self.trajectory) == 0:
            return None
        log_probs = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device) for t in self.trajectory])
        returns_base = float(episode_return) if episode_return is not None else 0.0
        lambda_rho = getattr(self.cfg, "lambda_rho", 1.0)
        rho_hat = float(coverage_stats.rho_hat) if coverage_stats is not None else 0.0
        adjusted_return = returns_base + lambda_rho * rho_hat
        # Update running baseline (EMA with alpha=0.05 for smooth tracking)
        ema_alpha = 0.05
        self._return_baseline = (1.0 - ema_alpha) * self._return_baseline + ema_alpha * adjusted_return
        # Advantage = adjusted return - baseline (centered, lower variance)
        n_bids = log_probs.shape[0]
        advantages = torch.full(
            (n_bids,),
            adjusted_return - self._return_baseline,
            device=self.device,
            dtype=log_probs.dtype,
        )
        loss_eta = -(log_probs * advantages).sum()
        self.optimizer_eta.zero_grad()
        lrho_scalar = lambda_rho * rho_hat
        grad_eta_rho_norm = 0.0
        if coverage_stats is not None and abs(lrho_scalar) > 1e-18:
            lrho_t = torch.as_tensor(lrho_scalar, device=log_probs.device, dtype=log_probs.dtype)
            L_cov_eta = -(log_probs.sum() * lrho_t)
            grads_cov = torch.autograd.grad(
                L_cov_eta,
                self.bidding_net.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            grad_eta_rho_norm = math.sqrt(
                sum(float(g.detach().float().pow(2).sum().item()) for g in grads_cov if g is not None)
            )
        loss_eta.backward()
        clip_norm = getattr(self.cfg, "gradient_clip_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(self.bidding_net.parameters(), max_norm=clip_norm)
        self.optimizer_eta.step()
        self.trajectory.clear()
        return {
            "bid_loss": float(loss_eta.detach().item()),
            "grad_eta_rho_norm": float(grad_eta_rho_norm),
        }

    # ---------- Local EDF Queue and Feasibility-Masked Bidding (unchanged logic) ----------

    def _sorted_queue_for_math(
        self,
        current_time_ms: float,
        env: Optional["TurtleBot3Env"] = None,
    ) -> List[Tuple[str, float, float]]:
        """
        Return queue as (job_id, absolute_deadline_ms, c_v) with c_v the chain-aware sequential
        cost per Eq. (10): travel from the EDF predecessor task location (j >= 2) or the agent (j == 1),
        plus remaining work, plus ``delta_max`` (network overhead).
        """
        delta_max_ms = getattr(self.cfg, "delta_max_ms", 50.0)
        agent_idx = self._agent_index()
        sorted_jobs = sorted(self._queue, key=lambda x: (x.absolute_deadline_ms, x.job_id))
        result: List[Tuple[str, float, float]] = []
        prev_xy: Optional[Tuple[float, float]] = None  # j == 1 marker
        for j in sorted_jobs:
            if env is not None and getattr(env, "task_locations", None) is not None and j.job_id in env.task_locations:
                target_xy = env.task_locations[j.job_id]
                chain_cost, _ = env.estimate_chain_wcet_ms(
                    agent_idx, prev_xy, target_xy, j.remaining_work_ms,
                )
                remaining_c = chain_cost + delta_max_ms
                prev_xy = target_xy  # j >= 2 onward roots at this task's position
            else:
                remaining_c = j.remaining_work_ms + 25000.0 + delta_max_ms
                prev_xy = None  # unknown location: reset chain
            result.append((j.job_id, j.absolute_deadline_ms, remaining_c))
        return result

    def receive_task_broadcast(self, sender: str, task: TaskNode, release_time_ms: float) -> None:
        self._log.debug("Agent %s received task broadcast: task_id=%s", self.agent_id, task.node_id)

    def _agent_index(self) -> int:
        """Zero-based index for env.estimate_dynamic_wcet_ms and get_agent_position (agent_1 -> 0, agent_2 -> 1, ...)."""
        try:
            return max(0, int(self.agent_id.split("_")[1]) - 1)
        except (IndexError, ValueError):
            return 0

    def compute_bid(
        self,
        new_task: TaskNode,
        current_time_ms: float,
        current_battery_wh: float,
        explore: bool = True,
        env: Optional["TurtleBot3Env"] = None,
        release_time_ms: float = 0.0,
    ) -> Tuple[bool, float, torch.Tensor, float]:
        """
        Epoch-level bid generation: bid_{i,v} ~ π^η_i(·|χ^h_i). χ^h_i = (Q_i, s_{i,v}, cert_{i,v}).
        Returns (is_feasible, final_bid, log_prob, dynamic_budget_ms) for use by assign_job.
        """
        overhead_ms = getattr(self.cfg, "delta_max_ms", 50.0)
        # Dynamic WCET: optimistic travel = actual Euclidean distance / max_velocity_mps (env.estimate_dynamic_wcet_ms)
        # when env and task location available; avoids premature Eq. 23 fallback from blanket MAX_TRAVEL_TIME_MS
        if env is not None and getattr(env, "task_locations", None) is not None and new_task.node_id in env.task_locations:
            target_xy = env.task_locations[new_task.node_id]
            dynamic_wcet = env.estimate_dynamic_wcet_ms(
                self._agent_index(), target_xy, new_task.wcet
            )
        else:
            dynamic_wcet = new_task.wcet + MAX_TRAVEL_TIME_MS
        dynamic_budget_ms = dynamic_wcet + overhead_ms
        # Normalized distance-to-task for spatial bid differentiation (6th feature of χ^h_i)
        dist_norm = 0.5
        if env is not None and getattr(env, "task_locations", None) is not None and new_task.node_id in env.task_locations:
            target_xy = env.task_locations[new_task.node_id]
            agent_pos = env.get_agent_position(self._agent_index()) if hasattr(env, "get_agent_position") else None
            if agent_pos is not None:
                ax, ay = agent_pos[0], agent_pos[1]
                tx, ty = target_xy[0], target_xy[1]
                dist = ((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5
                map_diag = getattr(env, "map_size", 10.0) * (2.0 ** 0.5)
                dist_norm = min(1.0, dist / map_diag) if map_diag > 0 else 0.5
        # Ensure dist_norm is always in [0, 1]
        dist_norm = max(0.0, min(1.0, dist_norm))
        d_v = release_time_ms + (
            new_task.relative_deadline or getattr(self.cfg, "global_deadline_ms", 120000.0)
        )

        # Certificate: cert = 1.0 if EDF temporal + energy feasible, else 0.0
        if getattr(self.cfg, "disable_feasibility_mask", False):
            # ABLATION: DARTS-NoCert — skip schedulability check; force cert mask on for bidding.
            is_feasible = True
            cert = 1.0
        else:
            is_feasible = self.check_feasibility(
                new_task, current_time_ms, current_battery_wh,
                dynamic_wcet_ms=dynamic_wcet,
                release_time_ms=release_time_ms,
                env=env,
            )
            cert = 1.0 if is_feasible else 0.0

        # Queue-aware slack per Eq. 8-9: s_{i,v} = d_v - PredictFinish_i(v, Q_i(t_k)).
        # Queue entries use chain-aware c_v from _sorted_queue_for_math; the hypothetical new
        # task uses agent-rooted dynamic_budget_ms so slack stays conservative (see _predict_finish_time).
        test_queue = self._sorted_queue_for_math(current_time_ms, env=env)
        test_queue.append((new_task.node_id, d_v, dynamic_budget_ms))
        test_queue.sort(key=lambda x: (x[1], x[0]))
        predicted_finish = _predict_finish_time(test_queue, current_time_ms, d_v)
        slack = d_v - predicted_finish
        slack_norm = max(0.0, slack) / max(d_v, 1.0)

        # Inferred graph: encode learned predecessor completion as soft feature (not hard gate)
        # predsafe gates readiness via global_tracker (Eq. 3); predlearn_psi is informational only
        inferred_pred_score = 1.0  # 1.0 = all inferred predecessors satisfied
        if new_task.node_id in self._graph_node_index:
            v_idx = self._graph_node_index[new_task.node_id]
            with torch.no_grad():
                probs = torch.sigmoid(self.psi_logits[:, v_idx])
            completed: set = set()
            if env is not None and hasattr(env, "completed_tasks_ms"):
                completed = set(env.completed_tasks_ms.keys())
            elif self.dag_tracker is not None and hasattr(self.dag_tracker, "node_states"):
                ns = self.dag_tracker.node_states
                completed = {nid for nid, s in ns.items() if s == "COMPLETED"}
            n_unsatisfied = 0
            for u in range(len(self._graph_node_ids)):
                if probs[u].item() > 0.5 and self._graph_node_ids[u] not in completed:
                    n_unsatisfied += 1
            inferred_pred_score = 1.0 - (n_unsatisfied / float(max(1, len(self._graph_node_ids))))

        # χ^h_i = (Q_i, s_{i,v}, cert_{i,v}, inferred_pred, dist_norm); all features in [0,1] for balanced BiddingNet input
        q_feat = self._get_queue_state()
        chi_h = torch.tensor(
            list(q_feat) + [slack_norm, cert, inferred_pred_score, dist_norm],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # μ = bidding_net(χ^h) ∈ [0, 1]
        mu = self.bidding_net(chi_h).squeeze(-1)
        if explore:
            # Beta distribution: proper support on (0,1), no clamp boundary bias.
            # Concentrate parameter controls exploration spread (higher = less noise).
            concentration = getattr(self.cfg, "bid_concentration", 10.0)
            alpha_param = mu.clamp(1e-3, 1.0 - 1e-3) * concentration
            beta_param = (1.0 - mu.clamp(1e-3, 1.0 - 1e-3)) * concentration
            bid_dist = torch.distributions.Beta(alpha_param, beta_param)
            bid_frac = bid_dist.sample()
            log_prob = bid_dist.log_prob(bid_frac).squeeze()
        else:
            bid_frac = mu.clamp(0.0, 1.0)
            log_prob = torch.tensor(0.0, device=self.device, dtype=mu.dtype)

        # Eq. 14 with Physical Prior: spatial discount (1 - dist_norm) scales bid by proximity to task.
        # dist_norm remains part of χ^h so the network can learn how distance affects its effective bid.
        final_bid = bid_frac.item() * (1.0 - dist_norm)
        return (is_feasible, float(final_bid), log_prob, dynamic_budget_ms)

    def assign_job(
        self,
        job: TaskNode,
        release_time_ms: float = 0.0,
        current_time_ms: float = 0.0,
        bid: Optional[float] = None,
        dynamic_budget_ms: Optional[float] = None,
        override_remaining_work_ms: Optional[float] = None,
    ) -> None:
        """Assign a DAG job to this agent (auction winner). Appends job to the EDF queue and sorts by deadline."""
        self.commit_task(job, release_time_ms, current_time_ms, bid=bid, dynamic_budget_ms=dynamic_budget_ms, override_remaining_work_ms=override_remaining_work_ms)

    def commit_task(
        self,
        task: TaskNode,
        release_time_ms: float,
        current_time_ms: float,
        bid: Optional[float] = None,
        dynamic_budget_ms: Optional[float] = None,
        override_remaining_work_ms: Optional[float] = None,
    ) -> None:
        d_v = release_time_ms + (
            task.relative_deadline or getattr(self.cfg, "global_deadline_ms", 120000.0)
        )
        delta_max = getattr(self.cfg, "delta_max_ms", 50.0)
        if dynamic_budget_ms is not None:
            total_c = dynamic_budget_ms
        else:
            total_c = task.wcet + MAX_TRAVEL_TIME_MS + delta_max
        job = CommittedJob(
            job_id=task.node_id,
            task_name=task.name,
            absolute_deadline_ms=d_v,
            remaining_work_ms=override_remaining_work_ms if override_remaining_work_ms is not None else task.wcet * np.random.uniform(0.7, 1.0),
            wcet=task.wcet,
            total_budget_ms=total_c,
        )
        self._queue.append(job)
        self._queue.sort(key=lambda x: (x.absolute_deadline_ms, x.job_id))
        self._log.info("Agent %s committed Task %s (d_v=%.1f, total_budget=%.1f); queue size=%d.", self.agent_id, task.node_id, d_v, total_c, len(self._queue))

    def get_wc_energy_wh(self, job_wcet_ms: float) -> float:
        """Worst-case energy (Wh) for one job: (wcet + travel + comm) time at moving power (Eq. 47)."""
        wcet_total_ms = (
            job_wcet_ms
            + 25000.0  # average point-to-point travel time on 10x10 map at 0.22 m/s
            + getattr(self.cfg, "delta_max_ms", 50.0)
        )
        hours = wcet_total_ms / 3600000.0
        return hours * getattr(self.cfg, "p_moving_w", 7.5)

    def check_feasibility(
        self,
        new_job: TaskNode,
        current_time_ms: float,
        current_battery_wh: float = 1.0,
        dynamic_wcet_ms: Optional[float] = None,
        release_time_ms: float = 0.0,
        env: Optional["TurtleBot3Env"] = None,
    ) -> bool:
        """
        Joint certificate (Eq. 48): temporal (Eq. 11 with Eq. 10 chain costs) and energy (Eq. 47) feasibility.
        Returns True only if both time_feasible and energy_feasible.
        If dynamic_wcet_ms is provided, use it for the new job instead of wcet + max_travel.
        If env is provided, _sorted_queue_for_math uses chain-aware costs for queued jobs.
        d_v = release_time_ms + relative_deadline (consistent with commit_task and absolute_deadline_ms).
        """
        if getattr(self.cfg, "disable_feasibility_mask", False):
            # ABLATION: DARTS-NoCert — skip EDF prefix-sum and energy certificate.
            return True

        overhead_ms = getattr(self.cfg, "delta_max_ms", 50.0)

        # New job budget: dynamic_wcet (if provided) + overhead, else WCET + average travel heuristic + overhead.
        if dynamic_wcet_ms is not None:
            new_job_budget = dynamic_wcet_ms + overhead_ms
        else:
            new_job_budget = new_job.wcet + 25000.0 + overhead_ms

        # Post-admission queue with chain recomposition (Lemma 2 of paper)
        dyn_budget_new = (
            (dynamic_wcet_ms if dynamic_wcet_ms is not None else new_job.wcet + 25000.0) + overhead_ms
        )
        d_v = release_time_ms + (
            new_job.relative_deadline or getattr(self.cfg, "global_deadline_ms", 120000.0)
        )

        # Build the union queue with new_job inserted, then resort by deadline,
        # then recompute chain costs against EDF-predecessor positions.
        union_jobs = list(self._queue) + [
            CommittedJob(
                job_id=new_job.node_id,
                task_name=new_job.name,
                absolute_deadline_ms=d_v,
                remaining_work_ms=new_job.wcet,
                wcet=new_job.wcet,
                total_budget_ms=dyn_budget_new,
            )
        ]
        union_jobs.sort(key=lambda x: (x.absolute_deadline_ms, x.job_id))

        agent_idx = self._agent_index()
        prev_xy: Optional[Tuple[float, float]] = None
        test_queue: List[Tuple[str, float, float]] = []
        for j in union_jobs:
            if env is not None and getattr(env, "task_locations", None) is not None and j.job_id in env.task_locations:
                target_xy = env.task_locations[j.job_id]
                chain_cost, _ = env.estimate_chain_wcet_ms(
                    agent_idx, prev_xy, target_xy, j.remaining_work_ms,
                )
                c_v = chain_cost + overhead_ms
                prev_xy = target_xy
            else:
                c_v = j.remaining_work_ms + 25000.0 + overhead_ms
                prev_xy = None
            test_queue.append((j.job_id, j.absolute_deadline_ms, c_v))
        time_feasible = _feasibility_test(test_queue, current_time_ms)

        if dynamic_wcet_ms is not None:
            new_job_energy_wh = (
                (dynamic_wcet_ms + overhead_ms) / 3600000.0
                * getattr(self.cfg, "p_moving_w", 7.5)
            )
        else:
            new_job_energy_wh = self.get_wc_energy_wh(new_job.wcet)
        total_required_energy_wh = sum(
            self.get_wc_energy_wh(j.wcet) for j in self._queue
        ) + new_job_energy_wh
        e_safe = getattr(self.cfg, "e_safe_wh", 0.05)
        energy_feasible = current_battery_wh >= (e_safe + total_required_energy_wh)

        return time_feasible and energy_feasible

    def get_active_job(self) -> Optional[CommittedJob]:
        """Return the EDF head (queue[0]) if the queue is not empty, else None. Queue is kept sorted on assign."""
        if not self._queue:
            return None
        return self._queue[0]

    def execute_task(self, current_time_ms: float, delta_ms: float = 1.0) -> Optional[CommittedJob]:
        """Drain WCET for the active job (only decrement remaining_work_ms). Does not remove the job."""
        active = self.get_active_job()
        if active is None:
            return None
        active.remaining_work_ms -= delta_ms
        return active

    def complete_active_job(self) -> Optional[CommittedJob]:
        """Remove and return the EDF head (queue[0]). Call only when the active job has finished (WCET drained)."""
        if not self._queue:
            return None
        completed = self._queue.pop(0)
        self._log.debug("Agent %s completed job %s.", self.agent_id, completed.job_id)
        return completed

    @contextlib.contextmanager
    def temporary_bidding_weights(self, alt_state_dict):
        """Temporarily replace self.bidding_net weights with alt_state_dict for the
        duration of the context; restore the at-entry weights on exit.

        Used by Path 1 (--bilevel_mode fixed_eta) to roll Trajectory B under η^m
        while the at-entry value of self.bidding_net is η̂^m, restoring η̂^m on exit
        so the next outer iteration sees the post-update weights as canonical.
        """
        at_entry = copy.deepcopy(self.bidding_net.state_dict())
        self.bidding_net.load_state_dict(alt_state_dict)
        try:
            yield
        finally:
            self.bidding_net.load_state_dict(at_entry)

    def freeze_incentive_at_zero(self):
        """Zero every parameter of self.incentive_net and set requires_grad=False.

        Used by the MC-7 NoIncentive ablation (--no_incentive). After this call,
        the incentive forward pass is still executed during bidding (to keep the
        pipeline shape-identical) but returns 0 by construction since all ϕ params
        are zero. The ϕ update step in train_darts.py is additionally skipped under
        --no_incentive; this freeze is the defense-in-depth layer.
        """
        with torch.no_grad():
            for p in self.incentive_net.parameters():
                p.zero_()
                p.requires_grad_(False)
        # Sanity assertion — fail loudly if the freeze did not take effect.
        for p in self.incentive_net.parameters():
            assert torch.all(p == 0).item(), (
                f"Agent {self.agent_id}: incentive_net parameter is non-zero after freeze"
            )
            assert not p.requires_grad, (
                f"Agent {self.agent_id}: incentive_net parameter has requires_grad=True after freeze"
            )

    def save_checkpoint(self, path: str) -> None:
        """Save bidding_net (η), incentive_net (φ), and psi_logits (ψ) for checkpointing.
        incentive_input_dim is stored so load_checkpoint can detect incompatible older checkpoints
        (e.g. when incentive net was state-based instead of bid-vector χ^p_s)."""
        torch.save({
            "bidding_net": self.bidding_net.state_dict(),
            "incentive_net": self.incentive_net.state_dict(),
            "incentive_input_dim": self.incentive_net.fc[0].weight.shape[1],
            "psi_logits": self.psi_logits.data,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load bidding_net, incentive_net, and psi_logits from checkpoint.
        If saved incentive_input_dim does not match current net, skip loading incentive_net (keep init)."""
        ckpt = torch.load(path, map_location=self.device)
        self.bidding_net.load_state_dict(ckpt["bidding_net"])
        if "incentive_net" in ckpt:
            current_dim = self.incentive_net.fc[0].weight.shape[1]
            saved_dim = ckpt.get("incentive_input_dim")
            if saved_dim is not None and saved_dim == current_dim:
                self.incentive_net.load_state_dict(ckpt["incentive_net"])
            elif saved_dim is not None:
                self._log.warning(
                    "Checkpoint incentive_net input_dim=%s != current %s; skipping incentive_net load.",
                    saved_dim, current_dim,
                )
            else:
                self._log.warning(
                    "Checkpoint has no incentive_input_dim (old format); skipping incentive_net load.",
                )
        if "psi_logits" in ckpt:
            saved_psi = ckpt["psi_logits"]
            if saved_psi.shape == self.psi_logits.shape:
                self.psi_logits.data.copy_(saved_psi)
            else:
                self._log.warning(
                    "Checkpoint psi_logits shape %s != current %s; skipping psi load.",
                    tuple(saved_psi.shape),
                    tuple(self.psi_logits.shape),
                )

    def update_policy(self, trajectory: List[Dict[str, Any]]) -> None:
        """Outer-loop (Algorithm 2): one on-policy update over full episode trajectory."""
        loss_dict = self.update_parameters()
        if loss_dict:
            self._log.debug("Agent %s update_parameters: %s", self.agent_id, loss_dict)


# ---------------------------------------------------------------------------
# Backward compatibility: DartsAgent as thin wrapper
# ---------------------------------------------------------------------------

class DartsAgent(DartsMARLAgent):
    """
    Alias for DartsMARLAgent for code that expects the original interface.
    Alias for DartsMARLAgent (compute_bid/store_bid_event/update_parameters at epoch-level).
    """

    def update_policy(self, trajectory: List[Dict[str, Any]]) -> None:
        if len(self.trajectory) > 0:
            DartsMARLAgent.update_policy(self, trajectory)
        else:
            self._log.info(
                "Agent %s: update_policy called (trajectory length=%d).",
                self.agent_id, len(self.trajectory),
            )
