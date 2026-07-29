"""
SADCHER benchmark agent (Bichler et al., IEEE MRS 2025).
MARL-only task allocation with GAT + Transformer; NO real-time guarantees
(no EDF, no feasibility certificates, no WCET budgets, no energy-aware masking).
"""

from __future__ import division, print_function

import logging
from typing import Any, Dict, List, Optional, Tuple

# Add project root for imports when run from any directory
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from alg.darts_agent import CommittedJob, MAX_TRAVEL_TIME_MS
from alg.config_sadcher import SADCHER_CONFIG


# ---------------------------------------------------------------------------
# 1. RobotEncoder
# ---------------------------------------------------------------------------
class RobotEncoder(nn.Module):
    """Encode robot graph with all-to-all GAT (paper Sec. IV-A). Input [B, N, robot_feature_dim] → [B, N, latent_dim]."""

    def __init__(self, cfg: SADCHER_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.robot_feature_dim, cfg.latent_dim)
        self.layers = nn.ModuleList()
        for _ in range(cfg.num_gat_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.latent_dim,
                    num_heads=cfg.num_gat_heads,
                    dropout=0.0,
                    batch_first=True,
                )
            )
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.latent_dim) for _ in range(cfg.num_gat_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, robot_feature_dim]
        x = self.proj(x)  # [B, N, latent_dim]
        for layer, norm in zip(self.layers, self.norms):
            attn_out, _ = layer(x, x, x)  # all-to-all
            x = norm(x + attn_out)
        return x


# ---------------------------------------------------------------------------
# 2. TaskEncoder
# ---------------------------------------------------------------------------
def _build_dag_attention_mask(
    node_ids: Tuple[str, ...], predecessors: Dict[str, List[str]]
) -> torch.Tensor:
    """Build [M, M] DAG attention mask from a workload's node_ids and predecessor map.
    True = do not attend. Node i attends to its predecessors and itself.
    """
    M = len(node_ids)
    mask = torch.ones(M, M, dtype=torch.bool)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    for i, nid in enumerate(node_ids):
        allowed = {idx[p] for p in predecessors.get(nid, [])} | {i}
        for j in range(M):
            if j in allowed:
                mask[i, j] = False
    return mask


class TaskEncoder(nn.Module):
    """Encode task graph with DAG-masked GAT. Input [B, M, task_feature_dim] → [B, M, latent_dim]."""

    def __init__(
        self,
        cfg: SADCHER_CONFIG,
        node_ids: Tuple[str, ...],
        predecessors: Dict[str, List[str]],
    ):
        super().__init__()
        self.cfg = cfg
        assert len(node_ids) == 4, (
            f"SADCHER baseline architecture is registered with M=4 fixed. "
            f"Got len(node_ids)={len(node_ids)} for node_ids={node_ids}. "
            f"This is a deliberate baseline limitation; SADCHER's IL training "
            f"paradigm assumes a fixed task-set size."
        )
        self.M = 4
        self.node_ids = tuple(node_ids)
        self.proj = nn.Linear(cfg.task_feature_dim, cfg.latent_dim)
        self.layers = nn.ModuleList()
        for _ in range(cfg.num_gat_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.latent_dim,
                    num_heads=cfg.num_gat_heads,
                    dropout=0.0,
                    batch_first=True,
                )
            )
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.latent_dim) for _ in range(cfg.num_gat_layers)])
        self.register_buffer(
            "_dag_mask",
            _build_dag_attention_mask(node_ids, predecessors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, task_feature_dim]
        x = self.proj(x)  # [B, M, latent_dim]
        # MultiheadAttention: attn_mask (M, M) True = don't attend (applied to all batches/heads)
        for layer, norm in zip(self.layers, self.norms):
            attn_out, _ = layer(x, x, x, attn_mask=self._dag_mask)
            x = norm(x + attn_out)
        return x


# ---------------------------------------------------------------------------
# 3. DistanceMLP
# ---------------------------------------------------------------------------
class DistanceMLP(nn.Module):
    """Map pairwise distances [B, N, M] → [B, N, M, latent_dim] via two-layer MLP with LeakyReLU."""

    def __init__(self, cfg: SADCHER_CONFIG):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, cfg.distance_mlp_hidden),
            nn.LeakyReLU(),
            nn.Linear(cfg.distance_mlp_hidden, cfg.latent_dim),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: [B, N, M]
        B, N, M = distances.shape
        d = distances.unsqueeze(-1)  # [B, N, M, 1]
        out = self.mlp(d)  # [B, N, M, latent_dim]
        return out


# ---------------------------------------------------------------------------
# 4. CrossAttentionTransformer
# ---------------------------------------------------------------------------
class CrossAttentionTransformer(nn.Module):
    """Robot embeddings attend to task embeddings over num_transformer_layers. Output [B, N, latent_dim]."""

    def __init__(self, cfg: SADCHER_CONFIG):
        super().__init__()
        self.cfg = cfg
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.num_gat_heads,
            dim_feedforward=cfg.latent_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_transformer_layers)

    def forward(self, robot_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        # robot_emb [B, N, d], task_emb [B, M, d]; decoder(tgt, memory) with tgt=robot, memory=task
        return self.decoder(robot_emb, task_emb)  # [B, N, latent_dim]


# ---------------------------------------------------------------------------
# 5. RewardMLP + IdleMLP → R matrix [B, N, M+1]
# ---------------------------------------------------------------------------
class RewardMLP(nn.Module):
    """Concatenate [robot_embed_i, task_embed_j, distance_embed_ij] → scalar; plus IdleMLP for column M."""

    def __init__(self, cfg: SADCHER_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * cfg.latent_dim, cfg.reward_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.reward_mlp_hidden, 1),
        )
        self.idle_mlp = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.reward_mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.reward_mlp_hidden, 1),
        )

    def forward(
        self,
        robot_emb: torch.Tensor,
        task_emb: torch.Tensor,
        distance_emb: torch.Tensor,
    ) -> torch.Tensor:
        # robot_emb [B, N, d], task_emb [B, M, d], distance_emb [B, N, M, d]
        B, N, M = distance_emb.shape[:3]
        robot_exp = robot_emb.unsqueeze(2).expand(-1, -1, M, -1)   # [B, N, M, d]
        task_exp = task_emb.unsqueeze(1).expand(-1, N, -1, -1)    # [B, N, M, d]
        pair_feat = torch.cat([robot_exp, task_exp, distance_emb], dim=-1)  # [B, N, M, 3*d]
        R_pairs = self.pair_mlp(pair_feat).squeeze(-1)  # [B, N, M]
        R_idle = self.idle_mlp(robot_emb).squeeze(-1)  # [B, N]
        R = torch.cat([R_pairs, R_idle.unsqueeze(-1)], dim=-1)  # [B, N, M+1]
        return R


# ---------------------------------------------------------------------------
# 6. SadcherRewardNet
# ---------------------------------------------------------------------------
class SadcherRewardNet(nn.Module):
    """Full SADCHER forward: robot_features, task_features, task_adjacency, distances → R [B, N, M+1], V [B]."""

    def __init__(
        self,
        cfg: SADCHER_CONFIG,
        node_ids: Tuple[str, ...],
        predecessors: Dict[str, List[str]],
    ):
        super().__init__()
        self.cfg = cfg
        self.robot_encoder = RobotEncoder(cfg)
        self.task_encoder = TaskEncoder(cfg, node_ids, predecessors)
        self.distance_mlp = DistanceMLP(cfg)
        self.cross_attn = CrossAttentionTransformer(cfg)
        self.reward_mlp = RewardMLP(cfg)
        self.critic_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(),
            nn.Linear(cfg.latent_dim, 1),
        )

    def forward(
        self,
        robot_features: torch.Tensor,
        task_features: torch.Tensor,
        task_adjacency: Optional[torch.Tensor],
        distances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # task_adjacency unused here (TaskEncoder uses built-in DAG mask)
        robot_emb = self.robot_encoder(robot_features)       # [B, N, d]
        task_emb = self.task_encoder(task_features)          # [B, M, d]
        distance_emb = self.distance_mlp(distances)          # [B, N, M, d]
        robot_cross = self.cross_attn(robot_emb, task_emb)  # [B, N, d]
        R = self.reward_mlp(robot_cross, task_emb, distance_emb)  # [B, N, M+1]
        # V(s): mean over robot embeddings then critic
        global_emb = robot_cross.mean(dim=1)  # [B, d]
        V = self.critic_head(global_emb).squeeze(-1)  # [B]
        return R, V


# ---------------------------------------------------------------------------
# 7. RelaxedBipartiteMatching (function)
# ---------------------------------------------------------------------------
def relaxed_bipartite_matching(
    R: torch.Tensor,
    task_ready_mask: torch.Tensor,
    robot_idle_mask: torch.Tensor,
    task_order: Tuple[str, ...],
) -> Dict[int, str]:
    """
    Greedy assignment: only idle robots and ready tasks (Eq. 12). Maximize total reward (Eq. 9).
    R: [N, M+1], task_ready_mask: [M], robot_idle_mask: [N]. Returns {agent_idx: task_node_id or "idle"}.
    """
    R_np = R.detach().cpu().numpy() if R.requires_grad else R.cpu().numpy()
    ready = np.where(task_ready_mask.cpu().numpy() > 0.5)[0]
    idle = np.where(robot_idle_mask.cpu().numpy() > 0.5)[0]
    assignment: Dict[int, str] = {i: "idle" for i in range(R_np.shape[0])}
    # Sort tasks by max reward (over idle robots) descending
    if len(ready) == 0 or len(idle) == 0:
        return assignment
    task_scores = [(j, max(R_np[i, j] for i in idle)) for j in ready]
    task_scores.sort(key=lambda x: -x[1])
    used_robots: set = set()
    for j, _ in task_scores:
        best_i = None
        best_r = -float("inf")
        for i in idle:
            if i in used_robots:
                continue
            if R_np[i, j] > best_r:
                best_r = R_np[i, j]
                best_i = i
        if best_i is not None:
            assignment[best_i] = task_order[j]
            used_robots.add(best_i)
    return assignment


# ---------------------------------------------------------------------------
# 8. SadcherAgent
# ---------------------------------------------------------------------------
class SadcherAgent:
    """
    Per-agent state for SADCHER: one active task at a time (no EDF queue).
    Interface-compatible with env.step(agents=...): get_active_job(), execute_task(),
    complete_active_job(), queue, dag_tracker. NO feasibility checks, NO WCET/energy masking.
    """

    def __init__(
        self,
        agent_id: str,
        cfg: SADCHER_CONFIG,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_id = agent_id
        self.cfg = cfg
        self._log = logger or logging.getLogger(f"sadcher.{agent_id}")
        self.dag_tracker: Optional[Any] = None
        self._queue: List[CommittedJob] = []  # 0 or 1 job

    @property
    def queue(self) -> List[CommittedJob]:
        return self._queue.copy()

    def reset(self) -> None:
        self._queue.clear()
        if self.dag_tracker is not None and hasattr(self.dag_tracker, "reset"):
            self.dag_tracker.reset()

    def reset_queue(self) -> None:
        """Clear queue (e.g. after heartbeat timeout). Compatible with CEA-EDF interface."""
        self._queue.clear()

    def get_active_job(self) -> Optional[CommittedJob]:
        if not self._queue:
            return None
        return self._queue[0]

    def execute_task(self, current_time_ms: float, delta_ms: float = 1.0) -> Optional[CommittedJob]:
        active = self.get_active_job()
        if active is None:
            return None
        active.remaining_work_ms -= delta_ms
        return active

    def complete_active_job(self) -> Optional[CommittedJob]:
        if not self._queue:
            return None
        completed = self._queue.pop(0)
        self._log.debug("Agent %s completed job %s.", self.agent_id, completed.job_id)
        return completed

    def assign_task(
        self,
        node_id: str,
        budget_ms: float,
        wcet_ms: Optional[float] = None,
        absolute_deadline_ms: Optional[float] = None,
    ) -> None:
        """Assign a single task to this agent (no EDF, no feasibility check)."""
        self._queue.clear()
        if wcet_ms is None:
            wcet_ms = getattr(self.cfg, "task_wcets", {}).get(node_id, 20000.0)
        d_v = (
            float(absolute_deadline_ms)
            if absolute_deadline_ms is not None
            else getattr(self.cfg, "global_deadline_ms", 180000.0)
        )
        job = CommittedJob(
            job_id=node_id,
            task_name=node_id,
            absolute_deadline_ms=d_v,
            remaining_work_ms=wcet_ms * np.random.uniform(0.7, 1.0),
            wcet=wcet_ms,
            total_budget_ms=budget_ms,
        )
        self._queue.append(job)
        self._log.info(
            "Agent %s assigned Task %s (total_budget=%.1f ms); queue size=%d.",
            self.agent_id, node_id, budget_ms, len(self._queue),
        )
