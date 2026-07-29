"""
DAG-based task representation for Real-Time CPS task allocation.
Models the "Time-Critical Coordinated Target Retrieval" mission.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from envs.workloads import Workload, get_workload
except ImportError:
    from workloads import Workload, get_workload


@dataclass
class TaskNode:
    """A single node in a DAG task with real-time attributes."""

    node_id: str
    name: str
    wcet: float  # Worst-Case Execution Time in ms
    predecessors: List[str]  # list of node_ids
    successors: List[str]  # list of node_ids
    relative_deadline: Optional[float] = None  # ms, optional per-node deadline

    def __post_init__(self) -> None:
        if self.predecessors is None:
            self.predecessors = []
        if self.successors is None:
            self.successors = []


class DAGGraph:
    """
    Directed Acyclic Graph representing a time-critical mission.
    Used for the Time-Critical Coordinated Target Retrieval scenario.
    """

    def __init__(self, global_deadline_ms: Optional[float] = None) -> None:
        self._nodes: dict[str, TaskNode] = {}
        self._global_deadline_ms = global_deadline_ms
        self._node_id_to_index: Dict[str, int] = {}

    @property
    def nodes(self) -> dict[str, TaskNode]:
        return self._nodes

    @property
    def global_deadline_ms(self) -> Optional[float]:
        return self._global_deadline_ms

    def add_node(self, node: TaskNode) -> None:
        if node.node_id not in self._node_id_to_index:
            self._node_id_to_index[node.node_id] = len(self._node_id_to_index)
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        return self._nodes.get(node_id)

    def get_sources(self) -> List[TaskNode]:
        """Return nodes with no predecessors (entry nodes)."""
        return [n for n in self._nodes.values() if not n.predecessors]

    def get_sinks(self) -> List[TaskNode]:
        """Return nodes with no successors (exit nodes)."""
        return [n for n in self._nodes.values() if not n.successors]

    def get_verified_adjacency_matrix(self) -> "torch.Tensor":
        """
        Return the physically verified dependency adjacency as an n×n float tensor
        (n = number of nodes in insertion order via ``add_node``).
        adj[i, j] = 1.0 if task i is a predecessor of task j (dependency i -> j).
        Used as ground truth for graph inference cross-entropy loss L_graph(ψ; τ_k).
        """
        if torch is None:
            raise RuntimeError("torch is required for get_verified_adjacency_matrix")
        n = len(self._node_id_to_index)
        adj = torch.zeros(n, n, dtype=torch.float32)
        for node_id, node in self._nodes.items():
            j = self._node_id_to_index.get(node_id)
            if j is None:
                continue
            for pred_id in node.predecessors:
                i = self._node_id_to_index.get(pred_id)
                if i is not None:
                    adj[i, j] = 1.0
        return adj

    @staticmethod
    def from_workload(workload: Workload, cfg: Optional[Any] = None) -> "DAGGraph":
        """
        Build a DAG from a registered CPS benchmark workload.

        Uses ``cfg.global_deadline_ms`` / ``cfg.task_wcets`` when present on ``cfg``;
        otherwise WCETs and graph deadlines come from the workload.
        """
        g_dl = workload.global_deadline_ms
        if cfg is not None:
            g_dl = getattr(cfg, "global_deadline_ms", g_dl)

        wcets: Dict[str, float] = dict(workload.task_wcets_ms)
        if cfg is not None:
            override = getattr(cfg, "task_wcets", None)
            if override:
                wcets = {**wcets, **override}

        preds_w = workload.predecessors
        derived_succ: Dict[str, List[str]] = {nid: [] for nid in workload.node_ids}
        for j in workload.node_ids:
            for p in preds_w[j]:
                derived_succ[p].append(j)

        dag = DAGGraph(global_deadline_ms=g_dl)
        rel_dl = workload.relative_deadline_ms
        for nid in workload.node_ids:
            dag.add_node(
                TaskNode(
                    node_id=nid,
                    name=workload.node_names[nid],
                    wcet=wcets[nid],
                    predecessors=list(preds_w[nid]),
                    successors=list(derived_succ[nid]),
                    relative_deadline=rel_dl,
                )
            )

        for nid in workload.node_ids:
            tn = dag.get_node(nid)
            assert tn is not None, f"from_workload: missing node {nid!r}"
            assert set(tn.predecessors) == set(
                preds_w[nid]
            ), (
                f"from_workload {workload.workload_id!r}: predecessor mismatch on {nid!r}"
            )
            assert set(tn.successors) == set(
                derived_succ[nid]
            ), (
                f"from_workload {workload.workload_id!r}: successor mismatch on {nid!r}"
            )

        expected_global = workload.global_deadline_ms
        if cfg is not None:
            expected_global = getattr(cfg, "global_deadline_ms", expected_global)
        assert dag.global_deadline_ms == expected_global, (
            f"from_workload {workload.workload_id!r}: global_deadline_ms should equal "
            f"workload value or cfg override"
        )

        return dag

    @staticmethod
    def create_from_workload_id(workload_id: str, cfg: Optional[Any] = None) -> "DAGGraph":
        """Canonical factory: build a DAG for a registered ``workload_id``."""
        return DAGGraph.from_workload(get_workload(workload_id), cfg)

    @staticmethod
    def create_target_retrieval_dag(cfg: Optional[Any] = None) -> "DAGGraph":
        """
        Factory method: build the Time-Critical Coordinated Target Retrieval DAG
        (benchmark workload ``B1_CDR``).

        When ``cfg`` is omitted, attempts ``DARTS_CONFIG()`` so existing call sites
        keep the same configuration resolution as before.
        """
        if cfg is None:
            try:
                from alg.config_darts import DARTS_CONFIG

                cfg = DARTS_CONFIG()
            except ImportError:
                pass
        return DAGGraph.from_workload(get_workload("B1_CDR"), cfg)
