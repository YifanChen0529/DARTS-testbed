"""
Simulation environment for Real-Time CPS task allocation.
Includes global clock and a message-passing network simulator for heterogeneous agents.

When using TurtleBot3Env for MARL: its clock (env.clock_ms) and env.network (NetworkSimulator)
are the single source of truth; step() advances the clock and poll_deliveries uses it so
network delays (up to DELTA_MAX ms) are applied correctly.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from envs.dag_task import DAGGraph, TaskNode


# Maximum one-way message delay (ms). Actual delay is uniform in [10, DELTA_MAX].
DELTA_MAX = 50
MIN_DELAY_MS = 10


@dataclass
class PendingMessage:
    """A message in flight with a fixed delay and delivery time."""

    sender: str
    receiver: str
    payload: Any
    send_time_ms: float
    delivery_time_ms: float


class NetworkSimulator:
    """
    Message-passing simulator for 4 heterogeneous agents.
    Every message incurs a delay uniformly sampled in [10, DELTA_MAX] ms.

    This class satisfies :class:`envs.messaging_transport.MessagingTransport` and is the transport
    used when ``deployment_mode == "simulator"`` (single Python process, training default).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._pending: List[PendingMessage] = []
        self._agent_ids = ["agent_1", "agent_2", "agent_3", "agent_4"]

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids.copy()

    def send(self, sender: str, receiver: str, payload: Any, current_time_ms: float) -> None:
        """
        Schedule a message from sender to receiver. Delay is uniform in [10, DELTA_MAX] ms.
        """
        delay_ms = self._rng.uniform(MIN_DELAY_MS, DELTA_MAX)
        delivery_time_ms = current_time_ms + delay_ms
        self._pending.append(
            PendingMessage(
                sender=sender,
                receiver=receiver,
                payload=payload,
                send_time_ms=current_time_ms,
                delivery_time_ms=delivery_time_ms,
            )
        )
        # Keep pending list sorted by delivery time for efficient polling
        self._pending.sort(key=lambda m: m.delivery_time_ms)

    def poll_deliveries(self, current_time_ms: float) -> List[PendingMessage]:
        """
        Return and remove all messages whose delivery_time_ms <= current_time_ms.
        """
        delivered = []
        remaining = []
        for m in self._pending:
            if m.delivery_time_ms <= current_time_ms:
                delivered.append(m)
            else:
                remaining.append(m)
        self._pending = remaining
        return delivered


# Node states for distributed DAG execution (local view, updated by delayed messages)
NODE_STATE_WAITING = "WAITING_FOR_PREDECESSORS"
NODE_STATE_READY = "READY"
NODE_STATE_ASSIGNED = "ASSIGNED"
NODE_STATE_COMPLETED = "COMPLETED"
NODE_STATE_DROPPED = "DROPPED"


class LocalDAGTracker:
    """
    Per-agent local view of a DAG's execution state, updated only by incoming network messages.
    Enforces distributed task execution: agents may only bid on tasks that are READY in this
    local view (i.e., unlocked by received NODE_COMPLETED messages).
    """

    def __init__(self, dag: DAGGraph) -> None:
        self._dag = dag
        self._node_state: dict[str, str] = {}
        self._init_node_states()

    def _init_node_states(self) -> None:
        """Set entry nodes READY, all others WAITING (used by __init__ and reset())."""
        self._node_state.clear()
        entry_nodes = self._dag.get_sources()
        for node in self._dag.nodes.values():
            if node in entry_nodes:
                self._node_state[node.node_id] = NODE_STATE_READY
            else:
                self._node_state[node.node_id] = NODE_STATE_WAITING

    def reset(self) -> None:
        """Reinitialize node states for a new episode (no state bleed)."""
        self._init_node_states()

    @property
    def dag(self) -> DAGGraph:
        """The DAG graph (e.g. for creating a central tracker with the same structure)."""
        return self._dag

    @property
    def node_states(self) -> dict[str, str]:
        """Mutable view of node_id -> state (READY, ASSIGNED, COMPLETED, etc.)."""
        return self._node_state

    def mark_completed(self, node_id: str) -> None:
        """Mark node_id COMPLETED and set any successor to READY if all its predecessors are COMPLETED."""
        if node_id not in self._node_state:
            return
        self._node_state[node_id] = NODE_STATE_COMPLETED
        node = self._dag.get_node(node_id)
        if node is None:
            return
        for succ_id in node.successors:
            succ = self._dag.get_node(succ_id)
            if succ is None:
                continue
            if all(
                self._node_state.get(pred_id) == NODE_STATE_COMPLETED
                for pred_id in succ.predecessors
            ):
                self._node_state[succ_id] = NODE_STATE_READY

    def process_incoming_messages(self, messages: List[PendingMessage]) -> None:
        """
        Apply delivered messages to the local state. For each message with
        payload {"event": "NODE_COMPLETED", "node_id": "X"}, mark X COMPLETED and
        then set any successor Y to READY iff all of Y's predecessors are COMPLETED.
        """
        for msg in messages:
            payload = msg.payload
            if not isinstance(payload, dict):
                continue
            if payload.get("event") != "NODE_COMPLETED":
                continue
            node_id = payload.get("node_id")
            if node_id is None or node_id not in self._node_state:
                continue
            self._node_state[node_id] = NODE_STATE_COMPLETED
            node = self._dag.get_node(node_id)
            if node is None:
                continue
            for succ_id in node.successors:
                succ = self._dag.get_node(succ_id)
                if succ is None:
                    continue
                if all(
                    self._node_state.get(pred_id) == NODE_STATE_COMPLETED
                    for pred_id in succ.predecessors
                ):
                    self._node_state[succ_id] = NODE_STATE_READY

    def get_ready_nodes(self) -> List[TaskNode]:
        """
        Return nodes that are READY in this local view (all predecessors completed
        according to received messages). The MARL agent uses this to know what it can bid on.
        """
        return [
            self._dag.nodes[nid]
            for nid, state in self._node_state.items()
            if state == NODE_STATE_READY
        ]

    def mark_dropped(self, node_id: str) -> None:
        """
        Mark a job DROPPED (Algorithm 1 case N1: no measurable fallback). Does not unlock
        successors — structural infeasibility / mission-planner alert per DARTS Section I-B-j.
        """
        if node_id not in self._node_state:
            return
        self._node_state[node_id] = NODE_STATE_DROPPED

    def get_node_state(self, node_id: str) -> Optional[str]:
        """Return current local state of a node, or None if unknown."""
        return self._node_state.get(node_id)


class SubtaskCompletionTracker:
    """
    Tracks the exact timestamp (ms) when each task node transitions to COMPLETED.
    Used to build step() info: subtask_completion_times_ms, task_time_ms,
    end_to_end_deadline_miss. Node and sink sets are workload-parametric.
    """

    def __init__(
        self,
        node_ids: Tuple[str, ...] = ("A", "B", "C", "D"),
        sink_node_ids: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._node_ids = tuple(node_ids)
        if sink_node_ids is None:
            self._sink_node_ids: Tuple[str, ...] = (
                (self._node_ids[-1],) if self._node_ids else ()
            )
        else:
            self._sink_node_ids = tuple(sink_node_ids)
        self._completion_ms: dict[str, float] = {}

    def record(self, node_id: str, time_ms: float) -> None:
        """Record that node_id transitioned to COMPLETED at time_ms."""
        self._completion_ms[node_id] = time_ms

    def reset(self) -> None:
        """Clear all completion timestamps (call at episode reset)."""
        self._completion_ms.clear()

    def get_info(self, global_deadline_ms: Optional[float] = None) -> dict[str, Any]:
        """
        Build the completion-time info dict for step() return.
        Returns:
            subtask_completion_times_ms: dict mapping each node_id to completion ms or None.
            task_time_ms: max completion time among sink nodes when all sinks are recorded,
                else None if any sink is incomplete.
            end_to_end_deadline_miss: True if task_time_ms is None (sinks incomplete)
                or if task_time_ms > global_deadline_ms.
        """
        subtask_completion_times_ms: dict[str, Optional[float]] = {
            nid: self._completion_ms.get(nid) for nid in self._node_ids
        }
        if self._sink_node_ids and all(
            s in self._completion_ms for s in self._sink_node_ids
        ):
            task_time_ms: Optional[float] = max(
                self._completion_ms[s] for s in self._sink_node_ids
            )
        else:
            task_time_ms = None
        end_to_end_deadline_miss = True
        if task_time_ms is not None and global_deadline_ms is not None:
            end_to_end_deadline_miss = task_time_ms > global_deadline_ms
        elif task_time_ms is not None:
            end_to_end_deadline_miss = False
        return {
            "subtask_completion_times_ms": subtask_completion_times_ms,
            "task_time_ms": task_time_ms,
            "end_to_end_deadline_miss": end_to_end_deadline_miss,
        }


class SimulationEnv:
    """
    Real-Time CPS simulation environment.
    Maintains a global clock (1 ms resolution) and supports stepping in time
    and releasing a DAG task at a specified time.
    """

    def __init__(self, network_seed: Optional[int] = None) -> None:
        self._global_clock_ms = 0.0
        self._network = NetworkSimulator(seed=network_seed)
        self._released_dags: List[tuple[float, DAGGraph]] = []  # (release_time, dag)

    @property
    def global_clock(self) -> float:
        """Current simulation time in ms (1 ms resolution)."""
        return self._global_clock_ms

    @property
    def network(self) -> NetworkSimulator:
        return self._network

    def step(self, dt_ms: float = 1.0) -> None:
        """Advance the global clock by dt_ms (default 1 ms)."""
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        self._global_clock_ms += dt_ms

    def release_dag(self, dag: DAGGraph, at_time_ms: Optional[float] = None) -> None:
        """
        Trigger the release of a DAG task. If at_time_ms is None, use current clock.
        """
        t = at_time_ms if at_time_ms is not None else self._global_clock_ms
        self._released_dags.append((t, dag))

    def get_dags_released_by(self, time_ms: float) -> List[DAGGraph]:
        """Return all DAGs released at or before time_ms."""
        return [dag for t, dag in self._released_dags if t <= time_ms]

    def get_pending_deliveries(self) -> List[PendingMessage]:
        """Return messages that would be delivered at or before current clock."""
        return self._network.poll_deliveries(self._global_clock_ms)

    def reset(self) -> None:
        """Reset environment for a new episode: clock to 0, clear released DAGs and pending messages."""
        self._global_clock_ms = 0.0
        self._released_dags = []
        self._network._pending = []
