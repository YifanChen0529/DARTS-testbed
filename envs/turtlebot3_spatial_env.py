"""
2D Spatial MARL environment for TurtleBot3, compatible with OpenAI Gym / PettingZoo style.
Aligns with DARTS draft: hardware dynamics (Eq. 21 Energy model), communication Δ_max,
and action space for bidding (η). Physical movement is EDF-driven, not RL output.

The environment is **workload-parametric**: task DAG edges and per-node map coordinates
come from a ``Workload`` instance (DBench-4 registry: ``B1_CDR``, ``B2_LML``, ``B3_IQI``,
``B4_DEM`` in ``envs.workloads``). When no workload is passed, ``B1_CDR`` is used so legacy
call sites behave as before aside from spatial layout, which follows the registered map.

State: [x_norm, y_norm, battery_norm, workload_norm, dx_norm, dy_norm] per agent.
No step-level MARL actions; movement is strictly EDF-driven. Each agent automatically
moves toward its active EDF job's task location; when distance <= PICK_RADIUS_M the agent
stops and executes in place. Bidding is epoch-level (auction) only.

Unified with simulation_env: uses NetworkSimulator for NODE_COMPLETED broadcasts and
delivery to LocalDAGTracker; global clock _clock_ms drives both physics and network.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.messaging_transport import MessagingTransport

import numpy as np

try:
    from envs.simulation_env import NetworkSimulator, LocalDAGTracker, SubtaskCompletionTracker
except ImportError:
    from simulation_env import NetworkSimulator, LocalDAGTracker, SubtaskCompletionTracker

try:
    from envs.workloads import Workload, get_workload
except ImportError:
    from workloads import Workload, get_workload


# ---------------------------------------------------------------------------
# TurtleBot3 / Orin Nano hardware (crucial for Eq. 21 Energy model)
# ---------------------------------------------------------------------------
MAX_VELOCITY_MPS = 0.22  # m/s, TurtleBot3
P_IDLE_W = 2.5           # Idle power (Orin Nano only), W
P_MOVING_W = 7.5         # Moving power (Motors + Nano), W
P_COMM_W = 0.5           # Communication transmission cost per message, W

# Communication latency (physical 2D network), ms
DELTA_MAX_MS = 50
MIN_DELAY_MS = 10

# Map
MAP_SIZE_M = 10.0        # 10m x 10m
PICK_RADIUS_M = 0.3      # Within this distance, Pick succeeds for a task location

# Default step duration for step(); 1 sec per env step
DEFAULT_STEP_DT_S = 1.0
# Battery capacity (Wh) for percentage scale
BATTERY_CAPACITY_WH = 50.0


@dataclass
class PendingMessage:
    """Message in flight with delivery time (latency Δ_max)."""
    sender: int
    receiver: int
    payload: Any
    send_time_ms: float
    delivery_time_ms: float


def _distance(xy1: Tuple[float, float], xy2: Tuple[float, float]) -> float:
    return math.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


def _point_in_rect(px: float, py: float, rx: float, ry: float, rw: float, rh: float) -> bool:
    """Strict (open) boundaries so points on obstacle edges are not considered inside."""
    return rx < px < rx + rw and ry < py < ry + rh


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    """
    True if line segment (ax,ay)-(bx,by) intersects segment (cx,cy)-(dx,dy).
    Uses parametric form; when parallel/collinear, checks 1D interval overlap.
    """
    denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denom) < 1e-10:
        # Parallel: if not collinear, no intersection
        if abs((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) > 1e-10:
            return False
        # Collinear: project onto x (or y if segment vertical) and check overlap
        if abs(bx - ax) >= abs(by - ay):
            p1_lo, p1_hi = min(ax, bx), max(ax, bx)
            p2_lo, p2_hi = min(cx, dx), max(cx, dx)
        else:
            p1_lo, p1_hi = min(ay, by), max(ay, by)
            p2_lo, p2_hi = min(cy, dy), max(cy, dy)
        return p1_lo <= p2_hi and p2_lo <= p1_hi
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
    s = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0


class TurtleBot3Env:
    """
    2D Spatial MARL environment for TurtleBot3 with standard reset() and step(actions).
    State per agent: 2D coords [x,y], battery level, local EDF queue workload, distance to target.
    Energy model (Eq. 21): Idle 2.5 W, Moving 7.5 W, Comm 0.5 W per message.
    Communication: latency in [MIN_DELAY_MS, DELTA_MAX_MS] ms per message.
    """

    def __init__(
        self,
        num_agents: int = 4,
        map_size: float = MAP_SIZE_M,
        task_locations: Optional[Dict[str, Tuple[float, float]]] = None,
        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
        step_dt_s: float = DEFAULT_STEP_DT_S,
        max_velocity_mps: float = MAX_VELOCITY_MPS,
        p_idle_w: float = P_IDLE_W,
        p_moving_w: float = P_MOVING_W,
        p_comm_w: float = P_COMM_W,
        delta_max_ms: float = DELTA_MAX_MS,
        battery_capacity_wh: float = BATTERY_CAPACITY_WH,
        global_deadline_ms: Optional[float] = None,
        reward_task_completion: float = 10.0,
        reward_step_penalty: Optional[float] = None,
        cfg: Optional[Any] = None,
        seed: Optional[int] = None,
        workload: Optional[Workload] = None,
        enable_rta_fallback: bool = True,
        rta_fallback_mode: str = "rebudget",
        transport: Optional["MessagingTransport"] = None,
    ):
        self.num_agents = num_agents
        self.map_size = map_size
        if workload is None:
            workload = get_workload("B1_CDR")
        self.workload = workload
        self._task_predecessors = dict(workload.predecessors)
        self._workload_node_ids = tuple(workload.node_ids)
        _all_predecessor_ids = {
            pred for preds in workload.predecessors.values() for pred in preds
        }
        self._sink_node_ids = tuple(
            s for s in workload.node_ids if s not in _all_predecessor_ids
        )
        if task_locations is not None:
            self.task_locations = dict(task_locations)
        else:
            self.task_locations = dict(workload.task_locations_m)
        self.step_dt_s = step_dt_s
        if cfg is not None:
            self._reward_task_completion = getattr(cfg, "reward_task_completion", 10.0)
            self._reward_step_penalty = getattr(cfg, "reward_step_penalty", -0.01)
        else:
            self._reward_task_completion = reward_task_completion
            self._reward_step_penalty = reward_step_penalty if reward_step_penalty is not None else -0.01
        self.max_velocity_mps = max_velocity_mps
        self.p_idle_w = p_idle_w
        self.p_moving_w = p_moving_w
        self.p_comm_w = p_comm_w
        self.delta_max_ms = delta_max_ms
        self.battery_capacity_wh = battery_capacity_wh  # fallback when per-agent list not used
        self.cfg = cfg  # for battery_capacities_wh in reset
        self._global_deadline_ms = global_deadline_ms if global_deadline_ms is not None else 120000.0
        self._rng = random.Random(seed)

        # Obstacles: list of (x, y, width, height) in meters; axis-aligned rectangles
        self.obstacles = obstacles or self._default_obstacles()

        # Per-agent state: position (x,y), battery_wh, workload_norm (0..1), messages_sent_this_step
        self._positions: List[Tuple[float, float]] = []
        self._battery_wh: List[float] = []
        self._initial_battery_wh: List[float] = []  # snapshot at reset for Eq. 22 relative depletion
        self._workload_norm: List[float] = []   # 0 = no work, 1 = max normalized workload
        self._messages_sent_this_step: List[int] = []

        # Global clock (ms): single source of truth for physics and network (unified with simulation_env)
        self._clock_ms = 0.0
        self._pending_messages: List[PendingMessage] = []
        # Inter-agent messaging: in-process simulator (default) or injected gRPC transport (Jetson).
        if transport is None:
            self._network: Any = NetworkSimulator(seed=seed)
        else:
            self._network = transport
        self._completion_tracker = SubtaskCompletionTracker(
            node_ids=workload.node_ids,
            sink_node_ids=self._sink_node_ids,
        )
        self.completed_tasks_ms: Dict[str, float] = {}

        # Step: move distance per step = max_velocity * step_dt
        self._move_per_step = max_velocity_mps * step_dt_s

        # Lightweight execution timing (used only to detect illegal execution-level RTA triggers).
        self._execution_start_ms: List[float] = []
        self._execution_job_id: List[Optional[str]] = []
        self._execution_budget_ms: List[float] = []
        if rta_fallback_mode not in ("raise", "rebudget"):
            raise ValueError(
                f"Unknown rta_fallback_mode={rta_fallback_mode!r}. "
                "Choose from {'raise', 'rebudget'}."
            )
        self.enable_rta_fallback = enable_rta_fallback
        self.rta_fallback_mode = rta_fallback_mode
        self._rta_fallback_events_total = 0

    def _default_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """Default static obstacles (x, y, w, h) in the 10x10 map."""
        return [
            (3.0, 3.0, 1.5, 1.5),
            (5.0, 6.0, 1.0, 2.0),
            (7.0, 2.0, 1.2, 1.2),
        ]

    def _clip_to_map(self, x: float, y: float) -> Tuple[float, float]:
        return (
            max(0.0, min(self.map_size, x)),
            max(0.0, min(self.map_size, y)),
        )

    def _get_target_location(self, agent_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Nearest incomplete task whose DAG predecessors are all completed (ready to pick)."""
        ready = [
            (tid, loc)
            for tid, loc in self.task_locations.items()
            if tid not in self.completed_tasks_ms
            and all(p in self.completed_tasks_ms for p in self._task_predecessors.get(tid, []))
        ]
        if not ready:
            return None
        return min(ready, key=lambda item: _distance(agent_pos, item[1]))[1]

    def _task_ready_to_complete(self, task_id: str) -> bool:
        """True if task_id is uncompleted and all its DAG predecessors are completed."""
        if task_id in self.completed_tasks_ms:
            return False
        return all(p in self.completed_tasks_ms for p in self._task_predecessors.get(task_id, []))

    def _collides_obstacles(self, x: float, y: float) -> bool:
        for ox, oy, w, h in self.obstacles:
            if _point_in_rect(x, y, ox, oy, w, h):
                return True
        return False

    def _line_intersects_rect(
        self,
        ax: float, ay: float,
        bx: float, by: float,
        rx: float, ry: float, rw: float, rh: float,
    ) -> bool:
        """
        True if the line segment from (ax,ay) to (bx,by) intersects the
        axis-aligned rectangle (rx, ry, rw, rh) (including interior or edges).
        """
        if _point_in_rect(ax, ay, rx, ry, rw, rh) or _point_in_rect(bx, by, rx, ry, rw, rh):
            return True
        x2, y2 = rx + rw, ry + rh
        # Four edges: bottom, top, left, right
        if _segments_intersect(ax, ay, bx, by, rx, ry, x2, ry):
            return True
        if _segments_intersect(ax, ay, bx, by, rx, y2, x2, y2):
            return True
        if _segments_intersect(ax, ay, bx, by, rx, ry, rx, y2):
            return True
        if _segments_intersect(ax, ay, bx, by, x2, ry, x2, y2):
            return True
        return False

    def _wcet_travel_from_origin_ms(
        self,
        origin_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        task_base_duration_ms: float,
        obstacle_penalty: float,
        safety_margin: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """Shared travel + execution WCET from origin (x,y) to target; returns (cost_ms, origin_xy)."""
        ox, oy = origin_xy
        tx, ty = target_xy
        distance_m = _distance((ox, oy), (tx, ty))
        for (ox_r, oy_r, w, h) in self.obstacles:
            if self._line_intersects_rect(ox, oy, tx, ty, ox_r, oy_r, w, h):
                distance_m *= obstacle_penalty
                break
        travel_time_ms = (distance_m / self.max_velocity_mps) * 1000.0
        chain_cost_ms = travel_time_ms * safety_margin + task_base_duration_ms
        return chain_cost_ms, (ox, oy)

    def estimate_dynamic_wcet_ms(
        self,
        agent_id: int,
        target_xy: Tuple[float, float],
        task_base_duration_ms: float,
        obstacle_penalty: float = 1.5,
        safety_margin: float = 1.1,
    ) -> float:
        """
        Dynamic WCET estimate for bidding-network features (distance from agent).

        Distance-from-agent variant; use ``estimate_chain_wcet_ms`` for Eq. (10)
        chain-topology sequential cost in feasibility tests.
        """
        if agent_id < 0 or agent_id >= len(self._positions):
            return task_base_duration_ms * safety_margin
        ax, ay = self._positions[agent_id]
        cost_ms, _ = self._wcet_travel_from_origin_ms(
            (ax, ay),
            target_xy,
            task_base_duration_ms,
            obstacle_penalty,
            safety_margin,
        )
        return cost_ms

    def estimate_chain_wcet_ms(
        self,
        agent_id: int,
        prev_xy: Optional[Tuple[float, float]],
        target_xy: Tuple[float, float],
        task_base_duration_ms: float,
        obstacle_penalty: float = 1.5,
        safety_margin: float = 1.1,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Eq. (10) chain-aware travel cost.

        ``prev_xy``: predecessor task's (x, y) for chain position j >= 2;
        pass ``None`` for j == 1 (agent's current position is used).

        Returns ``(chain_cost_ms, travel_origin_xy)`` where
        ``chain_cost_ms = travel_time_ms * safety_margin + task_base_duration_ms``.
        """
        if prev_xy is not None:
            origin_xy = (float(prev_xy[0]), float(prev_xy[1]))
        else:
            if agent_id < 0 or agent_id >= len(self._positions):
                return task_base_duration_ms * safety_margin, (0.0, 0.0)
            ax, ay = self._positions[agent_id]
            origin_xy = (ax, ay)
        return self._wcet_travel_from_origin_ms(
            origin_xy,
            target_xy,
            task_base_duration_ms,
            obstacle_penalty,
            safety_margin,
        )

    def estimate_chain_wcet_components_ms(
        self, agent_id, prev_xy, target_xy, task_base_duration_ms,
        obstacle_penalty=1.5, safety_margin=1.1,
    ):
        """TB-1 additive companion to estimate_chain_wcet_ms.
        Returns (travel_envelope_ms, exec_ms, origin_xy) with the invariant
        travel_envelope_ms + exec_ms == estimate_chain_wcet_ms(...)[0] bit-exactly.
        Arithmetic is duplicated VERBATIM from estimate_chain_wcet_ms /
        _wcet_travel_from_origin_ms so those AE-era methods stay untouched;
        the bit-exactness unit test guards drift -- edit both copies together
        or not at all. Degenerate guard (prev_xy None, agent_id out of range)
        reports the whole value as travel_envelope_ms with exec_ms = 0.0
        (unreachable in the trainer path)."""
        if prev_xy is not None:
            origin_xy = (float(prev_xy[0]), float(prev_xy[1]))
        else:
            if agent_id < 0 or agent_id >= len(self._positions):
                return task_base_duration_ms * safety_margin, 0.0, (0.0, 0.0)
            ax, ay = self._positions[agent_id]
            origin_xy = (ax, ay)
        ox, oy = origin_xy
        tx, ty = target_xy
        distance_m = _distance((ox, oy), (tx, ty))
        for (ox_r, oy_r, w, h) in self.obstacles:
            if self._line_intersects_rect(ox, oy, tx, ty, ox_r, oy_r, w, h):
                distance_m *= obstacle_penalty
                break
        travel_time_ms = (distance_m / self.max_velocity_mps) * 1000.0
        travel_envelope_ms = travel_time_ms * safety_margin
        return travel_envelope_ms, task_base_duration_ms, (ox, oy)

    def _move_toward_target(
        self, agent_id: int, target_xy: Optional[Tuple[float, float]]
    ) -> Tuple[float, float, bool]:
        """
        Move agent toward target by _move_per_step. Used by EDF-driven navigation and RTA Fallback.
        Wall-sliding: if diagonal move hits an obstacle, try X-only then Y-only so the agent can
        slide along rectangle edges and reach the target instead of getting stuck.
        Returns (new_x, new_y, is_moving). If target is None or distance <= PICK_RADIUS_M, do not move.
        """
        x, y = self._positions[agent_id]
        if target_xy is None:
            return (x, y, False)
        dist = _distance((x, y), target_xy)
        if dist <= PICK_RADIUS_M:
            return (x, y, False)
        tx, ty = target_xy
        dx = tx - x
        dy = ty - y
        scale = self._move_per_step / dist
        move_x = dx * scale
        move_y = dy * scale

        # 1) Try diagonal move first
        nx, ny = self._clip_to_map(x + move_x, y + move_y)
        if not self._collides_obstacles(nx, ny):
            return (nx, ny, True)

        # 2) Diagonal collided: try X-only (slide along vertical obstacle face)
        nx_x, ny_x = self._clip_to_map(x + move_x, y)
        if not self._collides_obstacles(nx_x, ny_x):
            return (nx_x, ny_x, True)

        # 3) X-only failed: try Y-only (slide along horizontal obstacle face)
        nx_y, ny_y = self._clip_to_map(x, y + move_y)
        if not self._collides_obstacles(nx_y, ny_y):
            return (nx_y, ny_y, True)

        # 4) All direct moves blocked: try perpendicular nudge to escape corner
        perp_x = -move_y
        perp_y = move_x
        for sign in [1.0, -1.0]:
            nx_p, ny_p = self._clip_to_map(x + sign * perp_x, y + sign * perp_y)
            if not self._collides_obstacles(nx_p, ny_p):
                return (nx_p, ny_p, True)

        # 5) All moves blocked: stay in place
        return (x, y, False)

    def _consume_energy(self, agent_id: int, is_moving: bool, num_messages: int) -> None:
        """Update battery using Eq. 21–style model: idle vs moving + comm cost."""
        dt_h = self.step_dt_s / 3600.0
        p = self.p_moving_w if is_moving else self.p_idle_w
        energy_wh = p * dt_h + num_messages * self.p_comm_w * (1.0 / 3600.0)  # comm as 1 sec per message for simplicity, or scale
        # Simpler: comm cost per message as W * (small dt); use same dt_h for comm
        energy_wh = p * dt_h + num_messages * self.p_comm_w * dt_h
        self._battery_wh[agent_id] = max(0.0, self._battery_wh[agent_id] - energy_wh)

    def _send_message(self, sender: int, receiver: int, payload: Any) -> None:
        """Queue message with latency in [MIN_DELAY_MS, DELTA_MAX_MS] ms."""
        delay_ms = self._rng.uniform(MIN_DELAY_MS, self.delta_max_ms)
        self._pending_messages.append(PendingMessage(
            sender=sender, receiver=receiver, payload=payload,
            send_time_ms=self._clock_ms, delivery_time_ms=self._clock_ms + delay_ms,
        ))
        self._pending_messages.sort(key=lambda m: m.delivery_time_ms)

    def poll_deliveries(self) -> List[PendingMessage]:
        """Return and remove messages with delivery_time_ms <= current clock."""
        delivered = [m for m in self._pending_messages if m.delivery_time_ms <= self._clock_ms]
        self._pending_messages = [m for m in self._pending_messages if m.delivery_time_ms > self._clock_ms]
        return delivered

    def send_message(self, sender: int, receiver: int, payload: Any) -> None:
        """Schedule a message with Δ_max latency; caller should track messages for energy."""
        self._send_message(sender, receiver, payload)

    def get_observation(self, agent_id: int) -> np.ndarray:
        """
        State for agent: [x_norm, y_norm, battery_norm, workload_norm, dx_norm, dy_norm].
        (dx_norm, dy_norm) is the normalized direction to the nearest incomplete task (directional vision for navigation).
        If all tasks complete, (dx_norm, dy_norm) = (0, 0).
        """
        x, y = self._positions[agent_id]
        cap_wh = (
            self._initial_battery_wh[agent_id]
            if agent_id < len(self._initial_battery_wh) and self._initial_battery_wh[agent_id] > 0
            else self.battery_capacity_wh
        )
        battery_pct = (self._battery_wh[agent_id] / cap_wh) * 100.0
        battery_norm = battery_pct / 100.0
        workload = self._workload_norm[agent_id]
        completed = self.completed_tasks_ms
        incomplete = [(tid, loc) for tid, loc in self.task_locations.items() if tid not in completed]
        if not incomplete:
            dx_norm = 0.0
            dy_norm = 0.0
        else:
            _, target_loc = min(incomplete, key=lambda item: _distance((x, y), item[1]))
            tx, ty = target_loc
            dx_norm = (tx - x) / self.map_size
            dy_norm = (ty - y) / self.map_size
        return np.array(
            [x / self.map_size, y / self.map_size, battery_norm, workload, dx_norm, dy_norm],
            dtype=np.float32,
        )

    def get_observations(self) -> Dict[int, np.ndarray]:
        """Observations for all agents (keyed by agent_id index)."""
        return {i: self.get_observation(i) for i in range(self.num_agents)}

    def set_workload(self, agent_id: int, workload_norm: float) -> None:
        """Set current local EDF queue workload (0..1 normalized) for this agent."""
        self._workload_norm[agent_id] = max(0.0, min(1.0, workload_norm))

    def reset(
        self,
        *,
        init_positions: Optional[List[Tuple[float, float]]] = None,
        rng_seed: Optional[int] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Reset environment for a new episode. Returns (observations, info).
        Zeros clock, clears completed tasks and RTA state so no state bleeds across episodes.
        init_positions: optional list of (x,y) per agent; otherwise random valid positions.
        rng_seed: if set, reseeds the env RNG so distributed replicas share identical world samples.
        """
        if rng_seed is not None:
            self._rng = random.Random(rng_seed)
        self._clock_ms = 0.0
        self._pending_messages = []
        if isinstance(self._network, NetworkSimulator):
            self._network._pending = []
        else:
            clear = getattr(self._network, "clear_inbox", None)
            if callable(clear):
                clear()
        self._completion_tracker.reset()
        if hasattr(self, "completed_tasks_ms") and isinstance(self.completed_tasks_ms, dict):
            self.completed_tasks_ms.clear()
        else:
            self.completed_tasks_ms = {}

        self._positions = []
        self._battery_wh = []
        self._workload_norm = []
        self._messages_sent_this_step = []

        self._execution_start_ms = [0.0] * self.num_agents
        self._execution_job_id = [None] * self.num_agents
        self._execution_budget_ms = [0.0] * self.num_agents
        self._rta_fallback_events_total = 0

        capacities_wh = list(getattr(self.cfg, "battery_capacities_wh", [1.0, 1.5, 2.0, 2.5])[:self.num_agents])
        while len(capacities_wh) < self.num_agents:
            capacities_wh.append(self.battery_capacity_wh)
        for i in range(self.num_agents):
            if init_positions is not None and i < len(init_positions):
                x, y = init_positions[i]
                x, y = self._clip_to_map(x, y)
                if self._collides_obstacles(x, y):
                    x, y = self._sample_valid_position()
            else:
                x, y = self._sample_valid_position()
            self._positions.append((x, y))
            self._battery_wh.append(capacities_wh[i])
            self._workload_norm.append(0.0)
            self._messages_sent_this_step.append(0)

        self._initial_battery_wh = self._battery_wh.copy()

        info = {"clock_ms": self._clock_ms}
        return self.get_observations(), info

    def _sample_valid_position(self) -> Tuple[float, float]:
        for _ in range(100):
            x = self._rng.uniform(0.0, self.map_size)
            y = self._rng.uniform(0.0, self.map_size)
            if not self._collides_obstacles(x, y):
                return (x, y)
        return (0.0, 0.0)

    def step(
        self,
        actions: Optional[List[Any]] = None,
        messages_sent: Optional[List[int]] = None,
        agents: Optional[List[Any]] = None,
    ) -> Tuple[Dict[int, np.ndarray], List[float], List[bool], Dict]:
        """
        Step environment. No MARL actions required; movement is strictly EDF-driven.

        actions: optional (unused); kept for API compatibility.
        messages_sent: optional list of number of messages sent this step per agent (for energy).
        agents: optional list of DartsMARLAgent (with .get_active_job(), .execute_task, .queue, .dag_tracker).
                When provided: (1) execute_task is run only when at target (distance <= PICK_RADIUS_M);
                (2) movement is EDF-driven: agent moves toward active job target; (3) execution-level fallback
                safety behavior uses rta_fallback_mode:
                    - "raise": keep fail-fast NotImplementedError behavior
                    - "rebudget": extend local execution budget and continue;
                (4) after advancing clock, delivered messages are routed to LocalDAGTracker.

        Returns: (observations, rewards, dones, info).
        """
        step_dt_ms = self.step_dt_s * 1000.0

        # 1) Task execution only when at target: drain WCET; on finish, complete_active_job and signal DAG
        completed_agent_indices: List[int] = []
        newly_completed_nodes: List[str] = []
        # TB-1: parallel to is_moving_list below; True iff execute_task ran for agent i this step.
        # Declared here because the drain loop that sets it precedes the movement loop.
        is_executing_list: List[bool] = [False] * self.num_agents
        if agents is not None and len(agents) == self.num_agents:
            for i, agent in enumerate(agents):
                active = agent.get_active_job()
                if active is not None:
                    target_xy = self.task_locations.get(active.job_id)
                    at_target = (
                        target_xy is not None
                        and _distance(self._positions[i], target_xy) <= PICK_RADIUS_M
                    )
                    if at_target:
                        agent.execute_task(self._clock_ms, delta_ms=step_dt_ms)
                        is_executing_list[i] = True
                    # When execution timer finishes (WCET drained), pop job and signal completion
                    if at_target and active.remaining_work_ms <= 0:
                        completed_job = agent.complete_active_job()
                        if completed_job is not None:
                            node_id = completed_job.job_id
                            newly_completed_nodes.append(node_id)
                            self._completion_tracker.record(node_id, self._clock_ms)
                            self.completed_tasks_ms[node_id] = self._clock_ms
                            completed_agent_indices.append(i)
                            logging.getLogger("episode_events").info(
                                "[Time: %.0fms] ✅ Agent %d physically COMPLETED Subtask %s!",
                                self._clock_ms, i + 1, node_id,
                            )
                            payload = {"event": "NODE_COMPLETED", "node_id": node_id}
                            for other in agents:
                                if other.agent_id != agent.agent_id:
                                    self._network.send(
                                        sender=agent.agent_id,
                                        receiver=other.agent_id,
                                        payload=payload,
                                        current_time_ms=self._clock_ms,
                                    )

        if messages_sent is None:
            messages_sent = [0] * self.num_agents
        for i in range(self.num_agents):
            self._messages_sent_this_step[i] = messages_sent[i] if i < len(messages_sent) else 0

        if len(self._execution_job_id) != self.num_agents:
            self._execution_start_ms = [0.0] * self.num_agents
            self._execution_job_id = [None] * self.num_agents
            self._execution_budget_ms = [0.0] * self.num_agents

        # 2) MARL movement (EDF-driven). Execution-level RTA Simplex removed: auction-time Algorithm 1
        #    is the sole defer/fallback mechanism. If this guard fires, the cert / budget pipeline is inconsistent.
        rewards: List[float] = []
        is_moving_list: List[bool] = []
        raw_env_reward_step_total = 0.0
        rta_fallback_jobs_step: List[str] = []
        for i in range(self.num_agents):
            agent_pos_before = self._positions[i]
            target_xy: Optional[Tuple[float, float]] = None
            active_job = agents[i].get_active_job() if (agents is not None and i < len(agents)) else None
            if active_job is not None:
                target_xy = self.task_locations.get(active_job.job_id)

            trigger_rta_bug = False
            if active_job is not None:
                if self._execution_job_id[i] != active_job.job_id:
                    self._execution_job_id[i] = active_job.job_id
                    self._execution_start_ms[i] = self._clock_ms
                    self._execution_budget_ms[i] = active_job.total_budget_ms
                elapsed_ms = self._clock_ms - self._execution_start_ms[i]
                dist_to_target = _distance(agent_pos_before, target_xy) if target_xy else 0.0
                effective_dist = dist_to_target
                if target_xy is not None:
                    for (ox, oy, w, h) in self.obstacles:
                        if self._line_intersects_rect(
                            agent_pos_before[0], agent_pos_before[1],
                            target_xy[0], target_xy[1],
                            ox, oy, w, h,
                        ):
                            effective_dist = dist_to_target * 1.5
                            break
                fallback_time_ms = (effective_dist / self.max_velocity_mps) * 1000.0
                wcet_budget_ms = self._execution_budget_ms[i]
                trigger_rta_bug = (
                    self.enable_rta_fallback
                    and target_xy is not None
                    and dist_to_target > PICK_RADIUS_M
                    and (elapsed_ms + fallback_time_ms) >= wcet_budget_ms
                )

            if trigger_rta_bug:
                log = logging.getLogger("episode_events")
                if self.rta_fallback_mode == "raise":
                    log.critical(
                        "[Time: %.0fms] FATAL: execution-level RTA fallback would have triggered for agent %d job %s — "
                        "removed under DARTS Algorithm 1; indicates auction/feasibility inconsistency.",
                        self._clock_ms,
                        i + 1,
                        active_job.job_id if active_job is not None else "?",
                    )
                    raise NotImplementedError(
                        "Execution-level RTA fallback was removed; defer/fallback is auction-time only (Algorithm 1). "
                        "This trigger indicates a deeper feasibility-test or budget bug."
                    )

                delta_max_ms = float(getattr(self.cfg, "delta_max_ms", self.delta_max_ms))
                old_budget_ms = float(self._execution_budget_ms[i])
                new_budget_ms = max(old_budget_ms, elapsed_ms + fallback_time_ms + delta_max_ms)
                self._execution_budget_ms[i] = new_budget_ms
                if active_job is not None and hasattr(active_job, "total_budget_ms"):
                    active_job.total_budget_ms = new_budget_ms
                    rta_fallback_jobs_step.append(active_job.job_id)
                self._rta_fallback_events_total += 1
                log.warning(
                    "[Time: %.0fms] Execution-level fallback REBUDGET for agent %d job %s: "
                    "old_budget=%.1fms -> new_budget=%.1fms (elapsed=%.1fms, travel=%.1fms, Δ_max=%.1fms).",
                    self._clock_ms,
                    i + 1,
                    active_job.job_id if active_job is not None else "?",
                    old_budget_ms,
                    new_budget_ms,
                    elapsed_ms,
                    fallback_time_ms,
                    delta_max_ms,
                )

            if target_xy is not None:
                new_x, new_y, is_moving = self._move_toward_target(i, target_xy)
            else:
                new_x, new_y = self._positions[i]
                is_moving = False

            self._positions[i] = (new_x, new_y)
            agent_pos = (new_x, new_y)

            r = 0.0
            old_dist = _distance(agent_pos_before, target_xy) if target_xy is not None else 0.0
            new_dist = _distance(agent_pos, target_xy) if target_xy is not None else 0.0
            r += (old_dist - new_dist) * 10.0
            r += self._reward_step_penalty
            if i in completed_agent_indices:
                r += self._reward_task_completion

            # r^env: task completion + step penalty only (Eq. 36 / unshaped env return)
            raw_env_i = self._reward_step_penalty
            if i in completed_agent_indices:
                raw_env_i += self._reward_task_completion
            raw_env_reward_step_total += raw_env_i

            is_moving_list.append(is_moving)
            self._consume_energy(i, is_moving, self._messages_sent_this_step[i])
            rewards.append(r)

        # 3) Advance global clock (unified: physics and network use same time)
        self._clock_ms += step_dt_ms

        # 4) Poll network deliveries and route to each agent's LocalDAGTracker
        delivered = self._network.poll_deliveries(self._clock_ms)
        if agents is not None:
            for agent in agents:
                tracker = getattr(agent, "dag_tracker", None)
                if tracker is not None:
                    mine = [m for m in delivered if m.receiver == agent.agent_id]
                    tracker.process_incoming_messages(mine)

        # Legacy: poll internal message queue (if anything still uses it)
        self.poll_deliveries()

        e_safe_wh = getattr(self.cfg, "e_safe_wh", 0.05) if getattr(self, "cfg", None) else 0.05
        dones = [self._battery_wh[i] < e_safe_wh for i in range(self.num_agents)]
        subtask_completion_times_ms = {
            nid: self.completed_tasks_ms.get(nid) for nid in self._workload_node_ids
        }
        sink_times = [
            self.completed_tasks_ms.get(sid) for sid in self._sink_node_ids
        ]
        if self._sink_node_ids and all(t is not None for t in sink_times):
            task_time_ms = max(sink_times)
        else:
            task_time_ms = None
        if task_time_ms is not None:
            end_to_end_deadline_miss = task_time_ms > self._global_deadline_ms
        else:
            end_to_end_deadline_miss = self._clock_ms > self._global_deadline_ms
        init_wh = self._initial_battery_wh if len(self._initial_battery_wh) == self.num_agents else [self.battery_capacity_wh] * self.num_agents
        info = {
            "clock_ms": self._clock_ms,
            "positions": self._positions.copy(),
            "battery_wh": self._battery_wh.copy(),
            "battery_depletion_pct": [(init_wh[i] - self._battery_wh[i]) / init_wh[i] if init_wh[i] > 0 else 0.0 for i in range(self.num_agents)],
            "is_moving": is_moving_list,
            "is_executing": is_executing_list,
            "bids": [0.0] * self.num_agents,  # no step-level bids; bidding is epoch-level only
            "newly_completed_nodes": newly_completed_nodes,
            "subtask_completion_times_ms": subtask_completion_times_ms,
            "task_time_ms": task_time_ms,
            "end_to_end_deadline_miss": end_to_end_deadline_miss,
            "e_safe_wh_threshold": e_safe_wh,
            "rta_fallback_events_step": len(rta_fallback_jobs_step),
            "rta_fallback_events_total": int(self._rta_fallback_events_total),
            "rta_fallback_jobs": sorted(set(rta_fallback_jobs_step)),
        }
        info["overall_completion_ms"] = task_time_ms
        info["raw_env_reward"] = raw_env_reward_step_total
        return self.get_observations(), rewards, dones, info

    @property
    def initial_battery_wh(self) -> List[float]:
        """Initial battery capacity (Wh) per agent at episode start, for Eq. 22 relative depletion."""
        if len(self._initial_battery_wh) == self.num_agents:
            return self._initial_battery_wh.copy()
        caps = getattr(self.cfg, "battery_capacities_wh", None) if getattr(self, "cfg", None) else None
        if caps:
            return list(caps)[:self.num_agents]
        return [self.battery_capacity_wh] * self.num_agents

    @property
    def clock_ms(self) -> float:
        """Global clock in ms (single source of truth for physics and network)."""
        return self._clock_ms

    @property
    def network(self) -> Any:
        """Message transport (simulator or gRPC) for NODE_COMPLETED broadcasts and delivery."""
        return self._network

    @property
    def positions(self) -> List[Tuple[float, float]]:
        return self._positions.copy()

    def get_agent_position(self, agent_id: int) -> Optional[Tuple[float, float]]:
        """Return (x, y) for the given agent index, or None if out of range."""
        if agent_id < 0 or agent_id >= len(self._positions):
            return None
        return self._positions[agent_id]

    def restore_battery_to_initial(self) -> List[float]:
        """EXP-2: restore each agent's Wh to the episode-start snapshot.

        Does not mutate ``_initial_battery_wh``. Not called from step/_consume_energy.
        """
        n = min(len(self._battery_wh), len(self._initial_battery_wh))
        for i in range(n):
            self._battery_wh[i] = self._initial_battery_wh[i]
        return self._battery_wh.copy()

    def register_job_instance(self, instance_id: str, class_id: str) -> None:
        """EXP-2: register a minted ``class#ell`` id on every class-id-keyed env map.

        Required: ``task_locations`` (execute/move lookup). Extended: ``_workload_node_ids``,
        ``_task_predecessors``, ``_completion_tracker._node_ids``. ``completed_tasks_ms`` is a
        dict and accepts unknown-id writes. ``_sink_node_ids`` is left on class ids so the
        AE e2e-miss test stays class-keyed (admitted periodic runs size global_deadline
        past L*P). Agent ``dag_tracker.process_incoming_messages`` already skips unknown
        ids (simulation_env.py:159).
        """
        xy = self.task_locations.get(class_id)
        if xy is None and getattr(self, "workload", None) is not None:
            xy = self.workload.task_locations_m.get(class_id)
        if xy is not None:
            self.task_locations[instance_id] = xy
        preds = list(self._task_predecessors.get(class_id, []))
        minted_preds = []
        suffix = ""
        if "#" in instance_id:
            suffix = "#" + instance_id.split("#", 1)[1]
        for p in preds:
            minted_preds.append(p + suffix if suffix and "#" not in p else p)
        self._task_predecessors[instance_id] = minted_preds
        if instance_id not in self._workload_node_ids:
            self._workload_node_ids = tuple(self._workload_node_ids) + (instance_id,)
        tr = getattr(self, "_completion_tracker", None)
        if tr is not None and instance_id not in tr._node_ids:
            tr._node_ids = tuple(tr._node_ids) + (instance_id,)

    @property
    def single_observation_size(self) -> int:
        return 6  # [x_norm, y_norm, battery_norm, workload_norm, dx_norm, dy_norm]
