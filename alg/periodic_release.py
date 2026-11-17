"""EXP-2 periodic-release manager (Theorem 2(a) sim path).

Default-inert: nothing here runs unless ``set_exp2_runtime(release_mode="periodic")``
was called from ``alg.train_darts``. Node instance ids are minted ``f"{class_id}#{ell}"``
with ell in 0..L-1 so a second DAG copy cannot collide with class-id maps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from envs.dag_task import DAGGraph, TaskNode
from envs.simulation_env import (
    LocalDAGTracker,
    NODE_STATE_COMPLETED,
)

logger = logging.getLogger("train_darts")


def mint_instance_id(class_id: str, ell: int) -> str:
    return f"{class_id}#{int(ell)}"


def parse_instance_id(node_id: str) -> Tuple[str, Optional[int]]:
    """Return (class_id, ell). Single-shot ids without '#' yield ell=None."""
    if node_id is None:
        return "", None
    if "#" not in node_id:
        return str(node_id), None
    cls, _, rest = str(node_id).partition("#")
    try:
        return cls, int(rest)
    except ValueError:
        return str(node_id), None


def class_of(node_id: str) -> str:
    return parse_instance_id(node_id)[0]


@dataclass
class Exp2Runtime:
    release_mode: str = "single"
    release_period_ms: Optional[float] = None
    num_releases: int = 1
    alloc_map: Optional[Dict[str, str]] = None
    alloc_map_path: Optional[str] = None
    per_period_energy: str = "none"
    log_path: Optional[str] = None
    workload_id: str = ""
    stress: str = ""
    seed: int = 0
    ckpt_dir: str = ""
    ckpt_ep: int = -1
    D_v_ms: float = 0.0
    episode_idx: int = 0


_RUNTIME = Exp2Runtime()


def set_exp2_runtime(**kwargs: Any) -> Exp2Runtime:
    global _RUNTIME
    cur = _RUNTIME.__dict__.copy()
    cur.update(kwargs)
    _RUNTIME = Exp2Runtime(**{k: v for k, v in cur.items() if k in Exp2Runtime.__dataclass_fields__})
    return _RUNTIME


def get_exp2_runtime() -> Exp2Runtime:
    return _RUNTIME


def reset_exp2_runtime() -> None:
    global _RUNTIME
    _RUNTIME = Exp2Runtime()


def validate_periodic_horizon(
    *,
    release_period_ms: float,
    num_releases: int,
    relative_deadline_ms: float,
    max_episode_steps: int,
    global_deadline_ms: float,
    disable_feasibility_mask: bool,
    alloc_map_active: bool,
) -> None:
    """Hard-fail with an actionable message when periodic mode is mis-sized."""
    if release_period_ms is None:
        raise SystemExit(
            "[EXP2][periodic] horizon/flag validation failed:\n  - "
            "--release_period_ms is required when --release_mode periodic "
            "(positive multiple of 1000)"
        )
    p = float(release_period_ms)
    l = int(num_releases)
    d_v = float(relative_deadline_ms)
    steps = int(max_episode_steps)
    g = float(global_deadline_ms)
    problems: List[str] = []
    if p <= 0 or abs(p / 1000.0 - round(p / 1000.0)) > 1e-9:
        problems.append(
            f"--release_period_ms={p} must be a positive multiple of 1000"
        )
    if l < 1:
        problems.append(f"--num_releases={l} must be >= 1")
    if d_v > p + 1e-9:
        problems.append(
            f"effective relative_deadline_ms={d_v} must be <= --release_period_ms={p} "
            f"(set --relative_deadline_ms <= --release_period_ms)"
        )
    need_span = l * p + p
    have_span = steps * 1000.0
    if have_span + 1e-9 < need_span:
        problems.append(
            f"max_episode_steps*1000={have_span} must be >= num_releases*P + P = {need_span} "
            f"(raise --max_episode_steps to at least {int(need_span // 1000)})"
        )
    need_g = l * p
    if g + 1e-9 < need_g:
        problems.append(
            f"effective global_deadline_ms={g} must be >= num_releases*P = {need_g} "
            f"(raise --global_deadline_ms to at least {need_g})"
        )
    if alloc_map_active and disable_feasibility_mask:
        problems.append(
            "--disable_feasibility_mask must be False when an --alloc_map is active"
        )
    if problems:
        raise SystemExit(
            "[EXP2][periodic] horizon/flag validation failed:\n  - "
            + "\n  - ".join(problems)
        )


def _remap_dag(base: DAGGraph, ell: int, relative_deadline_ms: float) -> DAGGraph:
    out = DAGGraph(global_deadline_ms=base.global_deadline_ms)
    for nid, node in base.nodes.items():
        new_id = mint_instance_id(nid, ell)
        preds = [mint_instance_id(p, ell) for p in node.predecessors]
        succs = [mint_instance_id(s, ell) for s in node.successors]
        out.add_node(
            TaskNode(
                node_id=new_id,
                name=node.name,
                wcet=node.wcet,
                predecessors=preds,
                successors=succs,
                relative_deadline=relative_deadline_ms,
            )
        )
    return out


class _NodeStateProxy:
    """Dict-like router so ``global_tracker.node_states[id] = ...`` stays valid."""

    def __init__(self, manager: "PeriodicReleaseManager") -> None:
        self._mgr = manager

    def _tracker_for(self, node_id: str) -> Optional[LocalDAGTracker]:
        _, ell = parse_instance_id(node_id)
        if ell is None:
            return self._mgr.trackers.get(0)
        return self._mgr.trackers.get(ell)

    def __getitem__(self, node_id: str) -> str:
        tr = self._tracker_for(node_id)
        if tr is None:
            raise KeyError(node_id)
        return tr.node_states[node_id]

    def __setitem__(self, node_id: str, value: str) -> None:
        tr = self._tracker_for(node_id)
        if tr is None:
            return
        tr.node_states[node_id] = value

    def get(self, node_id: str, default: Any = None) -> Any:
        tr = self._tracker_for(node_id)
        if tr is None:
            return default
        return tr.node_states.get(node_id, default)

    def __contains__(self, node_id: object) -> bool:
        if not isinstance(node_id, str):
            return False
        tr = self._tracker_for(node_id)
        return tr is not None and node_id in tr.node_states

    def items(self):
        for tr in self._mgr.trackers.values():
            yield from tr.node_states.items()

    def keys(self):
        for tr in self._mgr.trackers.values():
            yield from tr.node_states.keys()


class TrackerFacade:
    def __init__(self, manager: "PeriodicReleaseManager") -> None:
        self._mgr = manager
        self.node_states = _NodeStateProxy(manager)

    @property
    def dag(self) -> DAGGraph:
        return self._mgr.merged_dag

    def get_ready_nodes(self) -> List[TaskNode]:
        out: List[TaskNode] = []
        for ell, tr in self._mgr.trackers.items():
            if ell in self._mgr.released:
                out.extend(tr.get_ready_nodes())
        return out

    def mark_completed(self, node_id: str) -> None:
        tr = self.node_states._tracker_for(node_id)
        if tr is None:
            return
        tr.mark_completed(node_id)
        clock = None
        # Seed successor ready times at the completion instant (minted ids are
        # invisible to update_ready_times_on_completion's class-id succ map).
        node = tr.dag.get_node(node_id)
        if node is None:
            return
        for succ_id in node.successors:
            succ = tr.dag.get_node(succ_id)
            if succ is None:
                continue
            if all(
                tr.node_states.get(p) == NODE_STATE_COMPLETED for p in succ.predecessors
            ):
                if succ_id not in self._mgr.ready_time_ms:
                    # clock filled by caller via manager.note_completion_clock
                    t = self._mgr.last_completion_clock_ms
                    self._mgr.ready_time_ms[succ_id] = float(t)

    def mark_dropped(self, node_id: str) -> None:
        tr = self.node_states._tracker_for(node_id)
        if tr is not None:
            tr.mark_dropped(node_id)

    def process_incoming_messages(self, messages: List[Any]) -> None:
        buckets: Dict[int, List[Any]] = {}
        unknown: List[Any] = []
        for m in messages:
            payload = getattr(m, "payload", None)
            nid = None
            if isinstance(payload, dict):
                nid = payload.get("node_id")
            _, ell = parse_instance_id(nid or "")
            if ell is None or ell not in self._mgr.trackers:
                unknown.append(m)
                continue
            buckets.setdefault(ell, []).append(m)
        for ell, group in buckets.items():
            self._mgr.trackers[ell].process_incoming_messages(group)
        # Unknown / class-id messages: ignore (AE LocalDAGTracker also skips
        # node_id not in self._node_state; see simulation_env.py:159).

    def reset(self) -> None:
        for tr in self._mgr.trackers.values():
            tr.reset()


class PeriodicReleaseManager:
    def __init__(self, P: float, L: int, D_v: float, per_period_energy: str = "none") -> None:
        self.P = float(P)
        self.L = int(L)
        self.D_v = float(D_v)
        self.per_period_energy = per_period_energy
        self.trackers: Dict[int, LocalDAGTracker] = {}
        self.dags: Dict[int, DAGGraph] = {}
        self.released: Set[int] = set()
        self.ready_time_ms: Dict[str, float] = {}
        self.merged_dag = DAGGraph()
        self.facade = TrackerFacade(self)
        self.last_completion_clock_ms = 0.0
        self._next_ell = 0
        self._next_boundary = 1
        self._probes: Dict[int, Dict[str, Any]] = {}
        self._commits: Dict[str, Tuple[str, float]] = {}
        self.n_jobs_expected = 0
        self._base_dag: Optional[DAGGraph] = None
        self._workload: Any = None
        self._env: Any = None
        self._cfg: Any = None
        self._relative_deadline_ms = D_v

    def note_completion_clock(self, clock_ms: float) -> None:
        self.last_completion_clock_ms = float(clock_ms)

    def record_commit(self, instance_id: str, agent_id: str, commit_ms: float) -> None:
        self._commits[instance_id] = (agent_id, float(commit_ms))

    def _instantiate(self, ell: int) -> List[str]:
        assert self._base_dag is not None
        dag = _remap_dag(self._base_dag, ell, self._relative_deadline_ms)
        tracker = LocalDAGTracker(dag)
        self.dags[ell] = dag
        self.trackers[ell] = tracker
        for nid, node in dag.nodes.items():
            self.merged_dag.add_node(node)
        self._register_env_ids(ell, dag)
        sources = [n.node_id for n in dag.get_sources()]
        return sources

    def _register_env_ids(self, ell: int, dag: DAGGraph) -> None:
        env = self._env
        if env is None:
            return
        wl = self._workload
        for class_id in wl.node_ids:
            inst = mint_instance_id(class_id, ell)
            if hasattr(env, "register_job_instance"):
                env.register_job_instance(inst, class_id)
            else:
                xy = dict(getattr(env, "task_locations", {})).get(class_id)
                if xy is None and wl is not None:
                    xy = wl.task_locations_m.get(class_id)
                if xy is not None:
                    env.task_locations[inst] = xy

    def bootstrap(
        self, env: Any, cfg: Any, workload: Any, P: float, L: int, D_v: float
    ) -> Tuple[Dict[str, float], TrackerFacade]:
        self.P = float(P)
        self.L = int(L)
        self.D_v = float(D_v)
        self._relative_deadline_ms = float(D_v)
        self._env = env
        self._cfg = cfg
        self._workload = workload
        self._base_dag = DAGGraph.from_workload(workload, cfg)
        self.n_jobs_expected = self.L * len(tuple(workload.node_ids))
        sources = self._instantiate(0)
        self.released.add(0)
        self._next_ell = 1
        for sid in sources:
            self.ready_time_ms[sid] = 0.0
        return self.ready_time_ms, self.facade

    def maybe_release(self, clock_ms: float) -> List[str]:
        newly: List[str] = []
        while self._next_ell < self.L and float(clock_ms) + 1e-9 >= self._next_ell * self.P:
            ell = self._next_ell
            sources = self._instantiate(ell)
            self.released.add(ell)
            r = float(ell) * self.P
            for sid in sources:
                self.ready_time_ms[sid] = r
            newly.extend(sources)
            self._next_ell += 1
            logger.info(
                "[EXP2][periodic] released ell=%d at clock=%.0fms sources=%s",
                ell, clock_ms, sources,
            )
        return newly

    def boundary_probe(self, agents: List[Any], env: Any, clock_ms: float) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        bats = list(getattr(env, "_battery_wh", []))
        for i, agent in enumerate(agents):
            qlen = 0
            if hasattr(agent, "queue_length"):
                qlen = int(agent.queue_length())
            elif hasattr(agent, "queue"):
                qlen = len(agent.queue)
            bat = float(bats[i]) if i < len(bats) else float("nan")
            out[i] = {
                "queue_empty": qlen == 0,
                "battery_wh": bat,
                "agent_id": getattr(agent, "agent_id", f"agent_{i+1}"),
            }
        return out

    def on_post_step_clock(
        self,
        clock_ms: float,
        agents: List[Any],
        env: Any,
    ) -> bool:
        """maybe_release + boundary probe + optional restore. Returns True if restored."""
        self.maybe_release(clock_ms)
        restored = False
        while self._next_boundary <= self.L and float(clock_ms) + 1e-9 >= self._next_boundary * self.P:
            probe = self.boundary_probe(agents, env, clock_ms)
            self._probes[self._next_boundary] = probe
            if self.per_period_energy == "reset" and hasattr(env, "restore_battery_to_initial"):
                env.restore_battery_to_initial()
                restored = True
                logger.info(
                    "[EXP2][periodic] battery restored at boundary k=%d clock=%.0fms",
                    self._next_boundary, clock_ms,
                )
            self._next_boundary += 1
        return restored

    def sidecar_rows(self, env: Any, runtime: Optional[Exp2Runtime] = None) -> List[Dict[str, Any]]:
        from alg.defer_fallback import fallback_blocked_jobs

        rt = runtime or get_exp2_runtime()
        blocked = fallback_blocked_jobs()
        alloc_tok = rt.alloc_map_path or ""
        if rt.alloc_map:
            blob = json.dumps(rt.alloc_map, sort_keys=True)
            alloc_tok = f"{rt.alloc_map_path or ''}#sha256={hashlib.sha256(blob.encode()).hexdigest()[:16]}"
        completed = getattr(env, "completed_tasks_ms", {}) or {}
        rows: List[Dict[str, Any]] = []
        for ell, dag in sorted(self.dags.items()):
            next_b = ell + 1
            probe = self._probes.get(next_b, {})
            for nid, node in dag.nodes.items():
                cls, _ = parse_instance_id(nid)
                agent_id, commit_ms = self._commits.get(nid, ("", float("nan")))
                finish_ms = completed.get(nid)
                r_ms = self.ready_time_ms.get(nid, float(ell) * self.P)
                deadline_ms = float(r_ms) + self.D_v
                met = ""
                if finish_ms is not None:
                    met = str(bool(float(finish_ms) <= deadline_ms + 1e-9)).lower()
                q_empty = ""
                bat_b = ""
                if agent_id:
                    for rec in probe.values():
                        if rec.get("agent_id") == agent_id:
                            q_empty = str(bool(rec.get("queue_empty"))).lower()
                            bat_b = rec.get("battery_wh", "")
                            break
                rows.append({
                    "workload": rt.workload_id,
                    "stress": rt.stress,
                    "seed": rt.seed,
                    "ckpt_dir": rt.ckpt_dir,
                    "ckpt_ep": rt.ckpt_ep,
                    "P_ms": self.P,
                    "L": self.L,
                    "D_v_ms": self.D_v,
                    "per_period_energy": self.per_period_energy,
                    "alloc_map_sha_or_path": alloc_tok,
                    "release_idx": ell,
                    "job_class": cls,
                    "instance_id": nid,
                    "agent_id": agent_id,
                    "r_ms": r_ms,
                    "commit_ms": commit_ms,
                    "finish_ms": finish_ms if finish_ms is not None else "",
                    "deadline_ms": deadline_ms,
                    "met": met,
                    "queue_empty_at_next_boundary": q_empty,
                    "battery_at_boundary_wh": bat_b,
                    "fallback_blocked": str(nid in blocked).lower(),
                })
        return rows

    def write_sidecar(self, log_dir: str, env: Any, header: List[str]) -> str:
        path = os.path.join(log_dir, "release_log.csv")
        rows = self.sidecar_rows(env)
        write_header = not os.path.isfile(path)
        import csv
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            for r in rows:
                w.writerow([r.get(k, "") for k in header])
        return path
