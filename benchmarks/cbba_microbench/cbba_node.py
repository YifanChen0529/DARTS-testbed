#!/usr/bin/env python3
"""Per-agent CBBA process: CbbaPeer + CbbaControl on one port.

Quarantine: stdlib + grpc + generated stubs + cbba_core only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import grpc

from benchmarks.cbba_microbench import cbba_pb2 as pb2
from benchmarks.cbba_microbench import cbba_pb2_grpc as pb2_grpc
from benchmarks.cbba_microbench.cbba_core import (
    Message,
    build_bundle,
    consensus_update,
    fresh_state,
)

_FATAL = "[cbba_node] FATAL:"


def _nbytes(msg: object) -> int:
    fn = getattr(msg, "ByteSize", None)
    if callable(fn):
        return int(fn())
    return len(msg.SerializeToString())  # type: ignore[attr-defined]


def _digest(z: List[int], y: List[float]) -> str:
    return hashlib.sha256(repr((tuple(z), tuple(y))).encode("utf-8")).hexdigest()


def _peer_map(agent_id: int, n_agents: int, peers_str: str) -> Dict[int, str]:
    parts = [p.strip() for p in peers_str.split(",") if p.strip()]
    others = [i for i in range(n_agents) if i != agent_id]
    if len(parts) == n_agents:
        return {i: parts[i] for i in others}
    if len(parts) == n_agents - 1:
        return {aid: hp for aid, hp in zip(others, parts)}
    raise SystemExit(
        f"{_FATAL} --peers must have {n_agents - 1} or {n_agents} host:port entries, "
        f"got {len(parts)}"
    )


def _xy_pairs(flat: List[float]) -> List[Tuple[float, float]]:
    if len(flat) % 2 != 0:
        raise ValueError("coordinate list must have even length")
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


class _Mailbox:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._by_round: Dict[int, Dict[int, pb2.AgentLists]] = {}

    def reset(self) -> None:
        with self._cv:
            self._by_round.clear()
            self._cv.notify_all()

    def put(self, msg: pb2.AgentLists) -> None:
        with self._cv:
            bucket = self._by_round.setdefault(int(msg.round), {})
            bucket[int(msg.sender_id)] = msg
            self._cv.notify_all()

    def wait_round(
        self, round_idx: int, n_peers: int, timeout_s: float
    ) -> List[pb2.AgentLists]:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                bucket = self._by_round.get(round_idx, {})
                if len(bucket) >= n_peers:
                    return list(bucket.values())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"mailbox round {round_idx}: have {len(bucket)}/{n_peers}"
                    )
                self._cv.wait(timeout=remaining)


class _NodeServicer(pb2_grpc.CbbaPeerServicer, pb2_grpc.CbbaControlServicer):
    def __init__(self, agent_id: int, n_agents: int, peer_map: Dict[int, str]) -> None:
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.peer_map = peer_map
        self.mailbox = _Mailbox()
        self._lock = threading.Lock()
        self._stubs: Dict[int, pb2_grpc.CbbaPeerStub] = {}
        self._channels: List[grpc.Channel] = []
        self._worker: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._trial_done: Optional[pb2.TrialDone] = None
        self._trial_error: Optional[BaseException] = None
        self._pending_spec: Optional[pb2.StartTrial] = None
        self._receiver_id = agent_id

    def _stubs_ready(self) -> Dict[int, pb2_grpc.CbbaPeerStub]:
        if len(self._stubs) == len(self.peer_map):
            return self._stubs
        for aid, target in self.peer_map.items():
            if aid in self._stubs:
                continue
            ch = grpc.insecure_channel(target)
            self._channels.append(ch)
            self._stubs[aid] = pb2_grpc.CbbaPeerStub(ch)
        return self._stubs

    def ExchangeLists(self, request: pb2.AgentLists, context: grpc.ServicerContext) -> pb2.Ack:
        self.mailbox.put(request)
        return pb2.Ack(receiver_id=self.agent_id, round=int(request.round))

    def BeginTrial(self, request: pb2.StartTrial, context: grpc.ServicerContext) -> pb2.Empty:
        # Prepare only: reset mailbox and stash spec. The worker starts on
        # PollTrial so the runner can BeginTrial all four nodes before any
        # ExchangeLists is sent (otherwise a late reset drops early messages).
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._done.wait(timeout=120.0)
            self._done.clear()
            self._trial_done = None
            self._trial_error = None
            self._pending_spec = request
            self._worker = None
            self.mailbox.reset()
        return pb2.Empty()

    def PollTrial(self, request: pb2.Empty, context: grpc.ServicerContext) -> pb2.TrialDone:
        with self._lock:
            spec = self._pending_spec
            if spec is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "trial not ready")
            if self._worker is None:
                worker = threading.Thread(
                    target=self._run_trial,
                    args=(spec,),
                    name=f"cbba-trial-{spec.trial_id}",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
        while not self._done.wait(timeout=0.25):
            if context.is_active() is False:
                context.abort(grpc.StatusCode.CANCELLED, "client gone")
        if self._trial_error is not None:
            context.abort(grpc.StatusCode.INTERNAL, str(self._trial_error))
        if self._trial_done is None:
            context.abort(grpc.StatusCode.UNAVAILABLE, "trial not ready")
        return self._trial_done

    def _run_trial(self, spec: pb2.StartTrial) -> None:
        try:
            done = self._execute(spec)
            self._trial_done = done
        except BaseException as exc:  # noqa: BLE001 — surface to PollTrial
            self._trial_error = exc
            print(f"{_FATAL} agent {self.agent_id} trial {spec.trial_id}: {exc}", file=sys.stderr)
        finally:
            self._done.set()

    def _execute(self, spec: pb2.StartTrial) -> pb2.TrialDone:
        m = int(spec.m)
        l_t = int(spec.l_t)
        tasks = _xy_pairs(list(spec.task_xy))
        starts = _xy_pairs(list(spec.start_xy))
        if len(starts) != self.n_agents:
            raise ValueError(f"start_xy has {len(starts)} points, need {self.n_agents}")
        if m != len(tasks):
            raise ValueError(f"m={m} but task_xy has {len(tasks)} points")
        state = fresh_state(self.agent_id, starts[self.agent_id], m, self.n_agents)
        n_peers = self.n_agents - 1
        max_rounds = max(1, 4 * max(l_t, 0) * max(m, 0))
        msgs_sent = 0
        bytes_sent = 0
        last_change = 0
        round_digests: List[pb2.RoundDigest] = []
        stubs = self._stubs_ready()

        for r in range(1, max_rounds + 1):
            t_comp0 = time.perf_counter()
            build_changed = build_bundle(state, tasks, l_t)
            compute_ms = (time.perf_counter() - t_comp0) * 1000.0

            req = pb2.AgentLists(
                sender_id=self.agent_id,
                round=r,
                z=list(state.z),
                y=list(state.y),
                s=list(state.s),
                send_ns=time.perf_counter_ns(),
            )
            nbytes = _nbytes(req)
            for aid, stub in stubs.items():
                stub.ExchangeLists(req, timeout=60.0)
                msgs_sent += 1
                bytes_sent += nbytes

            t_blk0 = time.perf_counter()
            incoming_pb = self.mailbox.wait_round(r, n_peers, timeout_s=60.0)
            blocking_ms = (time.perf_counter() - t_blk0) * 1000.0

            incoming = [
                Message(
                    sender_id=int(msg.sender_id),
                    z=tuple(msg.z),
                    y=tuple(msg.y),
                    s=tuple(msg.s),
                )
                for msg in incoming_pb
            ]
            t_cons0 = time.perf_counter()
            cons_changed = consensus_update(state, incoming, r)
            compute_ms += (time.perf_counter() - t_cons0) * 1000.0

            changed = bool(build_changed or cons_changed)
            digest = _digest(state.z, state.y)
            round_digests.append(
                pb2.RoundDigest(
                    agent_id=self.agent_id,
                    trial_id=int(spec.trial_id),
                    round=r,
                    digest=digest,
                    changed=changed,
                    compute_ms=compute_ms,
                    blocking_ms=blocking_ms,
                )
            )
            if changed:
                last_change = r
            else:
                break

        n_rounds = last_change
        return pb2.TrialDone(
            agent_id=self.agent_id,
            trial_id=int(spec.trial_id),
            n_rounds=n_rounds,
            final_z=list(state.z),
            final_y=list(state.y),
            msgs_sent=msgs_sent,
            bytes_sent=bytes_sent,
            round_digests=round_digests,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="EXP-3 CBBA per-agent node (loopback/WLAN).")
    p.add_argument("--agent_id", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--peers", type=str, required=True)
    p.add_argument("--n_agents", type=int, default=4)
    args = p.parse_args()
    if not (0 <= args.agent_id < args.n_agents):
        raise SystemExit(f"{_FATAL} agent_id {args.agent_id} not in [0, {args.n_agents})")
    peer_map = _peer_map(args.agent_id, args.n_agents, args.peers)
    servicer = _NodeServicer(args.agent_id, args.n_agents, peer_map)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    pb2_grpc.add_CbbaPeerServicer_to_server(servicer, server)
    pb2_grpc.add_CbbaControlServicer_to_server(servicer, server)
    bind = f"0.0.0.0:{args.port}"
    server.add_insecure_port(bind)
    server.start()
    print(f"cbba_node agent_id={args.agent_id} port={args.port}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1.0)


if __name__ == "__main__":
    main()
