"""
gRPC-backed :class:`MessagingTransport` for ``deployment_mode == "distributed_jetson"``.

One OS process per Jetson; ``Send`` delivers pickled payloads to peers. ``poll_deliveries`` uses
``delivery_time_ms == current_time_ms`` (real network already delivered the message).
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from collections import deque
from concurrent import futures
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import grpc

from envs.proto import darts_messaging_pb2 as pb2
from envs.proto import darts_messaging_pb2_grpc as pb2_grpc
from envs.simulation_env import PendingMessage


class JetsonGrpcTransport:
    """
    Implements the same surface as :class:`envs.simulation_env.NetworkSimulator`
    (``agent_ids``, ``send``, ``poll_deliveries``) over gRPC.

    Parameters
    ----------
    local_agent_id
        This process's agent id (e.g. ``agent_1``). Must be a key of ``peer_endpoints``.
    peer_endpoints
        Map ``agent_id -> "host:port"`` for every team member including this node.
    bind_port
        Local TCP port this process listens on (must match the port in ``peer_endpoints[local_agent_id]``).
    calibration_output_dir
        Directory for ``measured_delta_max_ms_<agent_id>.json`` (typically ``results/<exp_name>/``).
    """

    def __init__(
        self,
        local_agent_id: str,
        peer_endpoints: Dict[str, str],
        bind_port: int,
        calibration_output_dir: Path,
    ) -> None:
        if local_agent_id not in peer_endpoints:
            raise ValueError(
                f"local_agent_id {local_agent_id!r} must be a key of jetson_endpoints "
                f"(1:1 agent ↔ Jetson). Keys: {sorted(peer_endpoints)}"
            )
        self._local = local_agent_id
        self._peers = dict(peer_endpoints)
        self._bind_port = int(bind_port)
        self._cal_dir = Path(calibration_output_dir)
        self._agent_ids = sorted(peer_endpoints.keys())

        self._inbox: Deque[PendingMessage] = deque()
        self._inbox_lock = threading.Lock()
        self._server: Optional[grpc.Server] = None
        self._stubs: Dict[str, pb2_grpc.AgentMessengerStub] = {}
        self._channels: Dict[str, grpc.Channel] = {}
        self._max_one_way_ms: float = 0.0
        self._started = False

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    def clear_inbox(self) -> None:
        with self._inbox_lock:
            self._inbox.clear()

    def _enqueue(self, msg: PendingMessage) -> None:
        with self._inbox_lock:
            self._inbox.append(msg)

    # --- gRPC servicer ---
    class _Servicer(pb2_grpc.AgentMessengerServicer):
        def __init__(self, outer: "JetsonGrpcTransport") -> None:
            self._outer = outer

        def Send(self, request: pb2.Envelope, context: grpc.ServicerContext) -> pb2.Ack:
            if request.receiver != self._outer._local:
                return pb2.Ack(ok=False)
            try:
                payload = pickle.loads(request.payload) if request.payload else None
            except Exception:
                payload = None
            self._outer._enqueue(
                PendingMessage(
                    sender=request.sender,
                    receiver=request.receiver,
                    payload=payload,
                    send_time_ms=float(request.send_time_ms),
                    delivery_time_ms=float(request.send_time_ms),
                )
            )
            return pb2.Ack(ok=True)

        def Ping(self, request: pb2.PingRequest, context: grpc.ServicerContext) -> pb2.PingResponse:
            return pb2.PingResponse(t_recv_ms=time.time() * 1000.0)

        def Subscribe(self, request: pb2.SubscribeRequest, context: grpc.ServicerContext):
            # Reserved; coordinator uses ``Send`` with ``receiver=agent_k`` for control messages.
            yield from ()

    def start_server(self) -> None:
        if self._started:
            return
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        pb2_grpc.add_AgentMessengerServicer_to_server(self._Servicer(self), self._server)
        self._server.add_insecure_port(f"[::]:{self._bind_port}")
        self._server.start()
        self._started = True

        for aid, target in self._peers.items():
            if aid == self._local:
                continue
            ch = grpc.insecure_channel(target)
            self._channels[aid] = ch
            self._stubs[aid] = pb2_grpc.AgentMessengerStub(ch)

    def stop_server(self, grace: float = 2.0) -> None:
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()
        self._stubs.clear()
        if self._server is not None:
            self._server.stop(grace)
            self._server = None
        self._started = False

    def run_calibration(self, cfg_delta_max_ms: float, window_s: float = 5.0, num_pings: int = 100) -> float:
        """
        Empirical one-way latency: ``RTT/2`` from ``Ping`` RPCs (100 per peer over ``window_s``).
        Writes ``measured_delta_max_ms_<local_agent_id>.json`` and returns the max one-way ms observed.
        """
        if not self._started:
            self.start_server()
        deadline = time.monotonic() + window_s
        per_peer: Dict[str, List[float]] = {a: [] for a in self._peers if a != self._local}

        while time.monotonic() < deadline:
            for peer, stub in self._stubs.items():
                if len(per_peer[peer]) >= num_pings:
                    continue
                t_before = time.time() * 1000.0
                try:
                    stub.Ping(pb2.PingRequest(sender=self._local, t_send_ms=t_before), timeout=2.0)
                    t_after = time.time() * 1000.0
                    rtt = max(0.0, t_after - t_before)
                    per_peer[peer].append(rtt / 2.0)
                except Exception:
                    continue
            if per_peer and all(len(per_peer[p]) >= num_pings for p in per_peer):
                break

        max_ow = 0.0
        for samples in per_peer.values():
            if samples:
                max_ow = max(max_ow, max(samples))
        self._max_one_way_ms = max_ow

        self._cal_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._cal_dir / f"measured_delta_max_ms_{self._local}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "local_agent_id": self._local,
                    "max_observed_one_way_ms": max_ow,
                    "per_peer_samples": {k: v[:20] for k, v in per_peer.items()},
                    "cfg_delta_max_ms": cfg_delta_max_ms,
                },
                f,
                indent=2,
            )
        return max_ow

    def send(self, sender: str, receiver: str, payload: Any, current_time_ms: float) -> None:
        if receiver not in self._stubs:
            return
        stub = self._stubs[receiver]
        env = pb2.Envelope(
            sender=sender,
            receiver=receiver,
            payload=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
            send_time_ms=float(current_time_ms),
        )
        try:
            stub.Send(env, timeout=5.0)
        except Exception:
            pass

    def poll_deliveries(self, current_time_ms: float) -> List[PendingMessage]:
        out: List[PendingMessage] = []
        with self._inbox_lock:
            while self._inbox:
                m = self._inbox.popleft()
                out.append(
                    PendingMessage(
                        sender=m.sender,
                        receiver=m.receiver,
                        payload=m.payload,
                        send_time_ms=m.send_time_ms,
                        delivery_time_ms=float(current_time_ms),
                    )
                )
        return out
