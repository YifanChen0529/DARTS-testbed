"""
Pluggable messaging layer for DARTS: same agent code runs with an in-process simulator
or with real gRPC between Jetson Orin Nano nodes.

The :class:`MessagingTransport` protocol matches the public API of
:class:`envs.simulation_env.NetworkSimulator` (``agent_ids``, ``send``, ``poll_deliveries``).

**Simulator mode (``deployment_mode == "simulator"``):** all agents share one Python process;
messages use :class:`envs.simulation_env.NetworkSimulator` with synthetic one-way latency in
``[MIN_DELAY_MS, DELTA_MAX]`` (training default).

**Distributed Jetson mode:** one OS process per agent; messages use
:class:`envs.jetson_grpc_transport.JetsonGrpcTransport` with empirically calibrated one-way
latency (see ``results/<exp_name>/measured_delta_max_ms.json`` and team-wide check vs
``cfg.delta_max_ms``).

---------------------------------------------------------------------------
Paper (Section I-B-d, communication latency Δ_max) — empirical calibration footnote
---------------------------------------------------------------------------

In our hardware deployment (Section V), Δ_max is calibrated empirically: at startup each
Jetson exchanges 100 ping messages with every peer Jetson over a 5-second window; the maximum
observed one-way latency across all (sender, receiver) pairs sets ``cfg.delta_max_ms``. The
feasibility certificate (Eq. 11) is recomputed at every epoch using this calibrated value, so
the bounded-latency assumption underlying Theorem 1 is empirically grounded rather than synthetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from envs.simulation_env import PendingMessage
else:
    PendingMessage = object  # type: ignore[misc, assignment]


@runtime_checkable
class MessagingTransport(Protocol):
    """Inter-agent message transport (simulator or distributed gRPC)."""

    @property
    def agent_ids(self) -> List[str]:
        ...

    def send(self, sender: str, receiver: str, payload: Any, current_time_ms: float) -> None:
        ...

    def poll_deliveries(self, current_time_ms: float) -> List[Any]:
        ...
