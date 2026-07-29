"""
Peer-to-peer epoch synchronization for ``deployment_mode == "distributed_jetson"``.

Each node broadcasts ``EPOCH_READY`` with a discrete ``epoch_slot``; auctions proceed only after
``EpochReady`` has been received from all peers or ``tau_timeout_ms`` elapses (Section I-G).
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Set

EPOCH_READY = "EPOCH_READY"


def _route_non_barrier_messages(agents: List[Any], messages: List[Any]) -> None:
    for m in messages:
        pl = getattr(m, "payload", None)
        if isinstance(pl, dict) and pl.get("type") == EPOCH_READY:
            continue
        recv = getattr(m, "receiver", None)
        for agent in agents:
            tracker = getattr(agent, "dag_tracker", None)
            if tracker is None or not hasattr(tracker, "process_incoming_messages"):
                continue
            if recv == getattr(agent, "agent_id", None):
                tracker.process_incoming_messages([m])


def sync_epoch_barrier(
    network: Any,
    cfg: Any,
    local_agent_id: str,
    epoch_slot: int,
    poll_time_ms: float,
    agents: List[Any],
) -> None:
    """
    Gossip barrier: send ``EPOCH_READY`` to every peer, then drain ``poll_deliveries`` until
    all peers' ready messages for ``epoch_slot`` are seen (or ``tau_timeout_ms`` timeout).
    """
    agent_ids: List[str] = list(getattr(network, "agent_ids", []) or [])
    peers = [a for a in agent_ids if a != local_agent_id]
    if not peers:
        return

    payload = {"type": EPOCH_READY, "agent_id": local_agent_id, "epoch_slot": int(epoch_slot)}
    for p in peers:
        network.send(local_agent_id, p, payload, float(poll_time_ms))

    tau_s = float(getattr(cfg, "tau_timeout_ms", 100.0)) / 1000.0
    deadline = time.monotonic() + max(tau_s, 0.05)
    got: Set[str] = set()

    while time.monotonic() < deadline:
        batch = network.poll_deliveries(float(poll_time_ms))
        if not batch:
            time.sleep(0.001)
            continue
        leftovers: List[Any] = []
        for m in batch:
            pl = getattr(m, "payload", None)
            if (
                isinstance(pl, dict)
                and pl.get("type") == EPOCH_READY
                and int(pl.get("epoch_slot", -1)) == int(epoch_slot)
            ):
                sender = getattr(m, "sender", None)
                if isinstance(sender, str) and sender in peers:
                    got.add(sender)
            else:
                leftovers.append(m)
        if leftovers:
            _route_non_barrier_messages(agents, leftovers)
        if len(got) >= len(peers):
            return

    raise RuntimeError(
        f"Epoch barrier timeout: epoch_slot={epoch_slot} local={local_agent_id!r} "
        f"received_ready_from={sorted(got)} expected_peers={peers} "
        f"(τ_timeout={getattr(cfg, 'tau_timeout_ms', 100.0)} ms > Δ_max; check network or raise τ_timeout)."
    )
