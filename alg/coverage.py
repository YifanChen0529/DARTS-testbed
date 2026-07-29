"""
Per-trajectory coverage statistics for DARTS Eqs. (28)–(33), (35), (36).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class CoverageStats:
    """Per-trajectory coverage statistics from Eqs. (28)-(33) of the paper."""

    V_total: int
    V_ff: Set[str]
    V_sp: Dict[int, Set[str]]
    V_ff_per_sponsor: Dict[int, Set[str]]

    @property
    def rho_hat(self) -> float:
        return len(self.V_ff) / max(1, self.V_total)

    def omega_i(self, agent_idx: int) -> float:
        return len(self.V_sp.get(agent_idx, set())) / max(1, self.V_total)

    def rho_sp_i(self, agent_idx: int) -> float:
        sponsored = self.V_sp.get(agent_idx, set())
        return len(self.V_ff_per_sponsor.get(agent_idx, set())) / max(1, len(sponsored))


class CoverageTracker:
    """Records sponsorship and full-fidelity commitments across one trajectory."""

    def __init__(self, V_total: int, num_agents: int) -> None:
        self.V_total = V_total
        self.num_agents = num_agents
        self.V_ff: Set[str] = set()
        self.V_sp: Dict[int, Set[str]] = {i: set() for i in range(num_agents)}
        self.V_ff_per_sponsor: Dict[int, Set[str]] = {i: set() for i in range(num_agents)}

    def record_sponsor(self, sponsor_idx: int, job_id: str) -> None:
        """Call when the auction for ``job_id`` opens and the sponsor is fixed (Eq. (15))."""
        if 0 <= sponsor_idx < self.num_agents:
            self.V_sp[sponsor_idx].add(str(job_id))

    def record_full_fidelity_commit(self, sponsor_idx: int, job_id: str) -> None:
        """
        Mark a job committed in full fidelity (not Eq. 23 fallback). Adds to ``V_ff`` and
        ``V_ff_per_sponsor[sponsor_idx]`` per Eq. (28).
        """
        jid = str(job_id)
        self.V_ff.add(jid)
        if 0 <= sponsor_idx < self.num_agents:
            self.V_ff_per_sponsor[sponsor_idx].add(jid)

    def freeze(self) -> CoverageStats:
        """Immutable snapshot for loss computation."""
        return CoverageStats(
            V_total=self.V_total,
            V_ff=set(self.V_ff),
            V_sp={i: set(s) for i, s in self.V_sp.items()},
            V_ff_per_sponsor={i: set(s) for i, s in self.V_ff_per_sponsor.items()},
        )
