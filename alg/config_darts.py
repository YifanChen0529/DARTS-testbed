"""
DARTS framework configuration for the Time-Critical Coordinated Target Retrieval mission.
Parameters align with the DARTS draft (Algorithm 2, Eqs 24–25) and standard MARL layout.

LIO-style bilevel structure:
  θ (theta): Execution policy parameters
  η (eta):   Bidding policy parameters
  φ (phi):   Incentive policy parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DARTS_CONFIG:
    """
    Central configuration for DARTS simulation and training.
    Mission: Time-Critical Coordinated Target Retrieval (factory / coordinated transport).
    """

    # =============================================================================
    # Mission & real-time system
    # =============================================================================
    num_agents: int = 4
    """Number of heterogeneous agents in the team."""

    workload_id: str = "B1_CDR"
    """Benchmark workload ID. One of B1_CDR, B2_LML, B3_IQI, B4_DEM."""

    delta_max_ms: float = 50.0
    """Maximum one-way network latency (ms). Used in feasibility and effective WCET (C̃ + h ≤ Δ_max)."""

    deployment_mode: str = "simulator"
    """One of ``{"simulator", "distributed_jetson"}``. In ``simulator``, all agents share one process
    and use :class:`envs.simulation_env.NetworkSimulator` (synthetic Δ_max). In ``distributed_jetson``,
    one DARTS process runs per Jetson Orin Nano, communicates via :class:`envs.jetson_grpc_transport.JetsonGrpcTransport`,
    and Δ_max should match empirically measured one-way latency after calibration."""

    jetson_endpoints: Dict[str, str] = field(
        default_factory=lambda: {
            "agent_1": "192.168.0.101:50051",
            "agent_2": "192.168.0.102:50051",
            "agent_3": "192.168.0.103:50051",
            "agent_4": "192.168.0.104:50051",
        }
    )
    """Per-agent Jetson ``host:port`` when ``deployment_mode == "distributed_jetson"``. One agent id per Jetson."""

    local_agent_id: Optional[str] = None
    """Set on each Jetson (``--local_agent_id`` or env ``DARTS_AGENT_ID``). Required for ``distributed_jetson``;
    must be a key of ``jetson_endpoints``."""

    coordinator_endpoint: str = "192.168.0.100:50060"
    """``host:port`` of the :mod:`alg.coordinator` ``TeamCoordinator`` gRPC service (calibration / episode reports)."""

    tau_timeout_ms: float = 100.0
    """Heartbeat timeout (ms) for crash-stop robustness (Eq. 49). Must be > delta_max_ms."""

    epoch_length_ms: float = 1000.0
    """Epoch length H (ms): auction and readiness update interval. Theorem 1(ii) guarantees RT(v) ≤ H for DARTS."""

    e_safe_wh: float = 0.05
    """Hardware safety margin (Wh) for battery — Eq. 47 energy feasibility."""

    # Scaled for 180s mission horizon; preserves 1:1.5:2:2.5 heterogeneity ratio.
    # Agent 1 (0.50 Wh) survives ~240s at P_moving=7.5 W (margin over 150s deadline).
    # Tight enough to stress energy mask while preventing systematic crash-stop depletion.
    battery_capacities_wh: List[float] = field(default_factory=lambda: [0.50, 0.75, 1.00, 1.25])
    """Per-agent battery capacities (Wh). ~3–8 min driving in 2-min episodes to stress energy mask."""

    global_deadline_ms: float = 150000.0
    """Global end-to-end deadline D_G for the mission (ms). 150 s for TurtleBot3 @ 0.22 m/s (critical path + travel)."""

    # --- Task graph: WCET per node (ms) — physical movement + execution (TurtleBot3 scale) ---
    # A: Reconnaissance, B: Debris Clearance, C: Payload Securing, D: Coordinated Transport
    task_wcets: Dict[str, float] = field(
        default_factory=lambda: {
            "A": 15000.0,
            "B": 25000.0,
            "C": 25000.0,
            "D": 30000.0,
        }
    )

    # =============================================================================
    # Simulation & episodes (per-episode / single-run)
    # =============================================================================
    max_simulation_steps: int = 1500
    """Total simulation steps (ms) per episode (discrete-event sim)."""

    max_episode_steps: int = 200
    """Max steps per episode in TurtleBot3Env (spatial MARL loop)."""

    num_episodes: int = 20000
    """Number of episodes for a single run (e.g. long training or evaluation loop)."""

    # =============================================================================
    # Global training schedule (Algorithm 2 – outer-loop)
    # =============================================================================
    train_episodes: int = 20000
    """Total number of outer-loop training episodes (Algorithm 2, for k = 0, 1, ...)."""

    eval_every: int = 500
    """Run evaluation every X training episodes (e.g. 500 for 20k run)."""

    eval_episodes: int = 10
    """Number of test episodes to average during each evaluation."""

    # =============================================================================
    # Policy training (Algorithm 2, Step 6 – target J^pol_i, Eq (24))
    # J^pol_i(θ_i, η_i) = E_τk [ Σ_t γ^t r̃_i(t) ]
    # =============================================================================
    policy_train_steps: int = 5
    """Step length: number of gradient updates for θ (execution) and η (bidding) per outer iteration."""

    lr_theta: float = 3e-4
    """Learning rate for (bid) policy parameters θ — step-level continuous bid."""

    lr_eta: float = 3e-4
    """Learning rate for bidding net η (auction-time continuous bid)."""

    use_continuous_bid: bool = True
    """If True, policy and bidding nets output continuous bid in [0, 1]; no discrete action heads."""

    bid_sigma: float = 0.1
    """Gaussian std for continuous bid exploration (policy_net step-level bid)."""

    bid_concentration: float = 5.0
    """Concentration for Beta bid distribution (explore=True). Higher = less noise; 5.0 gives more exploration (σ≈0.14 at mu=0.5)."""

    # =============================================================================
    # Incentive training (Algorithm 2, Step 11 – target J^inc_s, Eq (25))
    # J^inc_s(φ_s) = E_τ̂k [·] − λ_P E_τk [ Σ_t γ^t ∥p_{s→·}(t)∥_1 ]
    # =============================================================================
    incentive_train_steps: int = 1
    """Step length: number of gradient updates for φ (incentive) per outer iteration."""

    lr_phi: float = 1e-4
    """Learning rate for incentive policy parameters φ (π^p_φ_s)."""

    incentive_sigma: float = 0.1
    """Gaussian std for stochastic incentive policy in Phase B (Eq. 25 log_prob)."""

    # =============================================================================
    # Discounting and logging
    # =============================================================================
    gamma: float = 0.99
    """Discount factor γ used in Eq (24) and Eq (25) (Σ_t γ^t ...)."""

    # --- Shaped reward (Eq. 22): scaled for physical ms/Wh ---
    reward_task_completion: float = 10.0
    """Reward for successfully picking up a targeted subtask (env step return)."""
    lambda_D: float = 0.001
    """Weight for deadline cost (Eq. 20). 10,000 ms delay → -10.0 penalty, balancing task reward."""
    lambda_E: float = 10.0
    """Weight for energy cost (Eq. 21). 0.5 Wh → -5.0 penalty (energy efficiency)."""
    reward_step_penalty: float = -0.01
    """Per-step penalty to discourage idling and encourage fast completion."""

    # --- TurtleBot3 / Orin Nano (Eq. 21) — match envs.turtlebot3_spatial_env ---
    p_idle_w: float = 2.5
    p_moving_w: float = 7.5
    p_comm_w: float = 0.5

    # --- Agent queue state normalization ---
    max_queue_size: int = 10
    max_work_ms: float = 142500.0
    """Max total remaining work (ms) for normalization. Scaled with global_deadline (≈ sum WCETs × 1.5) for stable MARL state."""

    lambda_P: float = 0.01
    """Weight for payment penalty in J^inc (Eq. 25): λ_P E_τk [ Σ_t ∥p_{s→·}(t)∥_1 ]."""

    lambda_rho: float = 1.0
    """Coverage weight in Eq. (35) and Eq. (36). Must be large enough that Eq. (91) holds:
    lambda_rho^2 >= 4n(G_R^2 + G_P^2) / (Theta_min * min(Lambda_eta, Lambda_phi)).
    Default 1.0 is a starting point; sweep in {0.5, 1.0, 2.0, 5.0} for the proof's regime."""

    lipschitz_rho_estimate: float = 10.0
    """L_rho estimate (Lipschitz smoothness of expected coverage map). Used to enforce
    proof choice (1): L_rho * lr_eta * lambda_rho <= 1/4 and L_rho * lr_phi * lambda_rho <= 1/4."""

    gradient_clip_norm: float = 1.0
    """Global-norm gradient clip. Caps |grad_eta R^pol_i| <= G_R and |grad_phi R^inc_i| <= G_P
    per Assumption (A3a) of Theorem 1."""

    lambda_B: float = 2.0
    """Weight for winner bid cost in shaped reward (Eq. 22). Winner pays λ_B * bid_{i,v}.
    Set to create meaningful trade-off with task completion reward (10.0)."""

    results_dir: str = "results"
    """Base directory for saving logs, checkpoints, and TensorBoard runs."""

    save_interval: int = 500
    """Save PyTorch model checkpoints every this many episodes."""

    # =============================================================================
    # Reproducibility & logging
    # =============================================================================
    random_seed: int = 42
    """Random seed for environment and agent sampling (reproducibility)."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    disable_feasibility_mask: bool = False
    """If True, runs the DARTS-NoCert ablation: bidding proceeds without the
    schedulability certificate. All cert_seq_{i,v}(t_k) are forced to 1, allowing
    every agent to bid on every ready job regardless of EDF feasibility. This
    isolates the contribution of the cert-masking step. NEVER enable this in
    deployed DARTS — only used for the paper's ablation study."""

    def __post_init__(self) -> None:
        # Proof choice (1): L_rho * lr * lambda_rho <= 1/4 for both stepsizes (Theorem 1, Section C).
        bound_eta = self.lipschitz_rho_estimate * self.lr_eta * self.lambda_rho
        bound_phi = self.lipschitz_rho_estimate * self.lr_phi * self.lambda_rho
        eps = 1e-9
        if bound_eta > 0.25 + eps:
            lim = 0.25 / max(self.lipschitz_rho_estimate * self.lambda_rho, 1e-12)
            lim_rho = 0.25 / max(self.lipschitz_rho_estimate * self.lr_eta, 1e-12)
            raise ValueError(
                f"Theorem 1 proof choice (1) violated: L_rho * lr_eta * lambda_rho = "
                f"{bound_eta:.4f} > 0.25. Reduce lr_eta to <= {lim:.2e} "
                f"or reduce lambda_rho to <= {lim_rho:.4f}."
            )
        if bound_phi > 0.25 + eps:
            raise ValueError(
                f"Theorem 1 proof choice (1) violated: L_rho * lr_phi * lambda_rho = "
                f"{bound_phi:.4f} > 0.25. Reduce lr_phi or lambda_rho."
            )


# Use as: cfg = DARTS_CONFIG(); cfg.num_agents, cfg.lr_theta, cfg.train_episodes, etc.
