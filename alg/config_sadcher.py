"""
SADCHER baseline configuration (Bichler et al., IEEE MRS 2025).
SADCHER: MARL-only multi-robot task allocation with GAT + Transformer;
no real-time guarantees (no EDF, no RTA certificates).

Mission and environment parameters are inherited from DARTS so that the same
TurtleBot3 / coordinated target retrieval setup is used for fair comparison.
"""

from dataclasses import dataclass

from alg.config_darts import DARTS_CONFIG


@dataclass(frozen=True)
class SADCHER_CONFIG(DARTS_CONFIG):
    """
    Configuration for SADCHER baseline. Inherits from DARTS_CONFIG:
    num_agents, global_deadline_ms, task_wcets, max_episode_steps,
    battery_capacities_wh, delta_max_ms, e_safe_wh, tau_timeout_ms,
    p_idle_w, p_moving_w, p_comm_w, results_dir, random_seed, log_level,
    reward_task_completion, lambda_D, lambda_E, reward_step_penalty,
    gamma, max_work_ms, max_queue_size (and all other DARTS mission/sim fields).

    SADCHER-specific hyperparameters below follow the paper's architecture
    (GAT encoder for robot and task graphs, cross-attention Transformer,
    distance MLP, reward prediction MLP, actor-critic RL).
    """

    # =============================================================================
    # Graph / embedding architecture (SADCHER paper: GAT + Transformer)
    # =============================================================================
    latent_dim: int = 64
    """Embedding dimension d for GAT and Transformer (paper: latent representation size)."""

    num_gat_heads: int = 4
    """Number of attention heads in each GAT layer (multi-head attention in robot/task encoders)."""

    num_gat_layers: int = 2
    """Number of stacked GAT layers for robot graph and task graph encoding."""

    num_transformer_layers: int = 2
    """Number of cross-attention Transformer blocks (robot–task interaction)."""

    # =============================================================================
    # Input feature dimensions (paper: robot state, task state)
    # =============================================================================
    robot_feature_dim: int = 6
    """Robot node input dim: [x_norm, y_norm, battery_norm, workload_norm, remaining_time_norm, idle_flag]."""

    task_feature_dim: int = 6
    """Task node input dim: [target_x_norm, target_y_norm, wcet_norm, precedence_state_onehot(3)] (WAITING/READY/COMPLETED)."""

    # =============================================================================
    # MLP hidden dimensions (paper: distance MLP, reward prediction head)
    # =============================================================================
    distance_mlp_hidden: int = 32
    """Hidden dimension for the distance MLP (robot–task or edge scoring)."""

    reward_mlp_hidden: int = 64
    """Hidden dimension for the reward prediction MLP (centralized critic / reward shaping)."""

    # =============================================================================
    # Training: reward net and actor-critic (paper: RL with reward prediction)
    # =============================================================================
    lr_reward_net: float = 3e-4
    """Learning rate for the reward prediction network (RL training objective)."""

    lr_critic: float = 3e-4
    """Critic learning rate for actor-critic RL."""

    entropy_coeff: float = 0.01
    """Entropy bonus coefficient for exploration in the policy objective."""

    # Training schedule: inherited from DARTS_CONFIG (train_episodes=20000, eval_every=500, eval_episodes=10) for fair comparison.

    # =============================================================================
    # Decision step and pre-moving (paper: asynchronous replanning, pre-moving)
    # =============================================================================
    decision_step_mode: str = "epoch"
    """'epoch' = recompute assignment every env step; matches SADCHER asynchronous replanning at decision steps."""

    use_pre_moving: bool = True
    """SADCHER pre-moving strategy: idle robots move toward highest-reward unassigned task before next decision."""

    def __post_init__(self) -> None:
        super().__post_init__()
