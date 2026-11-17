"""Canonical CSV schemas for the TB (timing-budget) sweep era.

Two headers live here:

``SUBROUND_LOG_HEADER``
    The single source of truth for ``<exp_dir>/subround_log.csv``. ``alg.train_darts``
    imports this symbol, so the file written on disk cannot drift from this literal.

``MAIN_CSV_HEADER_TB76``
    A *copy* of ``alg.train_darts.EVAL_CSV_HEADER`` at 76 columns, kept here so analysis
    code can pin the expected width without importing torch. It is deliberately a copy
    rather than an import: ``tests/test_csv_schema_tb.py`` asserts the two agree, which
    turns any accidental edit to either list into a test failure.

Schema eras of ``darts_log.csv`` (header-keyed readers must never assume a width):

* 62 columns — retired. Carried ``cert_attempts`` / ``cert_rejects`` / ``cert_reject_rate``.
  Those three names must NOT be reused; the TB-era bid counters are deliberately named
  ``n_bids_evaluated`` / ``n_bids_cert_pass`` so header-keyed cross-era analysis cannot
  silently mix the two definitions.
* 63 columns — the AE-validated era, ending at ``n_miss_full_fidelity_late``.
* 76 columns — the TB era: the 63 AE columns unchanged and in place, plus 13 appended
  at the end. TB sweeps write to fresh result dirs, so no file ever mixes widths.

Column semantics for the 13 TB additions
----------------------------------------
``n_bids_evaluated``
    Σ over subrounds of the number of alive agents polled for a bid.
``n_bids_cert_pass``
    Σ over subrounds of ``|C_v|``, the bids that cleared the certificate. Bounded above
    by ``n_bids_evaluated``.
``mean_pool_size``
    Mean ``|C_v|`` over subrounds; nan when the episode ran no subround.
``n_ff_missed``
    Certified-miss numerator: jobs committed at full fidelity by the auction that then
    completed after ``d_v`` or never completed. Eq. 23 fallback commits are excluded by
    construction (``CoverageTracker.V_ff`` holds auction FF commits only). Its late term
    coincides with ``n_miss_full_fidelity_late``, which is retained untouched for D1
    continuity, so ``n_ff_missed >= n_miss_full_fidelity_late`` holds row-wise.
``episode_span_ms``
    ``env.clock_ms`` when the step loop exited.
``exec_ms_A1..A4`` / ``travel_ms_A1..A4``
    Per-agent busy time, accumulated only while the agent is alive. Mutually exclusive by
    construction (``execute_task`` only runs at target, where the mover reports not
    moving), so ``Σ exec + Σ travel <= n_agents * episode_span_ms``.

Subround log semantics
----------------------
One row per auction subround (one ready job in one epoch), buffered per episode and
flushed at episode end.

``t_subround_ms``
    The wall-clock sample appended by ``_delta_auc_subround_timer`` for this subround; it
    bounds the per-subround component timings below.
``t_cert_ms`` / ``t_infer_ms``
    Summed over the polled agents: time inside ``check_feasibility`` and inside the
    ``bidding_net`` forward plus Beta sampling block, respectively.
``t_exchange_ms``
    Always 0.0 in the simulator: bids are read from in-process objects, so there is no
    wire exchange to charge. The column exists so the distributed Jetson deployment can
    populate it without another schema bump.
``t_fallback_ms``
    Time in the no-bidder branch (Algorithm 1 defer / fallback classification). 0.0 on
    committing subrounds.
``t_commit_ms``
    Time from winner selection through ``assign_job`` and its bookkeeping. 0.0 on
    non-committing subrounds.
``outcome``
    One of ``commit``, ``defer``, ``fallback_now``, ``n1_drop``, ``n2_casualty``.
``winner_idx``
    0-based winning agent on ``commit``, the sponsor on the fallback outcomes, and -1
    where no agent took the job.
``queue_len_per_agent``
    JSON list of ``len(agent.queue)`` captured at row-emit time.
"""

from __future__ import annotations

from typing import List

SUBROUND_LOG_HEADER: List[str] = [
    "episode",
    "phase",
    "clock_ms",
    "job_id",
    "subround_pos_j",
    "A_k_size",
    "n_alive",
    "C_v_size",
    "t_subround_ms",
    "t_cert_ms",
    "t_infer_ms",
    "t_exchange_ms",
    "t_fallback_ms",
    "t_commit_ms",
    "queue_len_per_agent",
    "winner_idx",
    "outcome",
]

MAIN_CSV_HEADER_TB76: List[str] = [
    "episode",
    "A1_total_reward", "A1_remaining_battery_pct",
    "A2_total_reward", "A2_remaining_battery_pct",
    "A3_total_reward", "A3_remaining_battery_pct",
    "A4_total_reward", "A4_remaining_battery_pct",
    "subA_miss_rate", "subA_time",
    "subB_miss_rate", "subB_time",
    "subC_miss_rate", "subC_time",
    "subD_miss_rate", "subD_time",
    "task_miss_rate",
    "task_time",
    "rho_hat",
    "n_full_fidelity",
    "n_total_jobs",
    "rho_sp_A1", "rho_sp_A2", "rho_sp_A3", "rho_sp_A4",
    "omega_A1", "omega_A2", "omega_A3", "omega_A4",
    "A1_crash_rate", "A2_crash_rate", "A3_crash_rate", "A4_crash_rate",
    "subA_rt_ms", "subB_rt_ms", "subC_rt_ms", "subD_rt_ms",
    "max_rt_ms", "max_alloc_delay_ms", "rt_bound_violations",
    "max_t_auc_ms", "mean_t_auc_ms",
    "max_A_k", "mean_A_k",
    "max_delta_auc_ms", "mean_delta_auc_ms",
    "t_auc_Ak1_mean_ms", "t_auc_Ak2_mean_ms",
    "t_auc_Ak3_mean_ms", "t_auc_Ak4plus_mean_ms",
    "n_epochs_Ak1", "n_epochs_Ak2",
    "n_epochs_Ak3", "n_epochs_Ak4plus",
    "n_degradable_fallback", "n_nondegradable_n1", "n_nondegradable_n2", "n_pending",
    "n_miss_n1_drop", "n_miss_n2_fallback_insufficient", "n_miss_n3_unreached",
    "n_miss_full_fidelity_late",
    "n_bids_evaluated", "n_bids_cert_pass", "mean_pool_size",
    "n_ff_missed", "episode_span_ms",
    "exec_ms_A1", "exec_ms_A2", "exec_ms_A3", "exec_ms_A4",
    "travel_ms_A1", "travel_ms_A2", "travel_ms_A3", "travel_ms_A4",
]

MAIN_CSV_N_COLS_AE63 = 63
MAIN_CSV_N_COLS_TB76 = 76

TB_APPENDED_COLUMNS: List[str] = MAIN_CSV_HEADER_TB76[MAIN_CSV_N_COLS_AE63:]

SUBROUND_OUTCOMES = ("commit", "defer", "fallback_now", "n1_drop", "n2_casualty")

# ---------------------------------------------------------------------------
# TB-3 offline harness (appended; the 76-col main log is untouched)
# ---------------------------------------------------------------------------
# ``TB3_OFFLINE_HEADER`` is the single source of truth for
# ``results_tb3_offline/<subcommand>/*.csv``. ``scripts.offline_test_harness``
# imports this symbol and never retypes it.
#
# Column semantics
# ----------------
# ``benchmark`` / ``stress``
#     DBench-4 cell, written as the workload id (B1_CDR, …) and stress profile
#     name. Together they equal ``MissionInstance.cell()``.
# ``instance_seed``
#     Literal ``nominal`` when ``MissionInstance.instance_seed is None`` (the
#     paper-era instance). Otherwise the integer seed. The generator that
#     populates a non-None value lands in TB3-2.
# ``beta``
#     Deadline tightness already folded into ``D_e2e_ms`` / ``D_rel_ms`` by
#     ``MissionInstance.scaled``. ``evaluate`` compares ``R_G_off_ms`` against
#     those stored deadlines and must not multiply by ``beta`` again.
# ``test``
#     Registry name (``S0``, later ``S1`` / ``S4``).
# ``sound``
#     Whether the bound is a sound over-approximation of the paper's test.
#     ``S0`` and ``S1`` are sound; ``S4`` (oracle ceiling) is not.
# ``xi_source``
#     ``declared`` = ``workload.task_wcets_ms`` (what ``scripts/r_off_g.py``
#     uses, and what the published T3 table embeds). ``runtime`` = the
#     ``DAGGraph.from_workload`` merge (T0 H3: B3 becomes B1's 95 kms).
# ``D_e2e_ms`` / ``D_rel_ms``
#     The two deadline references after profile and beta. ``task_miss_rate``
#     compares against D_e2e; per-job ``d_v`` uses D_rel (T0 §2.3 / H6).
# ``R_G_off_ms``
#     Offline graph-response bound. S0 is the path-sum of
#     ``2H + Ā·δ_auc + ξ_v + h + τ̄`` with raw workspace-diameter τ̄ (no 1.1 /
#     1.5). Not rounded.
# ``accept_e2e`` / ``accept_rel``
#     ``R_G_off_ms <= D_*`` (paper's test is non-strict). Written 0/1.
# ``load_e2e`` / ``load_rel``
#     ``R_G_off_ms / D_*``.
# ``A_bar`` / ``T_bar_auc_ms`` / ``tau_bar_ms`` / ``longest_path``
#     S0 decomposition. ``longest_path`` is ``->``-joined. First-max tie
#     (``>``) matches ``scripts/r_off_g.py``. For tests with per-node travel
#     envelopes, ``tau_bar_ms`` carries max_v of the per-node value; per-node
#     values live in the run report, not the CSV.
# ``delta_auc_ms`` / ``H_ms`` / ``j_bar_ms`` / ``u_mps`` / ``h_comm_ms`` /
# ``start_x`` / ``start_y``
#     The ``Envelopes`` in force (Makefile constants).

TB3_OFFLINE_HEADER: List[str] = [
    "benchmark", "stress", "instance_seed", "beta", "test", "sound",
    "xi_source", "D_e2e_ms", "D_rel_ms", "R_G_off_ms", "accept_e2e", "accept_rel",
    "load_e2e", "load_rel", "A_bar", "T_bar_auc_ms", "tau_bar_ms", "longest_path",
    "delta_auc_ms", "H_ms", "j_bar_ms", "u_mps", "h_comm_ms", "start_x", "start_y",
]

# ---------------------------------------------------------------------------
# TB-3 offline acceptance curves (grid aggregate; appended)
# ---------------------------------------------------------------------------
# ``TB3_CURVES_HEADER`` is one row per (cell, test, beta). ``n_instances`` is
# the seed pool (200 on the canonical grid). ``accept_frac_e2e`` is the
# empirical CDF of per-instance ``beta_star`` evaluated at ``beta``, where
# ``beta_star_i := R_G_off_ms / D_e2e_ms`` at beta=1 (deployed deadlines).
# ``accept_frac_rel`` is the same against D_rel. Written 0/1 counts and
# fractions; the harness imports this symbol and never retypes it.

TB3_CURVES_HEADER: List[str] = [
    "cell", "test", "beta", "n_instances",
    "n_accept_e2e", "accept_frac_e2e", "n_accept_rel", "accept_frac_rel",
]

# ``TB3_FRONTIER_HEADER`` is one row per (cell, test): the distribution of
# ``beta_star_i`` over the seed pool (min / p25 / median / p75 / max).
# ``n_S1_gt_S0`` is the number of seeds on that cell where the S1 bound
# strictly exceeds the S0 bound (same value on both test rows). Registered-
# geometry dominance does not extend to jittered instances.

TB3_FRONTIER_HEADER: List[str] = [
    "cell", "test", "n_instances", "beta_star_min",
    "beta_star_p25", "beta_star_median", "beta_star_p75", "beta_star_max",
    "n_S1_gt_S0",
]

# ---------------------------------------------------------------------------
# TB-3 empirical anchor (frozen-policy replay; appended)
# ---------------------------------------------------------------------------
# ``e2e_clean`` is 1 iff ``task_miss_rate == 0.0`` exactly on the single
# appended eval row (the D4 end-to-end miss bit). ``perjob_clean`` is 1 iff
# the four miss counters ``n_miss_full_fidelity_late``, ``n_miss_n1_drop``,
# ``n_miss_n2_fallback_insufficient``, ``n_miss_n3_unreached`` are all 0.0.
# ``feasible_e2e_all`` / ``feasible_perjob_all`` require the bit on every
# checkpoint seed (3/3). ``feasible_*_majority`` requires >= 2/3.
# Feasibility-all over 3 ckpts x 5 episodes is the primary D4 headline;
# majority and ``epfrac_e2e_clean`` (1 - mean task_miss_rate over the 3
# runs) are the epsilon-sensitivity variants. ``crash_A1..A4`` are the
# log's ``A1_crash_rate``..``A4_crash_rate`` (renamed for the table).

TB3_EMP_RUNS_HEADER: List[str] = [
    "cell", "beta", "instance_seed", "ckpt_seed", "eval_seed_base",
    "eval_episodes", "task_miss_rate", "n_miss_full_fidelity_late",
    "n_miss_n1_drop", "n_miss_n2_fallback_insufficient", "n_miss_n3_unreached",
    "n_ff_missed", "rho_hat", "rt_bound_violations", "max_rt_ms",
    "crash_A1", "crash_A2", "crash_A3", "crash_A4", "episode_span_ms",
    "e2e_clean", "perjob_clean",
]

TB3_EMP_FEAS_HEADER: List[str] = [
    "cell", "beta", "instance_seed", "n_runs",
    "feasible_e2e_all", "feasible_e2e_majority", "epfrac_e2e_clean",
    "feasible_perjob_all", "feasible_perjob_majority",
    "s0_accept_e2e", "s1_accept_e2e",
]

# ---------------------------------------------------------------------------
# TB-3 over-rejection / soundness adjudication / empirical ceiling (appended)
# ---------------------------------------------------------------------------
# ``over_rejection_hat`` = n_rej_feas / n_feasible_e2e_all among the 10
# shared instances at one (cell, deadline-scale, test). Undefined when the
# denominator is 0. Bootstrap (B=10000, ``default_rng(20260825)``) resamples
# the 10 instance ids with replacement; percentile 95% CI. Resamples with
# zero feasible instances make over-rejection undefined: they are dropped
# from that CI and counted in ``n_undefined``.
# ``verdict`` is one of ``consistent``, ``energy_scope_miss``,
# ``TIMING_VIOLATION`` (pre-registered rule; TIMING_VIOLATION can fire on
# partition residual even when FFL and rt_bound_violations are globally 0).
# Empirical-ceiling ``s*_emp`` for a feasible instance is left-censored at
# the smallest tested deadline scale where ``feasible_e2e_all`` holds, so
# ``s*_test / s*_emp`` is a lower bound.

TB3_OVERREJ_HEADER: List[str] = [
    "cell", "test", "beta", "n_instances", "n_feasible_e2e_all",
    "n_rej_feas", "over_rejection_hat", "ci_lo", "ci_hi",
    "n_undefined", "n_bootstrap",
]

TB3_ADJUDICATION_HEADER: List[str] = [
    "cell", "test", "beta", "instance_seed", "accept_e2e",
    "feasible_e2e_all", "ffl_sum", "n1_sum", "n2_sum", "n3_sum",
    "partition_residual_max", "verdict",
]

TB3_CEILING_HEADER: List[str] = [
    "cell", "beta", "n_instances", "n_feasible", "feasible_frac",
    "ci_lo", "ci_hi",
]

# TB3-6b amendment. The v1 identity treated ``task_miss_rate`` as a job
# count; the live eval row is a per-episode mean of a binary e2e miss, so
# residual == 3 * task_miss_rate iff every miss episode has exactly one
# missed job (R1). ``verdict`` / ``verdict_v1`` copy the frozen P6 rule.
# ``verdict_v2`` is the corrected rule: TIMING_VIOLATION only if accept
# and (FFL_sum > 0 or rt_bound_violations_sum > 0 or NOT R1);
# energy_scope_miss if accept and not all-clean and not that.
# ``n_miss_episodes`` = sum over the 3 runs of ``task_miss_rate * 5``
# (integer-checked). ``n3_jobs`` = ``n3_sum * 5`` on the same row
# (sum of three per-episode N3 means, times 5 eval episodes).
TB3_ADJUDICATION_V2_HEADER: List[str] = [
    "cell", "test", "beta", "instance_seed", "accept_e2e",
    "feasible_e2e_all", "ffl_sum", "n1_sum", "n2_sum", "n3_sum",
    "partition_residual_max", "verdict",
    "r1_holds", "verdict_v1", "verdict_v2", "n_miss_episodes", "n3_jobs",
]

# TB-3 P7 merged sample. ci_lo/ci_hi are exact two-sided 95% Clopper-Pearson
# (primary). boot_lo/boot_hi are percentile 95% CIs from B=10000 resamples of
# the instance ids at that (cell, scale) with default_rng(20260825); over-
# rejection drops zero-feasible resamples into n_undefined. n_instances is 50
# at deployed deadlines (seeds 0..49) and 10 at the relaxed points.
# Adjudication uses the P6b verdict_v2 rule. Frontier rows are left-censored
# at the smallest tested scale where feasible_e2e_all holds.
#
# P7b gate semantics (collect G1-G6): G5 directional identity is
# ``dev := task_miss_rate - (FFL+N1+N2+N3) <= +1e-9`` on every run
# (positive = uncancelled zero-channel miss). G6 is strict ``|dev| < 1e-9``
# only on runs whose (cell, instance, scale) is offline-accepted
# (s0_accept_e2e OR s1_accept_e2e). Strict R1 over all runs is
# informational. The partition_residual / r1_holds columns keep the P6b
# one-job-per-miss-episode meaning.

TB3_EMP_RUNS_P7_HEADER: List[str] = TB3_EMP_RUNS_HEADER + ["plan"]

TB3_EMP_FEAS_P7_HEADER: List[str] = TB3_EMP_FEAS_HEADER + ["plan"]

TB3_CEILING_P7_HEADER: List[str] = [
    "cell", "beta", "n_instances", "n_feasible", "feasible_frac",
    "ci_lo", "ci_hi", "boot_lo", "boot_hi",
]

TB3_OVERREJ_P7_HEADER: List[str] = [
    "cell", "test", "beta", "n_instances", "n_feasible_e2e_all",
    "n_rej_feas", "over_rejection_hat", "ci_lo", "ci_hi",
    "boot_lo", "boot_hi", "n_undefined", "n_bootstrap",
]

TB3_ADJUDICATION_P7_HEADER: List[str] = [
    "cell", "test", "beta", "instance_seed", "accept_e2e",
    "feasible_e2e_all", "ffl_sum", "n1_sum", "n2_sum", "n3_sum",
    "partition_residual_max", "r1_holds", "verdict_v2",
    "n_miss_episodes", "n3_jobs", "plan",
]

TB3_FRONTIER_LB_P7_HEADER: List[str] = [
    "cell", "instance_seed", "test", "s_star_test", "s_emp_ub",
    "ratio_lb", "left_censored_at",
]

# ---------------------------------------------------------------------------
# EXP-2 periodic-release sidecar (appended; the 76-col main log is untouched)
# ---------------------------------------------------------------------------
# One row per minted job instance ``class#ell``. Run-level columns are repeated
# on every row so the file is self-describing. ``alloc_map_sha_or_path`` is the
# CLI path plus a short sha256 of the parsed JSON (or empty when no map).

RELEASE_LOG_HEADER: List[str] = [
    "workload", "stress", "seed", "ckpt_dir", "ckpt_ep",
    "P_ms", "L", "D_v_ms", "per_period_energy", "alloc_map_sha_or_path",
    "release_idx", "job_class", "instance_id", "agent_id",
    "r_ms", "commit_ms", "finish_ms", "deadline_ms", "met",
    "queue_empty_at_next_boundary", "battery_at_boundary_wh", "fallback_blocked",
]

# ---------------------------------------------------------------------------
# EXP-3 CBBA microbenchmark trial log (appended; no existing header is edited)
# ---------------------------------------------------------------------------
# One row per trial (including warmups and null trials). Run-level columns are
# repeated on every row so the file is self-describing.
#
# ``t_total_ms``
#     Runner-clock wall time (``time.perf_counter``) from just before the first
#     BeginTrial RPC to the moment the last node's TrialDone is collected,
#     including runner round-trips.
# ``t_overhead_floor_ms``
#     Median null-trial ``t_total_ms`` of this session, repeated on every row
#     for self-contained analysis.
# ``per_node_compute_ms_json`` / ``per_node_blocking_ms_json``
#     Local ``perf_counter`` decompositions on each node; never differenced
#     across nodes (single-clock rule: headline T_conv is the runner clock).
# ``per_round_wall_ms_json``
#     Per-round local compute+blocking, max-aggregated across nodes as a
#     descriptive round length (not a cross-node clock difference).
# ``ref_agree``
#     1 iff the distributed run's ``rounds`` and assignment match
#     ``cbba_core.run_sync_instance`` on the identical instance.

CBBA_LOG_HEADER: List[str] = [
    "trial", "config_id", "m", "l_t", "network", "seed", "warmup", "null",
    "rounds", "rounds_ref", "ref_agree", "t_total_ms", "t_overhead_floor_ms",
    "per_round_wall_ms_json", "per_node_compute_ms_json",
    "per_node_blocking_ms_json", "msgs_total", "bytes_total", "valid", "ts_iso",
]
