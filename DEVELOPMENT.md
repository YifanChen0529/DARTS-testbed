> **Development README.** Artifact evaluators: start at [README.md](README.md).
> This file documents repository internals and the development (Jetson/conda) environment.

# DARTSrt

**A real-time MARL framework with bounded auction latency and end-to-end schedulability** for distributed, precedence-constrained DAG scheduling on mobile robot teams.

DARTSrt assigns precedence-constrained mission-graph jobs to heterogeneous mobile agents and orders them locally to meet end-to-end deadlines. Unlike prior learned allocators, the timing contract is carried by the **framework**, not the learned policy: every guarantee is policy-independent and holds even for an untrained network. Training only improves the *quality* of certified assignments, never deadline safety.

This repository contains the simulator (used for all training/ablation sweeps), the four-TurtleBot3 + Jetson distributed deployment, and the CEA-EDF / SADCHER baselines.

---

## Overview

The workload is a mission DAG `G = (V, E)`: jobs are released by sensing events or upstream completions and must meet a common end-to-end deadline, executing on mobile agents whose travel and energy costs are coupled to allocation order. Scheduling decisions are taken at periodic epoch boundaries of length `H` (default 1000 ms).

DARTSrt redesigns the MARL assignment-and-scheduling loop around four mechanisms:

| | Mechanism | Closes | Code |
|---|---|---|---|
| **M1** | Safety-gated ready set: physical execution gated by *certified* safe predecessors, so learned graph inference cannot corrupt deadline safety. | Learning-induced precedence safety | `darts_agent.py` (`LocalDAGTracker`) |
| **M2** | Sequential Feasibility Certificate: per-agent admission predicate that re-sorts the deadline-ordered queue and recomputes chain-dependent travel costs online; admits a job only if every queued job stays EDF-feasible and within battery reserve. | Trajectory-dependent execution cost; certified admissible assignments | `darts_agent.py` (cert / EDF queue) |
| **M3** | Bounded certified auction: one subround per ready job over the admissible-bidder set; calibrated subround WCET `δauc` gives a closed per-epoch bound `T̄auc = Ā·δauc`. | Unbounded assignment latency | `darts_agent.py`, `defer_fallback.py` |
| **M4** | Held-fixed two-trajectory bilevel update: the incentive (sponsor-payment) gradient is evaluated with the recipient bidding policy frozen, isolating the payment's causal effect. | Confounded incentive gradient | `train_darts.py` (`--bilevel_mode fixed_eta`) |

**Learned policies.** Navigation is strictly EDF-driven (no learned execution policy). Two policy families are trained: the bidding policy **η** (inner) and the sponsor incentive policy **φ** (outer), plus a graph-inference model **ψ**. Training follows Algorithm 2: Trajectory A updates `η` and `ψ`; Trajectory B updates `φ` with `η` held fixed.

**Guarantees** (proved in the paper, instrumented in the logs):
- **Thm 1 (aperiodic):** (i) policy-independent deadline + battery-reserve safety; (ii-a) unconditional allocation-latency bound `Λv ≤ 2H + T̄auc`; (ii-b) admission-conditional per-job response-time identity; (iii) end-to-end graph-level schedulability bound; (iv) non-decreasing certified coverage under training.
- **Thm 2 (sporadic):** composes the per-admission certificate with classical multiprocessor tests — partitioned per-agent EDF density (`δ_sum^i ≤ 1`), with a homogeneous global-EDF corollary.

---

## Repository layout

```
.
├── alg/                      # Algorithms, training, baselines, deployment entry points
│   ├── config_darts.py       # DARTS_CONFIG (epochs, battery, λ weights, Thm-1 step-size guards)
│   ├── config_sadcher.py     # SADCHER baseline config
│   ├── darts_agent.py        # DartsMARLAgent: M1 ready set, M2 cert + EDF queue, M3 bidding
│   ├── coverage.py           # CoverageStats: certified coverage ρ̂ (Thm 1(iv))
│   ├── defer_fallback.py     # Bounded defer / fallback / drop state machine (N1–N3)
│   ├── train_darts.py        # ENTRY: simulator training/eval loop (Algorithm 2)
│   ├── baseline_cea_edf.py   # ENTRY: Centralized Energy-Aware EDF baseline
│   ├── baseline_sadcher.py   # ENTRY: SADCHER (GAT+Transformer) baseline, RL-retrained
│   ├── sadcher_agent.py       # SADCHER allocator network
│   ├── stress_profiles.py    # Per-benchmark light/medium/heavy stress overrides
│   ├── run_all_experiments_{light,medium,heavy}.py  # Simulator sweep drivers
│   ├── run_eval_matrix.py    # Distributed (hardware) one-episode-per-tuple eval driver
│   ├── coordinator.py        # ENTRY: episode-lifecycle orchestrator (distributed_jetson)
│   ├── agent_node.py         # ENTRY: one agent process per Jetson (distributed_jetson)
│   └── distributed_sync.py   # Peer-to-peer epoch barrier (EPOCH_READY gossip)
├── envs/                     # Environment, task model, transport
│   ├── workloads.py          # DBench-4 registry: B1_CDR, B2_LML, B3_IQI, B4_DEM
│   ├── dag_task.py           # DAGGraph, TaskNode
│   ├── simulation_env.py     # Global clock + NetworkSimulator (synthetic Δ_max)
│   ├── turtlebot3_spatial_env.py  # TurtleBot3Env: EDF-driven 2D spatial MARL loop
│   ├── messaging_transport.py     # Transport protocol (simulator | gRPC)
│   ├── jetson_grpc_transport.py   # gRPC transport between Jetsons
│   ├── eval_node_mapping.py
│   └── proto/                # darts_coordinator / darts_messaging .proto + generated pb2
├── scripts/                  # Sweep launchers + analysis (tmux, ntfy, summarize)
│   ├── sweep_*.sh
│   └── summarize_sweep.py    # Aggregate results → sweep_summary_{per_cell,aggregated}.csv
├── results/                  # (created at runtime) logs, checkpoints, TensorBoard
└── requirements.txt
```

All scripts assume the project root is on `PYTHONPATH` (`PYTHONPATH=.`).

---

## Installation

```bash
conda create -n darts python=3.10 -y
conda activate darts
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `tensorboard`, `grpcio`, `protobuf`, `gym`, `PyYAML`. The same stack runs on Ubuntu and on NVIDIA Jetson Orin Nano (CUDA wheels). gRPC and the generated `envs/proto/*_pb2.py` are required for `distributed_jetson` mode.

---

## Quickstart (simulator)

Train DARTSrt on B1 (Coordinated Disaster Response, diamond fork-join) for a short debug run:

```bash
PYTHONPATH=. python alg/train_darts.py \
    --exp_name darts_b1_debug \
    --workload B1_CDR \
    --num_episodes 200 \
    --bilevel_mode fixed_eta
```

> **Use `--bilevel_mode fixed_eta` to reproduce the paper.** `fixed_eta` is the held-fixed two-trajectory update of mechanism M4 (Thm 1(iv)). The default `legacy` rolls Trajectory B under the post-update `η̂` and is kept only for backward comparison.

Outputs land in `results/<exp_name>/`: `darts_log.csv` (per-eval metrics), `config.json`, and checkpoints.

### Ablations

| Variant | Command flag | Effect |
|---|---|---|
| **NoCert** | `--disable_feasibility_mask` | Forces `cert_seq = 1` for all (agent, job); bidding proceeds without the schedulability certificate. Isolates M2. **Never enable in deployment.** |
| **NoIncentive** | `--no_incentive` | Freezes `φ ≡ 0` throughout training. Isolates the incentive channel from the cert+EDF contribution. |

### Baselines

```bash
# Centralized Energy-Aware EDF — same cert (Eq. 11) + EDF queue, greedy centralized assignment
PYTHONPATH=. python alg/baseline_cea_edf.py --exp_name cea_b1 --workload B1_CDR

# SADCHER — GAT+Transformer MRTA allocator, RL-retrained with the same shaped reward; no RT guarantees
PYTHONPATH=. python alg/baseline_sadcher.py --exp_name sadcher_b1 --workload B1_CDR
```

CEA-EDF and SADCHER share DARTSrt's shaped reward and (for CEA-EDF) its certificate and local EDF scheduler, so the measured gap reflects the learned auction, not a difference in admission constraints or queue discipline.

---

## Benchmarks (DBench-4)

All four benchmarks share the same physical testbed (four TurtleBot3 + four Jetson Orin Nano, 10 m × 10 m workspace, battery capacities `(0.50, 0.75, 1.00, 1.25)` Wh, certified active-power bound `Pactive = 8.0 W`). They vary only the mission DAG and release pattern.

| `--workload` | Scenario | Topology | Release | Stress levels |
|---|---|---|---|---|
| `B1_CDR` | Coordinated Disaster Response | Diamond fork-join `A → {B, C} → D` | Aperiodic single-shot | light / medium / heavy |
| `B2_LML` | Last-Mile Autonomous Logistics | Linear 4-node chain `Pk → Hf → Dr → Cf` | Sporadic Poisson (mean 30 s @ medium) | light / medium / heavy |
| `B3_IQI` | Sequential Material Inspection | Linear chain `A → B → C → D` | Aperiodic single-shot | medium only |
| `B4_DEM` | Distributed Field Survey | 4 independent jobs `{J1..J4}` | Aperiodic single-shot | medium only |

Each registered node has a measurable fallback (`C_FB = 0.4 × WCET`), so Algorithm 1 case **N1** (undefined fallback) is structurally empty for DBench-4. Stress overrides (deadline tightening, battery reduction, Poisson rate) are in `alg/stress_profiles.py`. Heavy stress on B1 tightens deadlines by 1.6× and reduces battery to 0.55 Wh, calibrated to drive the CEA-EDF baseline across a 50% miss rate.

---

## Reproducing the paper sweeps

The full method × benchmark × stress × seed matrix is driven by the `run_all_experiments_*.py` scripts. Methods: `darts`, `darts_nocert`, `darts_no_incentive`, `cea_edf`, `sadcher`. `darts_nocert` is gated to B1/B2 only (degenerate elsewhere).

```bash
# Single stress level, all methods, all applicable benchmarks
PYTHONPATH=. python alg/run_all_experiments_medium.py \
    --methods darts,darts_nocert,darts_no_incentive,cea_edf,sadcher \
    --num_episodes 5000

# Resume an interrupted sweep (skips cells whose result CSV already exists)
PYTHONPATH=. python alg/run_all_experiments_medium.py --resume ...
```

A reference 5-pane tmux launcher (medium + light + heavy, paper-critical cells) is in `scripts/sweep_d1.sh`. Results are written to `results/exp_<idx>/<B#>/<stress>/<method>_log.csv`; aggregate them with:

```bash
python scripts/summarize_sweep.py
# → sweep_summary_per_cell.csv, sweep_summary_aggregated.csv
```

**Headline result** (B1, heavy stress, 10 seeds, convergence window): DARTSrt sustains a **17.0%** deadline-miss rate where SADCHER reaches **64.5%** and CEA-EDF **61.0%**, with **zero** response-time-bound violations across all evaluation cells and P99 auction latency inside the analytical `|Ak|·δauc` envelope (`δauc = 30 ms` calibrated).

---

## Hardware deployment (`distributed_jetson`)

Each agent runs as a distinct OS process on its own Jetson Orin Nano, driving a TurtleBot3 over a ROS 2 Humble / OpenCR bridge and communicating with peers over an isolated 802.11n WLAN via gRPC. One coordinator process (any host) orchestrates the episode lifecycle but does **not** participate in auctions.

```bash
# 1) Coordinator (one host)
PYTHONPATH=. python alg/coordinator.py --workload B1_CDR --exp_name hw_run1 --num_episodes 100

# 2) One agent process per Jetson (set --local_agent_id per board)
PYTHONPATH=. python alg/agent_node.py --local_agent_id agent_1 --workload B1_CDR --exp_name hw_run1
# ... agent_2 / agent_3 / agent_4 on their boards

# 3) Or drive a small (benchmark, stress, seed) eval matrix once agents are up
PYTHONPATH=. python alg/run_eval_matrix.py --exp_prefix hw_matrix
```

Set `host:port` endpoints in `DARTS_CONFIG.jetson_endpoints` / `coordinator_endpoint`. The one-way latency bound `Δ_max` is calibrated empirically at startup (each Jetson exchanges 100 pings per peer over a 5 s window; the max observed one-way latency sets `cfg.delta_max_ms`), so the bounded-latency assumption underlying Thm 1 is grounded rather than synthetic. Training itself only supports `simulator`.

---

## Outputs and metrics

Per-eval rows in `*_log.csv` (header in `train_darts.EVAL_CSV_HEADER`) include:

| Column | Meaning |
|---|---|
| `task_miss_rate` | End-to-end mission deadline-miss rate (headline safety metric) |
| `rho_hat` | Certified full-fidelity coverage ρ̂ (Thm 1(iv)) |
| `rt_bound_violations` | Response-time-bound violations — **should be 0 for DARTSrt** |
| `max_t_auc_ms`, `mean_t_auc_ms` | Per-episode auction-subround latency (Thm 1(ii-a) envelope) |
| `max_alloc_delay_ms` | Worst-case allocation delay `Λv` |
| `A{1..4}_crash_rate` | Per-agent battery-exhaustion / crash-stop rate |
| `sub{A..D}_miss_rate`, `sub{A..D}_rt_ms` | Per-subtask miss rate and response time |

---

## Citation

DARTSrt is under submission. Until publication, please cite the repository and the draft:

```bibtex
@misc{dartsrt,
  title  = {DARTSrt: A Real-Time MARL Framework with Bounded Auction Latency and Schedulability},
  note   = {Under submission},
  year   = {2026}
}
```

See `LICENSE` for terms; refer to the paper for the full theoretical framework and proofs.
