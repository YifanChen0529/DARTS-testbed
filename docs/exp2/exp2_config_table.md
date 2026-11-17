# EXP-2 configuration table (Theorem 2(a))

Frozen from `scripts/thm2a_offline_test.py` with **implementation factors ON**
(1.5× path-metric, 1.1× travel, matching `estimate_chain_wcet_ms`). The
unfactored (raw workspace-diameter) δ-sum is recorded alongside and is
informational only; admission uses the factored numbers.

Machine-readable twin: [`exp2_config_table.json`](exp2_config_table.json).
Alloc maps: [`alloc_b4_identity.json`](alloc_b4_identity.json),
[`alloc_b3_chain.json`](alloc_b3_chain.json).

## Execution model (D0)

The TurtleBot3 execution model samples an actual execution duration below WCET at commit time. `DartsMARLAgent.commit_task` (`alg/darts_agent.py:704`) initializes `CommittedJob.remaining_work_ms = task.wcet * np.random.uniform(0.7, 1.0)` from the global NumPy RNG (seeded by `--eval_seed_base`). Certificate and auction envelopes still use full WCET. Physical drain is deterministic once remaining work is drawn: `env.step` (`envs/turtlebot3_spatial_env.py:588, :605–606`) calls `execute_task(..., delta_ms=step_dt_ms)` with `step_dt_ms = step_dt_s * 1000 = 1000` only while the agent is within `PICK_RADIUS_M` of the task; `execute_task` (`alg/darts_agent.py:897–902`) does `remaining_work_ms -= delta_ms` with no per-agent speed or efficiency factor. Completions are timestamped at `_clock_ms` before the step increment (`turtlebot3_spatial_env.py:615, :745`). For G4 row J1#1 (commit 120000, finish 132000) the agent was already at the J1 location, so drain started on the next step; 13 ticks of 1000 ms with remaining work in (12000, 13000] (i.e. 18000×U for U in [0.70, 0.722]) yield finish − commit = 12000 ms. Observed G4 spans {12000, 13000, 15000, 16000, 17000} ms are exactly `(ceil(U·18) − 1)·1000` for U ∼ Uniform(0.7, 1.0). This sub-WCET draw is part of the calibrated simulator, not a drain-rate bug.

## Periodic-mode CSV note

In periodic mode the main eval CSV’s end-to-end miss columns are **not meaningful**: env sinks stay class-id keyed, and minted `class#ell` instance ids never complete those class sinks, so `end_to_end_deadline_miss` is the horizon clock test rather than a job-level miss. Claims A/B/C are adjudicated exclusively from `release_log.csv`.

## Solver constants

| Symbol | Value |
|---|---|
| h | 50 ms |
| H | 1000 ms |
| δ_auc (offline) | 30 ms |
| J̄ | 0 |
| u | 0.22 m/s |
| L | 30 |
| B4 Ā | 4 → Λ_lat = 2120 ms |
| B3 Ā | 1 → Λ_lat = 2030 ms |
| B4 τ̄ impl / raw | 90156.1146012848 / 54640.06945532413 ms |
| B3 τ̄ impl / raw | 77217.22605740251 / 46798.31882266819 ms |
| B4 ξ effective (= declared) | J1–J4 = 18000 ms |
| B3 ξ declared | A=10000, B=15000, C=20000, D=15000 ms |
| B3 ξ **runtime-effective (B1 merge)** | **A=15000, B=25000, C=25000, D=30000 ms** |
| checkpoints | `results/darts_{workload}_medium_seed100`, `--load_ckpt_ep 5000` |
| `per_period_energy` | reset |
| `eval_episodes` | 1 (`--eval_episodes`, already on `alg/train_darts.py`) |
| `max_episode_steps` | ceil((L·P + P)/1000) |
| `global_deadline_ms` | L·P + P |

## Hand cross-check (B4 factored, admitted d085)

c̄ = ξ + h + τ̄_impl = 18000 + 50 + 90156.1146012848 = **108206.1146012848** ms

δ = c̄ / (D_v − Λ_lat) = 108206.1146012848 / (130000 − 2120) = 108206.1146012848 / 127880 = **0.846153539265599**

Matches the solver’s per-agent factored δ-sum for `exp2_B4_DEM_d085` to all printed digits. (Using the three-decimal rounding 90156.115 from the P1 G5 table differs only at the 9th digit: 0.846153542.)

## Sweep rows (12)

Probe rows are labeled **outside the certified region**: D_v is a 1000 ms multiple whose factored bottleneck δ-sum lies in [1.05, 1.10]; verdict REJECT. `exp2_B4_DEM_d085_rep` is the named seed-200 repeat. `exp2_B3_IQI_d085_rep` is the matching B3 seed-200 repeat so the runner’s dry-run prints 12 commands (5 B4 + 5 B3 + 2 repeats).

δ columns: factored / unfactored (informational). Per-agent values are identical on B4 (one job each); B3 bottleneck is agent_4 (ξ_D = 30000).

| config_id | workload | alloc map | target | D_v | P | L | δ_sum factored (per agent) | δ_sum unfactored | Λ_lat | Ā | verdict | assumption | ckpt | ep | eval_seed_base | per_period_energy | max_episode_steps | global_deadline_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp2_B4_DEM_d030 | B4_DEM | alloc_b4_identity.json | 0.3 | 363000 | 363000 | 30 | 0.2998395993163512 ×4 | 0.20142448862592585 ×4 | 2120 | 4 | ADMIT | PASS | darts_B4_DEM_medium_seed100 | 5000 | 100 | reset | 11253 | 11253000 |
| exp2_B4_DEM_d050 | B4_DEM | alloc_b4_identity.json | 0.5 | 219000 | 219000 | 30 | 0.4989215907473479 ×4 | 0.3351626219813912 ×4 | 2120 | 4 | ADMIT | PASS | darts_B4_DEM_medium_seed100 | 5000 | 100 | reset | 6789 | 6789000 |
| exp2_B4_DEM_d070 | B4_DEM | alloc_b4_identity.json | 0.7 | 157000 | 157000 | 30 | 0.6986448515062293 ×4 | 0.46933154348737166 ×4 | 2120 | 4 | ADMIT | PASS | darts_B4_DEM_medium_seed100 | 5000 | 100 | reset | 4867 | 4867000 |
| exp2_B4_DEM_d085 | B4_DEM | alloc_b4_identity.json | 0.85 | 130000 | 130000 | 30 | 0.846153539265599 ×4 | 0.5684240651808267 ×4 | 2120 | 4 | ADMIT | PASS | darts_B4_DEM_medium_seed100 | 5000 | 100 | reset | 4030 | 4030000 |
| exp2_B4_DEM_probe | B4_DEM | alloc_b4_identity.json | outside [1.05, 1.10] | 103000 | 103000 | 30 | 1.0726220717811736 ×4 | 0.7205597685896522 ×4 | 2120 | 4 | REJECT | PASS | darts_B4_DEM_medium_seed100 | 5000 | 100 | reset | 3193 | 3193000 |
| exp2_B3_IQI_d030 | B3_IQI | alloc_b3_chain.json | 0.3 | 360000 | 663000 | 30 | 0.2578 / 0.2857 / 0.2857 / **0.29965423375534966** | 0.1728 / 0.2007 / 0.2007 / 0.2147 | 2030 | 1 | ADMIT | PASS | darts_B3_IQI_medium_seed100 | 5000 | 100 | reset | 20553 | 20553000 |
| exp2_B3_IQI_d050 | B3_IQI | alloc_b3_chain.json | 0.5 | 217000 | 520000 | 30 | 0.4292 / 0.4757 / 0.4757 / **0.49898695658651215** | 0.2877 / 0.3342 / 0.3342 / 0.3575 | 2030 | 1 | ADMIT | PASS | darts_B3_IQI_medium_seed100 | 5000 | 100 | reset | 16120 | 16120000 |
| exp2_B3_IQI_d070 | B3_IQI | alloc_b3_chain.json | 0.7 | 156000 | 459000 | 30 | 0.5993 / 0.6642 / 0.6642 / **0.6966761450763299** | 0.4017 / 0.4666 / 0.4666 / 0.4991 | 2030 | 1 | ADMIT | PASS | darts_B3_IQI_medium_seed100 | 5000 | 100 | reset | 14229 | 14229000 |
| exp2_B3_IQI_d085 | B3_IQI | alloc_b3_chain.json | 0.85 | 129000 | 432000 | 30 | 0.7267 / 0.8054 / 0.8054 / **0.844823391804383** | 0.4871 / 0.5659 / 0.5659 / 0.6052 | 2030 | 1 | ADMIT | PASS | darts_B3_IQI_medium_seed100 | 5000 | 100 | reset | 13392 | 13392000 |
| exp2_B3_IQI_probe | B3_IQI | alloc_b3_chain.json | outside [1.05, 1.10] | 102000 | 405000 | 30 | 0.9229 / 1.0230 / 1.0230 / **1.0729941588216716** | 0.6187 / 0.7187 / 0.7187 / 0.7687 | 2030 | 1 | REJECT | PASS | darts_B3_IQI_medium_seed100 | 5000 | 100 | reset | 12555 | 12555000 |
| exp2_B4_DEM_d085_rep | B4_DEM | alloc_b4_identity.json | 0.85 | 130000 | 130000 | 30 | 0.846153539265599 ×4 | 0.5684240651808267 ×4 | 2120 | 4 | ADMIT | PASS | darts_B4_DEM_medium_seed100 | 5000 | **200** | reset | 4030 | 4030000 |
| exp2_B3_IQI_d085_rep | B3_IQI | alloc_b3_chain.json | 0.85 | 129000 | 432000 | 30 | 0.7267 / 0.8054 / 0.8054 / **0.844823391804383** | 0.4871 / 0.5659 / 0.5659 / 0.6052 | 2030 | 1 | ADMIT | PASS | darts_B3_IQI_medium_seed100 | 5000 | **200** | reset | 13392 | 13392000 |

B3 per-agent factored values in the compact cells are rounded for display; JSON carries full precision (agent_1 … agent_4).
