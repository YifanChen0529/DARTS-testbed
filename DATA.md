# DATA.md: Data documentation for the DARTS^rt artifact (RTSS 2026, Paper #117)

This file documents everything in `data/`: layout, schema, conventions, per-element
provenance, reproduction fidelity, and the semantics of the timing columns. It is
the companion to `README.md` §4.

---

## 1. Scope and provenance summary

`data/` contains **769 curated episode-log CSVs (≈12 MB)** copied from the two raw
result trees on the experiment machine (a Jetson Orin Nano; all experiments in the
paper were executed on this hardware). Curation copied `*_log.csv` files only
(no checkpoints or auxiliary logs), stripped whitespace from CSV header fields
(§4), and recorded a SHA-256 manifest per tree. The curation tool is
`scripts/ae_curate_data.py`; it can be re-run with `--dry_run` to display the exact
selection plan.

## 2. Directory layout

```
data/
  raw/behavioral/
    sweep_a1a2b1/     246 files  exp_<i>/B1/<stress>/<method>_log.csv
                                 <stress> ∈ {light, medium, heavy,
                                             medium_n2, medium_n6, medium_n8,   # agent scaling
                                             energy_b02, energy_b03, energy_b04} # energy stress
    legacy_medium/    190 files  <method>_<workload>_<stress>_seed<seed>/<method>_log.csv
                                 (B2_LML / B3_IQI / B4_DEM benchmark suites)
    sweep_d1/         333 files  exp_<i>/B<j>/<stress>/<method>_log.csv
                                 (merged Supp.-Table-III input tree; see §6)
  manifests/
    MANIFEST_sweep_a1a2b1.tsv    dest_rel_path, src_abs_path, sha256, n_rows, n_cols
    MANIFEST_legacy_medium.tsv
    MANIFEST_sweep_d1.tsv
    SEEDCOUNTS.md                per-(benchmark, stress, method) file counts per tree
```

## 3. Episode-log CSV schema (63 columns)

Each row is an **evaluation snapshot** appended every `save_interval` training
episodes; each row's metrics are means over an evaluation batch run at that
checkpoint. The full column list, in file order:

<!-- COLUMN_TABLE_AUTOGEN -->
| # | column | unit | meaning |
|---:|---|---|---|
| 1 | episode | — | Training-episode index of this evaluation snapshot. |
| 2 | A1_total_reward | reward | Agent 1 episode return (eval-batch mean when aggregated). |
| 3 | A1_remaining_battery_pct | % | Agent 1 end-of-episode battery as percent of initial capacity. |
| 4 | A2_total_reward | reward | Agent 2 episode return (eval-batch mean when aggregated). |
| 5 | A2_remaining_battery_pct | % | Agent 2 end-of-episode battery as percent of initial capacity. |
| 6 | A3_total_reward | reward | Agent 3 episode return (eval-batch mean when aggregated). |
| 7 | A3_remaining_battery_pct | % | Agent 3 end-of-episode battery as percent of initial capacity. |
| 8 | A4_total_reward | reward | Agent 4 episode return (eval-batch mean when aggregated). |
| 9 | A4_remaining_battery_pct | % | Agent 4 end-of-episode battery as percent of initial capacity. |
| 10 | subA_miss_rate | fraction | Subtask-class A miss: 1 if that DAG slot never completed, else 0 (eval mean). |
| 11 | subA_time | ms | Subtask-class A completion time (`subtask_completion_times_ms`), or N/A if missed. |
| 12 | subB_miss_rate | fraction | Subtask-class B miss: 1 if that DAG slot never completed, else 0 (eval mean). |
| 13 | subB_time | ms | Subtask-class B completion time, or N/A if missed. |
| 14 | subC_miss_rate | fraction | Subtask-class C miss: 1 if that DAG slot never completed, else 0 (eval mean). |
| 15 | subC_time | ms | Subtask-class C completion time, or N/A if missed. |
| 16 | subD_miss_rate | fraction | Subtask-class D miss: 1 if that DAG slot never completed, else 0 (eval mean). |
| 17 | subD_time | ms | Subtask-class D completion time, or N/A if missed. |
| 18 | task_miss_rate | fraction | End-to-end task miss: 1 if `task_time` is missing or `> global_deadline_ms`, else 0. |
| 19 | task_time | ms | End-to-end task completion time (`csv_task_time_ms`), or N/A if unfinished. |
| 20 | rho_hat | fraction | Certified coverage \|V_ff\| / V_total (`CoverageStats.rho_hat`). |
| 21 | n_full_fidelity | count | Jobs committed in full fidelity this episode (auction counter / coverage V_ff). |
| 22 | n_total_jobs | count | Total jobs in the episode DAG (`CoverageStats.V_total`). |
| 23 | rho_sp_A1 | fraction | Agent 1 sponsor success rate: \|V_ff_per_sponsor\| / \|V_sp\| (`rho_sp_i`). |
| 24 | rho_sp_A2 | fraction | Agent 2 sponsor success rate (`rho_sp_i`). |
| 25 | rho_sp_A3 | fraction | Agent 3 sponsor success rate (`rho_sp_i`). |
| 26 | rho_sp_A4 | fraction | Agent 4 sponsor success rate (`rho_sp_i`). |
| 27 | omega_A1 | fraction | Agent 1 sponsorship share: \|V_sp\| / V_total (`omega_i`). |
| 28 | omega_A2 | fraction | Agent 2 sponsorship share (`omega_i`). |
| 29 | omega_A3 | fraction | Agent 3 sponsorship share (`omega_i`). |
| 30 | omega_A4 | fraction | Agent 4 sponsorship share (`omega_i`). |
| 31 | A1_crash_rate | % | Percent of eval episodes where agent 1 had an energy-certificate violation. |
| 32 | A2_crash_rate | % | Percent of eval episodes where agent 2 had an energy-certificate violation. |
| 33 | A3_crash_rate | % | Percent of eval episodes where agent 3 had an energy-certificate violation. |
| 34 | A4_crash_rate | % | Percent of eval episodes where agent 4 had an energy-certificate violation. |
| 35 | subA_rt_ms | ms | Subtask-class A response time R_v = C_v − τ(v) from `response_times_ms`. |
| 36 | subB_rt_ms | ms | Subtask-class B response time from `response_times_ms`. |
| 37 | subC_rt_ms | ms | Subtask-class C response time from `response_times_ms`. |
| 38 | subD_rt_ms | ms | Subtask-class D response time from `response_times_ms`. |
| 39 | max_rt_ms | ms | Max over jobs of measured response time (`max_response_time_ms`). |
| 40 | max_alloc_delay_ms | ms | Max over jobs of allocation delay A_v = t_v^com − τ(v) (`max_allocation_delay_ms`). |
| 41 | rt_bound_violations | count | Jobs whose measured R_v exceeded the certified RT bound (+1 ms slack). |
| 42 | max_t_auc_ms | ms | Max per-epoch auction wall time (`perf_counter` over the ready-set auction loop). |
| 43 | mean_t_auc_ms | ms | Mean per-epoch auction wall time over epochs with a nonempty ready set. |
| 44 | max_A_k | count | Max ready-frontier size \|A_k\| over auction epochs. |
| 45 | mean_A_k | count | Mean ready-frontier size \|A_k\| over auction epochs. |
| 46 | max_delta_auc_ms | ms | Max single-job auction-subround wall time (`_delta_auc_subround_timer`). |
| 47 | mean_delta_auc_ms | ms | Mean single-job auction-subround wall time. |
| 48 | t_auc_Ak1_mean_ms | ms | Mean epoch auction time over epochs with \|A_k\| = 1. |
| 49 | t_auc_Ak2_mean_ms | ms | Mean epoch auction time over epochs with \|A_k\| = 2. |
| 50 | t_auc_Ak3_mean_ms | ms | Mean epoch auction time over epochs with \|A_k\| = 3. |
| 51 | t_auc_Ak4plus_mean_ms | ms | Mean epoch auction time over epochs with \|A_k\| ≥ 4. |
| 52 | n_epochs_Ak1 | count | Number of auction epochs with \|A_k\| = 1. |
| 53 | n_epochs_Ak2 | count | Number of auction epochs with \|A_k\| = 2. |
| 54 | n_epochs_Ak3 | count | Number of auction epochs with \|A_k\| = 3. |
| 55 | n_epochs_Ak4plus | count | Number of auction epochs with \|A_k\| ≥ 4. |
| 56 | n_degradable_fallback | count | Jobs assigned via degradable fallback (`fallback_now`). |
| 57 | n_nondegradable_n1 | count | Non-degradable N1 drops (no feasible bidder; `n1_drop`). |
| 58 | n_nondegradable_n2 | count | Non-degradable N2 casualties (fallback insufficient; `n2_casualty`). |
| 59 | n_pending | count | Jobs not yet resolved: V_total − (FF + degradable FB + N1 + N2). |
| 60 | n_miss_n1_drop | count | Structural misses equal to N1 drops (`n_miss_n1_drop = n_nondegradable_n1`). |
| 61 | n_miss_n2_fallback_insufficient | count | Structural misses equal to N2 casualties (`= n_nondegradable_n2`). |
| 62 | n_miss_n3_unreached | count | FF/fallback commits that never physically completed (`task_outcome` in {full_fidelity, fallback_now} ∩ not in `completed_tasks_ms`). |
| 63 | n_miss_full_fidelity_late | count | Full-fidelity commits that completed after their absolute deadline. |

Column families, semantically:

- **`episode`**: training-episode index of the snapshot.
- **Per-agent state** (`A1..A4_total_reward`, `A1..A4_remaining_battery_pct`,
  `A1..A4_crash_rate`): episode reward, end-of-episode battery, crash rate.
- **Per-class subtask metrics** (`subA..subD_miss_rate`, `subA..subD_time`,
  `subA..subD_rt_ms`): deadline-miss rate, completion time, and response time per
  DAG subtask class.
- **Headline task metrics** (`task_miss_rate`, `task_time`): end-to-end task
  deadline-miss rate (the Supp. Table III quantity) and completion time.
- **Certificate / admission** (`rho_hat`, `n_full_fidelity`, `n_total_jobs`,
  degradable/non-degradable fallback counters, `n_pending`): certified coverage
  and admission-path counters.
- **Structural-miss breakdown** (N1 / N2 / N3 / full-fidelity-late counters): the
  Supp. Table V decomposition. In every shipped cell, N1 = N2 = FFL = 0: all
  DARTS-family misses are N3 (energy-unreached), i.e. no committed full-fidelity
  job ever overran its deadline (Theorem 1(ii-b)).
- **Response-time telemetry** (`max_rt_ms`, `max_alloc_delay_ms`,
  `rt_bound_violations`): measured response times against the certified bound;
  `rt_bound_violations` counts jobs whose measured RT exceeded the bound (0 for
  the DARTS family in all shipped cells).
- **Auction timing** (`max/mean_t_auc_ms`, `t_auc_Ak{1,2,3,4plus}_mean_ms`,
  `n_epochs_Ak*`, `max/mean_delta_auc_ms`, `max/mean_A_k`): see §8.

## 4. Header-whitespace note

The raw logs were written with whitespace-padded header fields (e.g.
`' task_miss_rate'`). Curation strips leading/trailing whitespace from the header
line of every copied file; **data rows are byte-identical to the originals**. The
manifests record the SHA-256 of the curated (stripped-header) files. Analysis
scripts also strip headers on read, so they accept both forms.

## 5. Conventions

- **Seed mapping:** `exp_<i>` ↔ seed `i × 100` (seeds 100..1000 for i = 1..10).
  Legacy directory names carry the seed literally (`..._seed300`).
- **Method names:** file/directory tokens map to paper labels as
  `darts` → DARTS^rt, `darts_nocert` → NoCert, `darts_no_incentive` → NoIncentive,
  `cea_edf` → CEA-EDF, `sadcher` → SADCHER.
- **Convergence-window semantics:** all Supp.-Table-III values are means of
  `task_miss_rate` over the evaluation rows at episodes **{4500, 5000}** (the
  final 1,000 of 5,000 training episodes at a 500-episode evaluation cadence),
  after the standard row-cleaning used by the figure generators, averaged across
  seeds. `scripts/ae_spotcheck_tab3.py` implements exactly this and is what
  `make verify` runs.
- **Episode counts:** some cells contain runs longer than 5,000 episodes; all
  published behavioral quantities use the {4500, 5000} window, so rows beyond
  episode 5,000 do not affect them.

## 6. Per-element provenance

| Element | Generator(s) | Data root(s) |
|---|---|---|
| Fig. 3 left (`fig0_headline_bar.py`) | B1/heavy headline bars | `sweep_a1a2b1` |
| Fig. 3 right (`fig3_t3_bound_tightness.py` ← `vg_t3_combined.py`) | bound tightness | `sweep_a1a2b1` + `legacy_medium` + in-code offline bounds (`r_off_g.py`) |
| Fig. 4 (`fig1_b1_grid.py`) | B1 grid + coverage | `sweep_a1a2b1` |
| Fig. 5 (`fig2_generalization_scaling.py`) | generalization + timing panels | `sweep_a1a2b1` + `legacy_medium` (+ frozen constants, §8) |
| Fig. 6 (`fig4_d1_heatmap.py` ← `recover_63col.py`) | admission heatmap | `sweep_a1a2b1` |
| Supp. Table II (`vg_t3_combined.py`) | offline-bound tightness | `sweep_a1a2b1` + `legacy_medium` |
| Supp. Table III (`summarize_sweep.py`; verified by `ae_spotcheck_tab3.py`) | miss-rate grid | `sweep_d1` |
| Supp. Table IV (`vg_analysis.py`) | auction-latency analysis | `sweep_a1a2b1` + `legacy_medium` |
| Supp. Table V (`recover_63col.py`) | admission decomposition | `sweep_a1a2b1` |

The `sweep_d1` tree is a **merged selection**: B1 cells (all five methods) from the
A1 sweep tree; B2–B4 cells from the exp-layout tree except B2/light, whose full
10-seed set exists in the legacy layout (mapped to `exp_<seed/100>` on copy). The
exact per-cell selection is encoded verbatim in `scripts/ae_curate_data.py`
(`SELECTION` dict) and printed by `--dry_run`.

## 7. Reproduction fidelity

`make verify` runs `scripts/ae_spotcheck_tab3.py`, which recomputes the full
Supp. Table III grid (40 cells, using the conventions of §5) from
`data/raw/behavioral/sweep_d1` and checks it against the packaged reference
values. It must report **all 40 cells PASS**. A failure indicates an incomplete
or corrupted data package.

Every DARTS^rt cell in the printed table reproduces exactly from the shipped
data. Ablation and baseline cells may differ slightly from the printed values,
within cross-seed variation; the gaps between the DARTS family and the baselines
are unaffected. The regenerated `vg_*` analysis files may likewise differ
slightly from the committed repo-root copies. Per-cell seed counts are listed in
`data/manifests/SEEDCOUNTS.md`.

## 8. Timing columns and the Fig. 5 timing panels

**What the timing columns measure.** `*_t_auc_*` and `*_delta_auc_*` are wall-clock
(`time.perf_counter`) measurements around the auction computation on the host CPU:
bid computation, the sequential-feasibility and energy certificates, and
assignment.
`t_auc` spans the full ready-set auction loop of an epoch; `delta_auc` is a single
auction subround. They are recorded identically in simulator and distributed
deployment modes, and exclude both the simulator's synthetic message latencies and
the distributed mode's epoch-synchronization barrier. Since all experiments ran on
a Jetson Orin Nano, these columns are **Jetson-silicon auction-compute
measurements** in every shipped file. (The analytical bound parameter T̄_auc used
in the response-time bound is separate from these measured columns and is 0 in
simulator mode by construction.)

**Fig. 5 bottom panels (auction timing, agent scaling).** The generator
`fig2_generalization_scaling.py` carries the paper's panel values as explicit,
labeled constants. When regenerating, it uses CSV-derived values wherever the
shipped tree provides them and the constants elsewhere. Passing `--strict`
disables this fallback and fails loudly instead; `make fig5` uses the default
mode. The raw timing CSVs are shipped as well (`sweep_a1a2b1`, including the
`medium_n{2,6,8}` agent-scaling cells), so the derivation chain is fully
inspectable.

## 9. Intentionally not included

- `t2_instr_*` per-commit instrumentation runs (support a journal-deferred
  analysis; no published element reads them).
- Smoke / demo / dry-run / debug directories from development.
- Model checkpoints (`.pth`) and TensorBoard logs (large; not consumed by any
  published element).
- Distributed-transport byte logs. Calibration constants from testbed operation
  enter the analysis as explicit CLI parameters of `vg_t3_combined.py` and
  `r_off_g.py`, encoded in the Makefile recipes: δ_auc = 30 ms (set just above
  the maximum observed single-subround time of about 22 ms) and h_comm = 50 ms.

## 10. Verifying the package

```bash
make verify        # recomputes the 40-cell Table III grid from data/; must PASS
```

A PASS confirms the package is complete and intact. The gate recomputes every
cell with the conventions of §5.