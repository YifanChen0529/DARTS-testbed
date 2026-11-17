# EXP-2 pre-registered adjudication gates (Theorem 2(a))

Registered before any sweep-result file was read. Do not edit the D0 rationale
or the Gates section after this file is written. Analysis appends a RESULTS
heading only.

## D0 rationale (`rt_bound_violations` in periodic mode)

Verdict: **VACUOUS**.

`run_episode` computes Theorem 1(ii-a) at `alg/train_darts.py:874-888` by
iterating `for v in workload_ep.node_ids` and looking up `ready_time_ms`,
`commit_time_ms`, and `env.completed_tasks_ms` under those keys. `workload_ep`
is bound once at `:229` via `runtime_workload(env, cfg)` and is never rebound
in periodic mode (`:247-267` rebind `ready_time_ms`, `global_tracker`, and
`dag` to `PeriodicReleaseManager`, but leave `workload_ep` on the class-id
workload). Minted instance ids are `class#ell` (`alg/periodic_release.py:26,
:146`). Completions and ready times after bootstrap are keyed by those minted
ids (`PeriodicReleaseManager.bootstrap` / `register_job_instance` at
`envs/turtlebot3_spatial_env.py:837-864`). Class-id lookups therefore miss,
every `tau`/`tc`/`cv` is `None`, and `rt_violations` stays 0 without inspecting
any committed instance. The `subA`..`subD` response-time columns
(`alg/train_darts.py:1164-1168`) use `csv_subtask_slots(workload)` from
`envs/eval_node_mapping.py:10-24`, which returns the same class ids
(`J1`..`J4` or `A`..`D`) and so are likewise empty (`N/A`). Gate GA therefore
does not apply the `rt_bound_violations` sum.

## Gates

### GI (integrity, strict)

12 of 12 runs completed with rc=0; every run directory contains
`release_log.csv` with exactly 120 rows (4 job classes x 30 releases); zero
alloc-map mismatches on every row of every run; `fallback_blocked` is false on
every row of every ADMIT-verdict config (probe values reported verbatim, not
gated).

### GA (safety, strict)

Scope = every job row of every ADMIT-verdict config, i.e. 10 configs x 120
rows = 1200 rows: `met == true` on all 1200 rows.

Optional clause: clause not applicable, see D0 rationale.

### GB (reduction, strict)

Same 1200-row scope: `queue_empty_at_next_boundary == true` on all rows.

### GC (tightness, informational, no pass/fail)

Per config, pooled max and P99 of `(finish_ms - r_ms)/D_v` over its 120 rows,
plus per-class max. Direction statement only: admitted-region values are
expected well below 1. No monotonicity, ordering, or cross-config relation is
registered.

### GP (probe, report-only)

The two REJECT-verdict configs' miss counts, drain-failure counts, and
response ratios are reported verbatim. Zero misses at the probe is consistent
with the one-sided sufficient nature of test (27) and is not a gate in either
direction; nonzero misses there do not contradict the theorem.

### GS (stability, report-only)

For each benchmark, the density-0.85 config and its seed-200 repetition are
reported side by side (GA/GB outcomes, GC summaries). No identity between them
is registered.

## RESULTS

GI PASS: 12/12 rc=0; 12/12 release_log.csv with 120 rows; alloc-map mismatches 0; ADMIT fallback_blocked 0.
GA PASS: met=true on 1200 of 1200 ADMIT job rows (misses 0). rt_bound_violations clause not applicable, see D0 rationale.
GB PASS: queue_empty_at_next_boundary=true on 1200 of 1200 ADMIT job rows (drain failures 0).

GC (informational):
  exp2_B4_DEM_d030: max=0.121212 P99=0.0953719 per-class max [J1=0.0909091, J2=0.0606061, J3=0.121212, J4=0.0964187]
  exp2_B4_DEM_d050: max=0.200913 P99=0.158082 per-class max [J1=0.150685, J2=0.100457, J3=0.200913, J4=0.159817]
  exp2_B4_DEM_d070: max=0.280255 P99=0.22051 per-class max [J1=0.210191, J2=0.140127, J3=0.280255, J4=0.22293]
  exp2_B4_DEM_d085: max=0.338462 P99=0.266308 per-class max [J1=0.253846, J2=0.169231, J3=0.338462, J4=0.269231]
  exp2_B4_DEM_probe: max=0.427184 P99=0.336117 per-class max [J1=0.320388, J2=0.213592, J3=0.427184, J4=0.339806]
  exp2_B3_IQI_d030: max=0.105556 P99=0.0873056 per-class max [A=0.0388889, B=0.105556, C=0.0666667, D=0.0888889]
  exp2_B3_IQI_d050: max=0.175115 P99=0.144839 per-class max [A=0.0645161, B=0.175115, C=0.110599, D=0.147465]
  exp2_B3_IQI_d070: max=0.24359 P99=0.201474 per-class max [A=0.0897436, B=0.24359, C=0.153846, D=0.205128]
  exp2_B3_IQI_d085: max=0.294574 P99=0.243643 per-class max [A=0.108527, B=0.294574, C=0.186047, D=0.248062]
  exp2_B3_IQI_probe: max=0.372549 P99=0.308137 per-class max [A=0.137255, B=0.372549, C=0.235294, D=0.313725]
  exp2_B4_DEM_d085_rep: max=0.684615 P99=0.544 per-class max [J1=0.361538, J2=0.684615, J3=0.561538, J4=0.469231]
  exp2_B3_IQI_d085_rep: max=0.496124 P99=0.299767 per-class max [A=0.217054, B=0.310078, C=0.255814, D=0.496124]

GP (report-only, REJECT-verdict configs):
  exp2_B4_DEM_probe: misses=0 drain_failures=0 fallback_blocked=0 ratio_max=0.427184 ratio_P99=0.336117
  exp2_B3_IQI_probe: misses=0 drain_failures=0 fallback_blocked=0 ratio_max=0.372549 ratio_P99=0.308137

GS (report-only, density-0.85 vs seed-200 repetition):
  B4_DEM: exp2_B4_DEM_d085 (seed 100) vs exp2_B4_DEM_d085_rep (seed 200); GA misses 0/0; GB drain 0/0; GC max 0.338462/0.684615 P99 0.266308/0.544
  B3_IQI: exp2_B3_IQI_d085 (seed 100) vs exp2_B3_IQI_d085_rep (seed 200); GA misses 0/0; GB drain 0/0; GC max 0.294574/0.496124 P99 0.243643/0.299767
