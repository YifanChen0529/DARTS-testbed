# TB3-2 — `--instance_seed` deterministic instance generator

Conclusion first: the generator is a torch-free private-RNG function shared
byte-identically by `apply_instance_seed` (harness) and
`apply_instance_seed_runtime` (`alg/train_darts.py`). Unset `--instance_seed`
leaves the nominal B3 T0 H3 merge untouched. G-B (seed 7 twice, seed 8 once)
is a clean pass: 68 non-wall-clock columns byte-identical across the two seed-7
runs, and seed 8 moves miss / battery / busy-time columns. G-A vs the archived
TB-1 P0 row is **pre-existing drift** (protocol stop): two fresh HEAD runs
agree with each other on all 68 columns; they differ from the archive on
exactly four `A*_total_reward` strings at ~1e-9. Nothing was adjusted to force
an archive match. TB3-3 (S1 + lemma) is not started.

Base HEAD was `8eb49a3` (`tb3-stack-p1`).

## G-A — nominal byte-identity vs `results_tb1_p0`

Cell: B1_CDR / heavy / κ=1.0 / seed=100 / β=1.0.
Command shape: `scripts.sweep_tb1.build_command` (conda + `LD_LIBRARY_PATH` +
`PYTHONPATH` + `python3 -u`), `--eval_episodes 100`, `--eval_seed_base 100`,
`--subround_log`, **no** `--instance_seed`,
`--results_dir results_tb3_p2_identity` (repeat into
`results_tb3_p2_identity_b`).

Logged: `[TB3][instance] instance_seed=None: nominal benchmark instance.`
`config.json` carries `instance_seed: null`, `workload_task_wcets_ms` =
`{A:15000, B:25000, C:25000, D:30000}`, and the published B1 locations.
All **old** config keys match the archive (path-valued `load_ckpt` /
`results_dir` excluded).

68 non-wall-clock columns (76 minus the eight `perf_counter` fields
`max/mean_t_auc_ms`, `max/mean_delta_auc_ms`,
`t_auc_Ak{1,2,3,4plus}_mean_ms`):

| comparison | mismatches |
|---|---|
| fresh-1 vs fresh-2 | **0** (byte-identical strings) |
| fresh vs archive | **4** (`A1..A4_total_reward` only) |

| column | archive | fresh (both runs) |
|---|---|---|
| A1_total_reward | 18.06623364202025 | 18.066233636357808 |
| A2_total_reward | 12.7041206360664 | 12.704120636814562 |
| A3_total_reward | 25.9260467526655 | 25.926046759363235 |
| A4_total_reward | 23.286535431823328 | 23.286535433261147 |

Miss rates, remaining-battery percents, `exec_ms_*`, `travel_ms_*`, and the
rest of the 64 columns match the archive as strings. The reward deltas are
~1e-9. This predates TB3-2 (HEAD `bf8ebfb` → `e168cff` → `8eb49a3`); this
eval also ran on CPU (`NvRmMemMgrInit failed` / torch CUDA unavailable in the
sandbox). Per protocol: **report and stop; do not adjust**.

## G-B — seeded reproducibility

Same command shape, `--eval_episodes 3`, `--instance_seed 7` twice
(`results_tb3_p2_seedtest/s7a`, `s7b`) and `--instance_seed 8` once (`s8`).

| check | result |
|---|---|
| seed 7 vs seed 7, 68 columns | **0 mismatches** |
| seed 7 vs seed 7, instance provenance | identical `instance_seed`, WCETs, locations, batteries |
| seed 8 vs seed 7 | **38** columns moved |

Interest columns that moved (seed 8): `task_miss_rate`, `n_miss_n3_unreached`,
`A1..A4_remaining_battery_pct`, `exec_ms_A1..A4`, `travel_ms_A1..A4`.

Seed-7 provenance: WCETs
`A=16259.95, B=28658.91, C=26132.44, D=27059.46`;
batteries `[0.7981, 0.8457, 0.9089, 0.8193]`.
Seed 8 is a different WCET/location/battery draw.

## Harness smoke — `eval --instance-seeds 0..4`

`xi_source=declared` (the new default). Five distinct S0 `R_G_off_ms` on
B1_CDR:heavy β=1.0 (locations move the workspace diameter):

| seed | R_G_off_ms | load_e2e |
|---|---|---|
| 0 | 222455.777299 | 1.8538 |
| 1 | 223404.171142 | 1.8617 |
| 2 | 228300.661306 | 1.9025 |
| 3 | 192818.227962 | 1.6068 |
| 4 | 201941.457419 | 1.6828 |

Nominal declared S0 on this cell is 240250.208366; jitter + resampled map
pulls the bound around.

## Constants

| name | value | provenance |
|---|---|---|
| `WCET_JITTER_LO/HI` | 0.8 / 1.2 | ±20% around the **declared** workload WCET |
| battery draw | `uniform(min(bounds), max(bounds))` iid per agent | does **not** preserve the 1:1.5:2:2.5 ordering |
| `TB3_MAP_SIZE_M` | 10.0 | duplicate of `MAP_SIZE_M` |
| `TB3_OBSTACLES_M` | three rects | duplicate of `TurtleBot3Env._default_obstacles` |
| location sampling | rejection, 200 tries, then `RuntimeError` | contrast: env falls back to `(0,0)` |
| `_point_in_obstacle` | strict `rx < x < rx+w` | same as `_point_in_rect` |
| field RNGs | `default_rng([WORKLOAD_CODES[wid], field, seed])` | independent streams; same seed shares uniforms across stress |

## Declared-centering (B3 consequence)

Jitter multiplies `workload.task_wcets_ms` (declared). Both application paths
then write that same dict to `cfg.task_wcets`, so `DAGGraph.from_workload`'s
merge is the identity — including on B3. A **seeded** B3 instance therefore
runs at the declared-centered regime (nominal declared sum 60000 × jitter),
unlike the **nominal** B3 runtime (T0 H3: B1 15/25/25/30 kms, sum 95000).
B3 has no checkpoints and is offline-only in TB-3. Unset `--instance_seed`
does not touch that merge.

## File list (line counts)

| file | lines | role |
|---|---|---|
| `alg/tb3_instance.py` | 372 | generator + both apply paths (TB3-1 file, editable) |
| `scripts/offline_test_harness.py` | 616 | `--instance-seed(s)`, `xi` default `declared` |
| `alg/train_darts.py` | +38 insertion-only | argparse, generator block, config.json keys |
| `tests/test_tb3_instance_gen.py` | 239 | g1–g7 |
| `docs/tb3/TB3-2_report.md` | 192 | this report |

`git diff a1d8491 -- alg/train_darts.py` has **0** deletion lines.
Frozen at this prompt (byte-identical vs `8eb49a3`): `envs/*`,
`alg/darts_agent.py`, `scripts/r_off_g.py`, `scripts/vg_t3_combined.py`,
`scripts/sweep_tb1.py`, `scripts/csv_schema_tb.py`.

## Commands run

All analysis under
`conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 ...`.
Gates used the `build_command` inner shape (`LD_LIBRARY_PATH` + `PYTHONPATH` +
`python3 -u`). Nothing wrote into `results_tb1_p0/` or `results/`.

```
python3 -m pytest -q tests/
# 66 passed (59 at tb3-stack-p1 + 7 in test_tb3_instance_gen.py)

python3 scripts/offline_test_harness.py eval \
  --cell B1_CDR:heavy --beta 1.0 --instance-seeds 0..4 \
  --out results_tb3_offline/eval_p2

# G-A: build_command(..., results_dir=results_tb3_p2_identity,
#                    eval_episodes=100, eval_seed_base=100, subround_log=True)
#      repeat into results_tb3_p2_identity_b

# G-B: same + --instance_seed 7 twice (s7a/s7b) and --instance_seed 8 (s8),
#      eval_episodes=3, under results_tb3_p2_seedtest/
```

## `git status --porcelain`

### Before the commit

```
 M README.md
 M alg/tb3_instance.py
 M alg/train_darts.py
 M scripts/offline_test_harness.py
?? docs/tb3/T0_recon_report.md
?? tests/test_tb3_instance_gen.py
?? results_tb3_p2_identity/
?? results_tb3_p2_identity_b/
?? results_tb3_p2_seedtest/
… (pre-existing DIAG/LAYOUT/fig/results_tb1_* leftovers omitted in spirit;
   full listing captured at commit time)
```

Staged only: `alg/tb3_instance.py`, `alg/train_darts.py`,
`scripts/offline_test_harness.py`, `tests/test_tb3_instance_gen.py`,
`docs/tb3/TB3-2_report.md`. Gate output dirs and the pre-existing dirty tree
were left alone.

### After the commit

TB3-2 files leave porcelain. Gate dirs stay untracked.

## Commit and tag

- message: `TB-3 P2: --instance_seed deterministic instance generator (private RNG, workload+cfg+battery)`
- tag: `tb3-stack-p2`
- hash: see `git rev-parse tb3-stack-p2`

## Carried forward

- **TB3-4** needs the beta-grid decision (P0 12-point literal vs the spec's
  "13 points"; B1/heavy β*_e2e = 2.0021 needs points > 2.0). Seeded instances
  move R_G_off, so the grid should be chosen after the offline curves exist.
- **TB3-5** needs `sweep_tb1.py` passthrough for `--instance_seed` and the
  empirical beta set re-decided from those curves.
- G-A archive reward-string drift (~1e-9 on `A*_total_reward`) remains a
  camera-ready footnote, not a generator bug.
- B3 declared-vs-runtime ξ (T0 H3) still stands on the **nominal** path.

Stop. Do not start TB3-3.
