# TB3-1 — offline test harness skeleton + S0 port + anchor check

Conclusion first: S0 is a bit-exact port of `scripts/r_off_g.py:90-106`. Every
profiled cell matches `compute_r_off_g` at exact float equality. The published
T3 table (declared ξ, method=`darts`) reproduces to the integer bound, the
measured means ±0.5 ms, and the two-decimal ratios. The B3 runtime-ξ line is
informational only (T0 H3): 290513.275291 ms / 2.39× against the same measured
121692.9 ms. Nominal-instance β* values match the spec to four decimals. Eval
smoke on B1/heavy is accept_e2e = 0,0,0,0,1 and accept_rel = 0,0,0,1,1.

Base HEAD was `e168cff` (TB-1 P5.9), descendant of the prompt's `bf8ebfb`.
TB-1 remains frozen. This prompt did not start TB3-2.

## Anchor (a) tripwire — `s0_path_sum` vs `compute_r_off_g` (`xi_source=declared`)

Exact `==` on `R_G_off_ms` / `R_off_G_ms`, `longest_path`, `A_bar`, `tau_bar_ms`.

| cell | R_G_off_ms | path | A_bar | τ̄_ms | (a) |
|---|---|---|---|---|---|
| B1_CDR:light | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:medium | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:heavy | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:adversarial | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:energy_b04 | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:energy_b03 | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B1_CDR:energy_b02 | 240250.20836597236 | A→B→D | 2 | 54640.06945532413 | PASS |
| B2_LML:light | 255513.27529067278 | Pk→Hf→Dr→Cf | 1 | 46798.31882266819 | PASS |
| B2_LML:medium | 255513.27529067278 | Pk→Hf→Dr→Cf | 1 | 46798.31882266819 | PASS |
| B2_LML:heavy | 255513.27529067278 | Pk→Hf→Dr→Cf | 1 | 46798.31882266819 | PASS |
| B3_IQI:medium | 255513.27529067275 | A→B→C→D | 1 | 46798.31882266819 | PASS |
| B4_DEM:medium | 74810.06945532412 | J1 | 4 | 54640.06945532413 | PASS |

## Anchor (b) published T3 table (declared ξ, method=`darts`, n_seeds=10)

| cell | round(R) | want | meas mean ± std | target mean | ratio | want ratio | (b) |
|---|---|---|---|---|---|---|---|
| B1_CDR:light | 240250 | 240250 | 109677.8±1741.5 | 109677.8 | 2.19 | 2.19 | PASS |
| B1_CDR:medium | 240250 | 240250 | 103381.5±1468.0 | 103381.5 | 2.32 | 2.32 | PASS |
| B1_CDR:heavy | 240250 | 240250 | 100540.4±996.9 | 100540.4 | 2.39 | 2.39 | PASS |
| B2_LML:medium | 255513 | 255513 | 94403.1±945.1 | 94403.1 | 2.71 | 2.71 | PASS |
| B3_IQI:medium | 255513 | 255513 | 121692.9±729.5 | 121692.9 | 2.10 | 2.10 | PASS |
| B4_DEM:medium | 74810 | 74810 | 57059.4±1505.3 | 57059.4 | 1.31 | 1.31 | PASS |

Measured side: `scripts/vg_t3_combined.py:41-60` `measured_r_g`, loaded via
`importlib.util` (same pattern that module uses for `r_off_g.py`). Globals
`NEW_SWEEP` / `OLD_RESULTS` patched to the Makefile roots
`data/raw/behavioral/sweep_a1a2b1` and `data/raw/behavioral/legacy_medium`.
Body not copied.

## Anchor (c) T0 H3, not published

B3 `xi_source=runtime`: **R_G_off = 290513.275291**, ratio vs 121692.9 = **2.39**.

Declared B3 path-sum is 255513.275291 (ξ sum 60000). Runtime merge (T0 H3)
replaces A/B/C/D with B1's 15000/25000/25000/30000 (sum 95000); the chain has
four nodes so Δξ = 35000 and R grows by exactly that.

## Frontier (S0, nominal instances, both ξ sources)

P0 β grid is **12** points
`0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,2.0`. The spec's "13 points"
does not match this literal.

| cell | xi | R_G_off | D_e2e | D_rel | β*_e2e | β*_rel |
|---|---|---|---|---|---|---|
| B1_CDR:light | declared | 240250.208366 | 200000 | 150000 | 1.2013 | 1.6017 |
| B1_CDR:light | runtime | 240250.208366 | 200000 | 150000 | 1.2013 | 1.6017 |
| B1_CDR:medium | declared | 240250.208366 | 150000 | 150000 | 1.6017 | 1.6017 |
| B1_CDR:medium | runtime | 240250.208366 | 150000 | 150000 | 1.6017 | 1.6017 |
| B1_CDR:heavy | declared | 240250.208366 | 120000 | 150000 | 2.0021 | 1.6017 |
| B1_CDR:heavy | runtime | 240250.208366 | 120000 | 150000 | 2.0021 | 1.6017 |
| B2_LML:medium | declared | 255513.275291 | 240000 | 170000 | 1.0646 | 1.5030 |
| B2_LML:medium | runtime | 255513.275291 | 240000 | 170000 | 1.0646 | 1.5030 |
| B3_IQI:medium | declared | 255513.275291 | 180000 | 150000 | 1.4195 | 1.7034 |
| B3_IQI:medium | runtime | 290513.275291 | 180000 | 150000 | 1.6140 | 1.9368 |
| B4_DEM:medium | declared | 74810.069455 | 100000 | 100000 | 0.7481 | 0.7481 |
| B4_DEM:medium | runtime | 74810.069455 | 100000 | 100000 | 0.7481 | 0.7481 |

`accept_e2e` on the P0 grid (1 = accept). B1/medium is rejected at β=1.6
(1.6017 > 1.6). B1/heavy is rejected at β=2.0 (2.0021 > 2.0).

| cell | xi | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 | 1.6 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_CDR:light | declared | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| B1_CDR:medium | declared | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B1_CDR:heavy | declared | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B2_LML:medium | declared | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| B3_IQI:medium | declared | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| B3_IQI:medium | runtime | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| B4_DEM:medium | declared | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Runtime bits equal declared for B1/B2/B4 (ξ vectors coincide on DAG nodes).

## Eval smoke — `B1_CDR:heavy`, S0, `--xi-source` default `runtime`

R = 240250.208366. D_e2e = 120000, D_rel = 150000 at β=1.

| β | D_e2e | D_rel | accept_e2e | accept_rel | load_e2e | load_rel |
|---|---|---|---|---|---|---|
| 0.8 | 96000 | 120000 | 0 | 0 | 2.5026 | 2.0021 |
| 1.0 | 120000 | 150000 | 0 | 0 | 2.0021 | 1.6017 |
| 1.25 | 150000 | 187500 | 0 | 0 | 1.6017 | 1.2813 |
| 2.0 | 240000 | 300000 | 0 | 1 | 1.0010 | 0.8008 |
| 2.5 | 300000 | 375000 | 1 | 1 | 0.8008 | 0.6407 |

accept_e2e = 0,0,0,0,1; accept_rel = 0,0,0,1,1 (240250.2/150000 = 1.6017 ≤ 2.0).

## File list (line counts)

| file | lines | role |
|---|---|---|
| `alg/tb3_instance.py` | 178 | torch-free `MissionInstance`, `effective_task_wcets`, `nominal_instance` |
| `scripts/offline_test_harness.py` | 562 | S0 port, registry, CLI `anchor`/`frontier`/`eval` |
| `tests/test_tb3_s0_port.py` | 176 | t1–t7 tripwire / H3 lock / torch-free import |
| `docs/tb3/TB3-1_report.md` | 251 | this report |
| `scripts/csv_schema_tb.py` | 196 (+54 appended) | `TB3_OFFLINE_HEADER` only; 76-col tests still pass |

Not in this commit: `docs/tb3/T0_recon_report.md` (T0, still untracked),
`results_tb3_offline/` (fresh, gitignored-by-policy / not added).

## Commands run

All under `conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 ...`.
Nothing launched `train_darts.py`. Nothing wrote under `results/`.

```
python3 -m pytest -q tests/
# 59 passed, 0 failed (52 pre-existing at e168cff + 7 new in test_tb3_s0_port.py)

python3 scripts/offline_test_harness.py anchor \
  --new_sweep_dir data/raw/behavioral/sweep_a1a2b1 \
  --old_results_dir data/raw/behavioral/legacy_medium \
  --out results_tb3_offline/anchor
# every (a)/(b) line PASS; (c) prints 290513.275291 / 2.39

python3 scripts/offline_test_harness.py frontier --xi-source both \
  --out results_tb3_offline/frontier
# β* table matches spec to 4 decimals

python3 scripts/offline_test_harness.py eval \
  --cell B1_CDR:heavy --beta 0.8,1.0,1.25,2.0,2.5 \
  --out results_tb3_offline/eval
# accept_e2e 0,0,0,0,1 ; accept_rel 0,0,0,1,1
```

Frozen-path check (this prompt did not touch these files; the a1d8491…HEAD
delta is TB-1 history):

```
git diff --stat a1d8491 -- scripts/r_off_g.py scripts/vg_t3_combined.py \
  envs alg/train_darts.py alg/darts_agent.py
# this-prompt vs HEAD: empty
# vs a1d8491: alg/darts_agent.py, alg/train_darts.py, envs/turtlebot3_spatial_env.py
#             (TB-1 insertions already on main; r_off_g.py and vg_t3_combined.py
#             byte-identical to a1d8491)
```

`git diff HEAD -- scripts/r_off_g.py scripts/vg_t3_combined.py envs alg/train_darts.py alg/darts_agent.py`
is empty.

## `git status --porcelain`

### Before the commit

```
 M README.md
 M scripts/csv_schema_tb.py
?? DATA_INVENTORY_20260523_2116.md
?? DATA_INVENTORY_20260523_2121.md
?? DIAG_REPORT_20260523_2215.md
?? DIAG_REPORT_20260523_2220_FINAL.md
?? DIAG_REPORT_20260523_2221.md
?? LAYOUT_FIX_REPORT_20260523_2252.md
?? LAYOUT_FIX_REPORT_v2_20260523_2315.md
?? LAYOUT_FIX_v3_REPORT_20260523_2330.md
?? alg/tb3_instance.py
?? diag_ablation_b1medium.md
?? docs/
?? fig/
?? results_tb1_noisecheck/
?? results_tb1_p0/
?? results_tb3_offline/
?? scripts/offline_test_harness.py
?? sweep_logs_b2_noinc_20260524_0717/
?? sweep_logs_b2_noinc_20260524_1357/
?? sweep_logs_noinc_completion_20260523_1726/
?? sweep_logs_t2_20260524_023919/
?? sweep_logs_t2_20260524_025153/
?? t2_analysis_output.md
?? tests/test_tb3_s0_port.py
```

Staged for this commit only: `alg/tb3_instance.py`,
`scripts/offline_test_harness.py`, `tests/test_tb3_s0_port.py`,
`docs/tb3/TB3-1_report.md`, `scripts/csv_schema_tb.py`. Dirty TB-1 leftovers
(`README.md`, `fig/`, `results_tb1_*`, T0 `docs/tb3/T0_recon_report.md`, …)
were left untouched.

### After the commit

```
 M README.md
?? DATA_INVENTORY_20260523_2116.md
?? DATA_INVENTORY_20260523_2121.md
?? DIAG_REPORT_20260523_2215.md
?? DIAG_REPORT_20260523_2220_FINAL.md
?? DIAG_REPORT_20260523_2221.md
?? LAYOUT_FIX_REPORT_20260523_2252.md
?? LAYOUT_FIX_REPORT_v2_20260523_2315.md
?? LAYOUT_FIX_v3_REPORT_20260523_2330.md
?? diag_ablation_b1medium.md
?? docs/tb3/T0_recon_report.md
?? fig/
?? results_tb1_noisecheck/
?? results_tb1_p0/
?? results_tb3_offline/
?? sweep_logs_b2_noinc_20260524_0717/
?? sweep_logs_b2_noinc_20260524_1357/
?? sweep_logs_noinc_completion_20260523_1726/
?? sweep_logs_t2_20260524_023919/
?? sweep_logs_t2_20260524_025153/
?? t2_analysis_output.md
```

TB3-1 files are gone from porcelain. Pre-existing dirty tree is unchanged. `results_tb3_offline/` remains untracked (not committed). T0 recon report remains untracked.

## Commit and tag

- message: `TB-3 P1: offline_test_harness skeleton + S0 port + anchor check`
- tag: `tb3-stack-p1`
- hash: `3d518189c618195440e5bd9c5e78bebea409fe3a` (`3d51818`); tag `tb3-stack-p1` is the durable pointer (`git rev-parse tb3-stack-p1`)

## Carried forward

(i) **B3 ξ discrepancy** (declared 60000 vs runtime 95000; published T3 ratio
2.10× vs runtime-ξ 2.39×) is a camera-ready correction candidate. Decision
owner: Yifan / advisor. Not "fixed" here. `tests/test_tb3_s0_port.py` t3
locks the merge: B3 DAG-node sum becomes 95000; B1/B2/B4 DAG-node sums stay
95000 / 60000 / 72000. The merged dict also carries unused A–D keys on B2/B4
(disjoint node ids); that is the `dag_task.py:106-110` behaviour.

(ii) **B1/heavy D_e2e=120000 vs D_rel=150000** is the D4 definition basis
(T0 H6). Per-job `d_v` uses D_rel; e2e miss uses D_e2e. Both appear as
columns and as `accept_e2e` / `accept_rel`.

(iii) **Nominal-instance S0 thresholds vs the P0 β grid.** B1/heavy β*_e2e =
2.0021 > 2.0 and B1/medium β*_e2e = 1.6017 > 1.6 are **rejected** at those
grid points (non-strict `<=`). The TB3-4 grid and the TB3-5 empirical β set
must be chosen after the offline curves exist.

Stop. Do not start TB3-2.
