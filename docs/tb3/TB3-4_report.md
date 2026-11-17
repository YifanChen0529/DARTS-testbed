# TB3-4 — offline acceptance grid (M=200 × 15 β × 5 cells) + F3 draft

Conclusion first: the 15-point grid is live. All ten e2e acceptance curves
saturate at 1.0 by β=2.4. B4 saturates by β=0.9 (max β* = 0.857). B1/heavy is
still 0/200 at β=0.5 (lower support 0.5194 even at zero travel). On jittered
instances the path-sum test and the per-node travel refinement are
**incomparable**: `n_S1_gt_S0` is 97–136 of 200 per cell, so registered-geometry
dominance does not extend here. At deployed deadlines (β=1.0) B2 accepts 0.170
(path-sum) vs 0.040 (refinement) — the opposite of the TB3-3 *nominal*-instance
headline. Mapping used in this report: path-sum test = S0, per-node travel
refinement = S1 (those codes do not appear in F3). TB3-5 is not started.

Base HEAD was `89d4b1c` (`tb3-stack-p3`).

## (a) Acceptance curves (accept_frac_e2e, M=200)

Rows = deadline scale. Columns = cell × test. Fractions are n/200.

| β | B1m S0 | B1m S1 | B1h S0 | B1h S1 | B2 S0 | B2 S1 | B3 S0 | B3 S1 | B4 S0 | B4 S1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.090 | 0.000 |
| 0.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | 0.475 | 0.085 |
| 0.8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.000 | 0.000 | 0.000 | 0.900 | 0.820 |
| 0.9 | 0.000 | 0.000 | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.170 | 0.040 | 0.005 | 0.000 | 1.000 | 1.000 |
| 1.1 | 0.010 | 0.000 | 0.000 | 0.000 | 0.485 | 0.450 | 0.010 | 0.000 | 1.000 | 1.000 |
| 1.2 | 0.035 | 0.000 | 0.000 | 0.000 | 0.820 | 0.970 | 0.040 | 0.000 | 1.000 | 1.000 |
| 1.3 | 0.115 | 0.000 | 0.005 | 0.000 | 0.975 | 1.000 | 0.125 | 0.000 | 1.000 | 1.000 |
| 1.4 | 0.280 | 0.080 | 0.010 | 0.000 | 1.000 | 1.000 | 0.360 | 0.195 | 1.000 | 1.000 |
| 1.6 | 0.815 | 0.885 | 0.095 | 0.000 | 1.000 | 1.000 | 0.810 | 0.970 | 1.000 | 1.000 |
| 1.8 | 1.000 | 1.000 | 0.385 | 0.145 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2.0 | 1.000 | 1.000 | 0.815 | 0.885 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2.2 | 1.000 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2.4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

β*_i := R_G_off_ms / D_e2e_ms at β=1 (deployed deadlines). Acceptance at β is
the empirical CDF of β* evaluated at that β.

### Frontier summary (β* distribution) and flips

| cell | test | min | p25 | median | p75 | max | n_S1_gt_S0 |
|---|---|---|---|---|---|---|---|
| B1_CDR:medium | S0 | 1.0326 | 1.3817 | 1.4891 | 1.5743 | 1.7819 | 115 |
| B1_CDR:medium | S1 | 1.3303 | 1.4629 | 1.5171 | 1.5640 | 1.6835 | 115 |
| B1_CDR:heavy | S0 | 1.2907 | 1.7271 | 1.8614 | 1.9678 | 2.2274 | 115 |
| B1_CDR:heavy | S1 | 1.6629 | 1.8286 | 1.8964 | 1.9550 | 2.1044 | 115 |
| B2_LML:medium | S0 | 0.6909 | 1.0229 | 1.1049 | 1.1726 | 1.3420 | 97 |
| B2_LML:medium | S1 | 0.9383 | 1.0725 | 1.1036 | 1.1333 | 1.2566 | 97 |
| B3_IQI:medium | S0 | 0.9596 | 1.3700 | 1.4659 | 1.5485 | 1.7811 | 99 |
| B3_IQI:medium | S1 | 1.3096 | 1.4198 | 1.4713 | 1.5145 | 1.6330 | 99 |
| B4_DEM:medium | S0 | 0.5177 | 0.6580 | 0.7044 | 0.7569 | 0.8558 | 136 |
| B4_DEM:medium | S1 | 0.6206 | 0.7316 | 0.7603 | 0.7894 | 0.8571 | 136 |

`n_S1_gt_S0` is a per-cell count (same value on both test rows): number of the
200 seeds where the S1 bound strictly exceeds the S0 bound. B1 medium and B1
heavy share 115 because R does not depend on deadlines or batteries, and the
per-field RNGs reuse the same uniforms across stress at a given seed.

Jittered-instance incomparability (honest): S0's published start set is the
parking corner (0,0) while S1 takes the farthest workspace corner from each
node. On the four *registered* geometries S1 ≤ S0, but once locations are
resampled that dominance is gone (s1c; never assert S1 ≤ S0 on seeded
instances). A majority of B1/B4 draws and nearly half of B2/B3 draws have
S1 R > S0 R, which is why several S1 curves sit *below* the corresponding S0
curve over part of the grid.

## (b) 50% crossings, B2 at deployed, B1/heavy at 1.6/1.8/2.0

Linear interpolation of accept_frac_e2e between adjacent grid members.

| curve | interpolated β at 0.5 acceptance |
|---|---|
| B1 medium path-sum | 1.482 |
| B1 medium refinement | 1.504 |
| B1 heavy path-sum | 1.853 |
| B1 heavy refinement | 1.896 |
| B2 path-sum | 1.104 |
| B2 refinement | 1.110 |
| B3 path-sum | 1.462 |
| B3 refinement | 1.479 |
| B4 path-sum | 0.706 |
| B4 refinement | 0.756 |

B2 at deployed deadlines (β=1.0): path-sum **0.170** (34/200) vs refinement
**0.040** (8/200).

B1/heavy pair:

| β | path-sum | refinement |
|---|---|---|
| 1.6 | 0.095 | 0.000 |
| 1.8 | 0.385 | 0.145 |
| 2.0 | 0.815 | 0.885 |

## (c) PROPOSAL for the TB3-5 empirical tier

**Pending Yifan's sign-off.** Derived from the curves above; no runner edits
in this prompt.

Per cell `{B1_CDR:medium, B1_CDR:heavy}`, three deadline-scale points:

| cell | β points | why |
|---|---|---|
| B1_CDR:medium | 1.0, 1.4, 1.6 | 1.0 anchors TB-1 / Table III; 1.4 is the grid member nearest the path-sum 50% crossing (1.482); 1.6 is nearest the refinement 50% crossing (1.504) |
| B1_CDR:heavy | 1.0, 1.8, 2.0 | 1.0 same TB-1 anchor; both 50% interpolants (1.853 path-sum, 1.896 refinement) sit between 1.8 and 2.0. Independent nearest-grid rounding would collapse both onto 1.8, so keep the two bracketing grid members |

`instance_seeds` **0..9** — the first ten of the grid pool, so the empirical
tier and this offline grid share instances.

Open runner parameters to settle in TB3-5 (not this prompt):

- `scripts/sweep_tb1.py --instance_seed` passthrough (frozen here).
- How the published "5 seeds" axis maps onto checkpoint seeds `{100, 300, 500}`
  × `eval_seed_base`.
- Exp-name suffix scheme for the instance-seed × deadline-scale cells.
- Expected episode budget vs the 300-episode-equivalent spec.

B1/medium and B1/heavy offline acceptance at β=1.0 is 0/200 for both tests;
the 1.0 point is still the TB-1 Table III anchor, not an offline 50% point.

## (d) Commands, SHA-256, wall time, commit

```
# captured BEFORE edits (byte-identical to p3 capture)
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/offline_test_harness.py anchor \
  --new_sweep_dir data/raw/behavioral/sweep_a1a2b1 \
  --old_results_dir data/raw/behavioral/legacy_medium \
  --out results_tb3_offline/anchor_pre_p4_run \
  > results_tb3_offline/anchor_pre_p4.txt

conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/offline_test_harness.py grid \
  --out results_tb3_offline/grid \
  --curves docs/tb3/tb3_acceptance_curves.csv \
  --frontier docs/tb3/tb3_frontier_summary.csv
# wrote results_tb3_offline/grid/tb3_offline_grid.csv (30000 rows)
# sha256 0db098e9563a72f6961e380b40448a66d15bb1ba312e5088919606b116d6c42e
# wall_s 3.886

conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/fig_tb3_f3.py \
  --curves docs/tb3/tb3_acceptance_curves.csv \
  --out figures/FIG_tb3_f3_draft.pdf

conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  -m pytest -q tests/
# 79 passed (72 at p3 + 7 in test_tb3_grid.py)

conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/offline_test_harness.py anchor ... \
  > results_tb3_offline/anchor_post_p4.txt
diff -u results_tb3_offline/anchor_pre_p4.txt results_tb3_offline/anchor_post_p4.txt
# empty
```

- Raw grid CSV is **untracked** under `results_tb3_offline/grid/` (30000 data
  rows + header). SHA-256
  `0db098e9563a72f6961e380b40448a66d15bb1ba312e5088919606b116d6c42e`.
  Wall time **3.886 s**.
- Committed aggregates: `docs/tb3/tb3_acceptance_curves.csv` (150 rows),
  `docs/tb3/tb3_frontier_summary.csv` (10 rows).
- F3 draft: `figures/FIG_tb3_f3_draft.pdf` + `figures/FIG_tb3_f3_caption.txt`.
  No empirical series, no placeholder text. Legend: "path-sum test" (solid +
  filled markers) and "per-node travel refinement" (dashed + open markers).
- `anchor` was not edited. Pre/post stdout byte-diff empty.

## Grid constants

`BETA_GRID_TB3` = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8,
2.0, 2.2, 2.4. Supersedes the design-doc "13 points" (literal enumeration had
12). 1.8 fills the B1 medium/heavy gap; 2.2/2.4 cover B1/heavy S0 saturation
(declared-ξ upper support
`(3*2110 + 1.2*70000 + 3*(sqrt(200)/0.22*1000)) / 120000 = 2.3598`).

`GRID_CELLS_TB3` = B1_CDR:medium, B1_CDR:heavy, B2_LML:medium, B3_IQI:medium,
B4_DEM:medium (B1–B4 medium + B1 heavy; light is frontier-table-only).
`GRID_SEEDS_TB3` = 0..199. Defaults `--xi-source declared`, `--tests S0,S1`.

One instance per (cell, seed) via `apply_instance_seed(nominal_instance(...),
seed)`, then `evaluate` at every `scaled(beta)`. Row order: cells, seeds
ascending, betas ascending, tests S0 then S1. Fixed `.16g` float formatting.

## File list

| file | role |
|---|---|
| `scripts/offline_test_harness.py` | `grid` subcommand, `BETA_GRID_TB3` / `GRID_CELLS_TB3` / `GRID_SEEDS_TB3` |
| `scripts/csv_schema_tb.py` | appended `TB3_CURVES_HEADER`, `TB3_FRONTIER_HEADER` |
| `scripts/fig_tb3_f3.py` | F3 draft (F1 style provenance) |
| `tests/test_tb3_grid.py` | t1–t6 + canonical row-order check |
| `docs/tb3/tb3_acceptance_curves.csv` | 150-row committed aggregate |
| `docs/tb3/tb3_frontier_summary.csv` | 10-row β* summary |
| `docs/tb3/TB3-4_report.md` | this report |
| `figures/FIG_tb3_f3_draft.pdf` | F3 draft |
| `figures/FIG_tb3_f3_caption.txt` | caption fragment |

Frozen list untouched (`envs/*`, `alg/darts_agent.py`, `alg/train_darts.py`,
`scripts/r_off_g.py`, `scripts/vg_t3_combined.py`, `scripts/sweep_tb1.py`).
`cmd_anchor` not edited.

## `git status --porcelain`

Staged only this prompt's files. `results_tb3_offline/` (raw grid, anchor
captures) not committed. Pre-existing dirty tree (`README.md`,
`docs/tb3/TB3-3_report.md`, DIAG leftovers, …) unchanged.

## Commit and tag

- message: `TB-3 P4: offline acceptance grid (5 cells x 200 instances x 15 beta) + F3 draft`
- tag: `tb3-stack-p4`
- hash: see `git rev-parse tb3-stack-p4`

Stop. Do not start TB3-5. Do not edit `sweep_tb1.py`.
