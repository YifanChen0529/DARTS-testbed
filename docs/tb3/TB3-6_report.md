# TB3-6 — over-rejection + pre-registered soundness adjudication + F3 final

## (0) Three-line verdict

The pre-registered rule produces **23 TIMING_VIOLATION** rows and **0
energy_scope_miss** (97 consistent of 120). Global `sum(FFL)=0` and
`sum(rt_bound_violations)=0`; every violation is partition-residual on an
accepted instance that is not all-clean. At deployed deadlines both tests
over-reject every feasible instance: B1 medium 3/3 = 1.00 [1.00, 1.00]
(`n_undefined=292`); B1 heavy 1/1 = 1.00 [1.00, 1.00]
(`n_undefined=3472`). F3 final is `figures/FIG_tb3_f3.pdf`: TB3-4 offline
curves unchanged, black empirical all-clean ceiling (points + bootstrap
95% CI bars) on B1 medium and B1 heavy only; symbol gates pass.

Base HEAD was `10defec` (`tb3-stack-p5b`). Pure post-processing; no
simulation; no harness edits. `FIG_tb3_f3_draft.pdf` is byte-identical to
`eb70c854d072cf857d8daf0b3b3670eed830d6cd`.

Counter semantics (`alg/train_darts.py` `evaluate_agents`, eval-only
appended row): `task_miss_rate` (:1321) is a per-episode mean of a binary
end-to-end miss; the four miss counters (:1369–1376) are per-episode means
of job counts (construction :904–922; 4 jobs per episode on B1); FFL uses
the per-job `absolute_deadline_ms`, not the mission deadline;
`rt_bound_violations` (:1274, :1344) is a run total. Pre-registered
identity: `residual := task_miss_rate * 4 − (FFL + N1 + N2 + N3)`,
tolerance `1e-9`.

Adjudication paragraph (from the CSV, not softened): the rule does **not**
classify `n_acc_infeas` as energy-scope. All 23 accept-and-not-all-clean
instance×test cells trip TIMING_VIOLATION solely through
`partition_residual > 1e-9` on at least one of their three runs. Miss mass
on those runs is N3-unreached plus the residual (N1 = N2 = FFL = 0). The
residual is the binary end-to-end miss scaled by four jobs minus the
per-job miss counts — the channel the prompt associates with completions
after the mission deadline but within the per-job deadline (T0 H6; on
heavy, `D_rel = 150000·s` exceeds `D_e2e = 120000·s`). The response-time
test contains no energy term; that does not stop the residual rule from
firing. The earlier parking-corner late-completion hypothesis is ruled out
by `FFL == 0` (late completions after the per-job deadline would be
required). `energy_scope_miss` is empty because it requires NOT
`timing_violation`.

## (a) Adjudication and partition-residual census

Verdict counts (120 rows = 2 cells × 3 s-points × 2 tests × 10 instances):

| verdict | n |
|---|---|
| consistent | 97 |
| energy_scope_miss | 0 |
| TIMING_VIOLATION | 23 |

TIMING_VIOLATION table (verbatim; FFL/N1/N2 sums are 0 on every row):

| cell | s | test | instance | n3_sum | resid_max |
|---|---|---|---|---|---|
| B1_CDR:medium | 1.6 | S0 | 0 | 0.40 | 0.60 |
| B1_CDR:medium | 1.6 | S0 | 1 | 0.60 | 1.20 |
| B1_CDR:medium | 1.6 | S0 | 5 | 1.60 | 1.80 |
| B1_CDR:medium | 1.6 | S0 | 6 | 0.20 | 0.60 |
| B1_CDR:medium | 1.6 | S0 | 8 | 0.80 | 1.20 |
| B1_CDR:medium | 1.6 | S0 | 9 | 0.20 | 0.60 |
| B1_CDR:medium | 1.6 | S1 | 0 | 0.40 | 0.60 |
| B1_CDR:medium | 1.6 | S1 | 5 | 1.60 | 1.80 |
| B1_CDR:medium | 1.6 | S1 | 6 | 0.20 | 0.60 |
| B1_CDR:medium | 1.6 | S1 | 8 | 0.80 | 1.20 |
| B1_CDR:medium | 1.6 | S1 | 9 | 0.20 | 0.60 |
| B1_CDR:heavy | 1.8 | S0 | 8 | 0.80 | 1.20 |
| B1_CDR:heavy | 2.0 | S0 | 0 | 0.40 | 0.60 |
| B1_CDR:heavy | 2.0 | S0 | 1 | 0.60 | 1.20 |
| B1_CDR:heavy | 2.0 | S0 | 5 | 1.60 | 1.80 |
| B1_CDR:heavy | 2.0 | S0 | 6 | 0.20 | 0.60 |
| B1_CDR:heavy | 2.0 | S0 | 8 | 0.80 | 1.20 |
| B1_CDR:heavy | 2.0 | S0 | 9 | 0.40 | 1.20 |
| B1_CDR:heavy | 2.0 | S1 | 0 | 0.40 | 0.60 |
| B1_CDR:heavy | 2.0 | S1 | 5 | 1.60 | 1.80 |
| B1_CDR:heavy | 2.0 | S1 | 6 | 0.20 | 0.60 |
| B1_CDR:heavy | 2.0 | S1 | 8 | 0.80 | 1.20 |
| B1_CDR:heavy | 2.0 | S1 | 9 | 0.40 | 1.20 |

These 23 rows are exactly the TB3-5 `n_acc_infeas` cells (medium 1.6 S0: 6,
S1: 5; heavy 1.8 S0: 1; heavy 2.0 S0: 6, S1: 5). Accepted and all-clean
instances (18) have residual 0 and verdict `consistent`. Rejected
instances with residual > 0 stay `consistent` because `accept_e2e` is
false.

Partition-residual census on all 180 runs: **96 nonzero**, all positive,
**max = 2.4**, none negative. `sum(N1)=0`, `sum(N2)=0`, `sum(N3)=34.8`
(per-episode means summed). Runs with residual > tol (`i<seed>c<ckpt>`):

| cell | s | n_nonzero / 30 | runs (residual) |
|---|---|---|---|
| B1_CDR:heavy | 1.0 | 20 | i0c100=1.2, i0c300=0.6, i1c100=1.8, i1c300=1.2, i1c500=0.6, i3c300=1.2, i4c500=0.6, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=1.2, i6c500=2.4, i7c100=1.2, i7c300=2.4, i7c500=2.4, i8c100=1.8, i8c300=1.2, i8c500=1.2, i9c100=1.2, i9c500=1.2 |
| B1_CDR:heavy | 1.8 | 15 | i0c100=0.6, i0c300=0.6, i1c100=0.6, i1c300=1.2, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=0.6, i7c100=0.6, i7c300=1.2, i7c500=1.8, i8c100=0.6, i8c300=1.2, i8c500=0.6, i9c100=1.2 |
| B1_CDR:heavy | 2.0 | 15 | i0c100=0.6, i0c300=0.6, i1c100=0.6, i1c300=1.2, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=0.6, i7c100=0.6, i7c300=1.2, i7c500=1.8, i8c100=0.6, i8c300=1.2, i8c500=0.6, i9c100=1.2 |
| B1_CDR:medium | 1.0 | 16 | i0c100=0.6, i0c300=0.6, i1c100=0.6, i1c300=1.2, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=0.6, i7c100=0.6, i7c300=1.2, i7c500=1.8, i8c100=0.6, i8c300=1.2, i8c500=0.6, i9c100=0.6, i9c500=1.2 |
| B1_CDR:medium | 1.4 | 15 | i0c100=0.6, i0c300=0.6, i1c100=0.6, i1c300=1.2, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=0.6, i7c100=0.6, i7c300=0.6, i7c500=1.8, i8c100=0.6, i8c300=1.2, i8c500=0.6, i9c100=0.6 |
| B1_CDR:medium | 1.6 | 15 | i0c100=0.6, i0c300=0.6, i1c100=0.6, i1c300=1.2, i5c100=1.8, i5c300=1.2, i5c500=1.8, i6c100=0.6, i7c100=0.6, i7c300=0.6, i7c500=1.8, i8c100=0.6, i8c300=1.2, i8c500=0.6, i9c100=0.6 |

## (b) Over-rejection and empirical ceiling

`over_rejection_hat = n_rej_feas / n_feasible_e2e_all`. Bootstrap: resample
the 10 instance ids with replacement, B=10000,
`np.random.default_rng(20260825)` (fresh generator per cell × s-point ×
test). Resamples with 0 feasible instances are dropped from that CI and
counted in `n_undefined`. Point estimates match the prompt 2.2 fractions
recomputed from the committed feasibility CSV.

| cell | test | s | n_feas | n_rej_feas | hat | 95% CI | n_undefined |
|---|---|---|---|---|---|---|---|
| B1_CDR:medium | S0 | 1.0 | 3 | 3 | 1.000 | [1.000, 1.000] | 292 |
| B1_CDR:medium | S1 | 1.0 | 3 | 3 | 1.000 | [1.000, 1.000] | 292 |
| B1_CDR:medium | S0 | 1.4 | 3 | 1 | 0.333 | [0.000, 1.000] | 292 |
| B1_CDR:medium | S1 | 1.4 | 3 | 2 | 0.667 | [0.000, 1.000] | 292 |
| B1_CDR:medium | S0 | 1.6 | 3 | 0 | 0.000 | [0.000, 0.000] | 292 |
| B1_CDR:medium | S1 | 1.6 | 3 | 0 | 0.000 | [0.000, 0.000] | 292 |
| B1_CDR:heavy | S0 | 1.0 | 1 | 1 | 1.000 | [1.000, 1.000] | 3472 |
| B1_CDR:heavy | S1 | 1.0 | 1 | 1 | 1.000 | [1.000, 1.000] | 3472 |
| B1_CDR:heavy | S0 | 1.8 | 3 | 1 | 0.333 | [0.000, 1.000] | 292 |
| B1_CDR:heavy | S1 | 1.8 | 3 | 2 | 0.667 | [0.000, 1.000] | 292 |
| B1_CDR:heavy | S0 | 2.0 | 3 | 0 | 0.000 | [0.000, 0.000] | 292 |
| B1_CDR:heavy | S1 | 2.0 | 3 | 0 | 0.000 | [0.000, 0.000] | 292 |

Empirical ceiling CSV is the strict all-clean definition (schema consumed
by `fig_tb3_f3 --empirical`). Majority is report-only.

| cell | s | strict n/10 | strict frac [95% CI] | majority n/10 | majority frac [95% CI] |
|---|---|---|---|---|---|
| B1_CDR:medium | 1.0 | 3 | 0.30 [0.00, 0.60] | 4 | 0.40 [0.10, 0.70] |
| B1_CDR:medium | 1.4 | 3 | 0.30 [0.00, 0.60] | 5 | 0.50 [0.20, 0.80] |
| B1_CDR:medium | 1.6 | 3 | 0.30 [0.00, 0.60] | 5 | 0.50 [0.20, 0.80] |
| B1_CDR:heavy | 1.0 | 1 | 0.10 [0.00, 0.30] | 3 | 0.30 [0.00, 0.60] |
| B1_CDR:heavy | 1.8 | 3 | 0.30 [0.00, 0.60] | 5 | 0.50 [0.20, 0.80] |
| B1_CDR:heavy | 2.0 | 3 | 0.30 [0.00, 0.60] | 5 | 0.50 [0.20, 0.80] |

## (c) Frontier lower bounds (no interpolation)

`s*_test` = grid `load_e2e` at deployed deadlines (s = 1.0, declared
WCETs). `s*_emp` is left-censored at the smallest *tested* s-point where
`feasible_e2e_all` holds, so `s*_test / s*_emp` is a **lower bound** equal
to `s*_test / s_emp_ub`. Medium {2,3,4} and heavy {2} are feasible at
deployed deadlines (`s*_emp ≤ 1.0`); heavy {3,4} first all-clean at 1.8
(`s*_emp ≤ 1.8`). A lower bound below 1 means the bound is uninformative
in the conservative direction, not that the true ratio is below 1.

| cell | instance | test | s*_test | s_emp_ub | ratio lb | censored at |
|---|---|---|---|---|---|---|
| B1_CDR:medium | 2 | S0 | 1.5220 | 1.0 | 1.5220 | 1.0 |
| B1_CDR:medium | 2 | S1 | 1.4728 | 1.0 | 1.4728 | 1.0 |
| B1_CDR:medium | 3 | S0 | 1.2855 | 1.0 | 1.2855 | 1.0 |
| B1_CDR:medium | 3 | S1 | 1.3829 | 1.0 | 1.3829 | 1.0 |
| B1_CDR:medium | 4 | S0 | 1.3463 | 1.0 | 1.3463 | 1.0 |
| B1_CDR:medium | 4 | S1 | 1.4629 | 1.0 | 1.4629 | 1.0 |
| B1_CDR:heavy | 2 | S0 | 1.9025 | 1.0 | 1.9025 | 1.0 |
| B1_CDR:heavy | 2 | S1 | 1.8410 | 1.0 | 1.8410 | 1.0 |
| B1_CDR:heavy | 3 | S0 | 1.6068 | 1.8 | 0.8927 | 1.8 |
| B1_CDR:heavy | 3 | S1 | 1.7287 | 1.8 | 0.9604 | 1.8 |
| B1_CDR:heavy | 4 | S0 | 1.6828 | 1.8 | 0.9349 | 1.8 |
| B1_CDR:heavy | 4 | S1 | 1.8287 | 1.8 | 1.0159 | 1.8 |

## (d) Symbol-discipline mapping (writing-phase standing reference)

| internal | paper wording |
|---|---|
| CSV/code `beta` | axis text: the words **deadline scale** only; if a formula needs a symbol, reserve **s**. Never β — paper β is agent battery (cert-eng). |
| TB-1 additive certificate conservatism (was κ in older notes) | **zeta**. Zero uses in the current draft/supp. |
| `kappa` / κ | **never** for conservatism or deadline scale. Taken by the bid's spatial reachability discount. |
| `xi` / WCET columns | **ξ_{i,v}**, aligned with the paper's WCET symbol. |
| S0 | **path-sum test** (workspace-diameter travel envelope). |
| S1 | **per-node travel refinement** (farthest-corner eccentricity, Lemma). |
| `u_min` vs `u_max` | the S1 lemma divides by **u_max**. The deployed fleet is homogeneous at 0.22 m/s, so u_min = u_max = 0.22. |

Grep gate (test f3): caption file and the fig script's rendered text (caption,
legend, axis labels, panel titles) contain none of `beta`, `β`, `S0`, `S1`.
Internal CSV keys are unchanged.

## (e) Commands, SHAs, wall, commit

F3 `--empirical` path edits (`scripts/fig_tb3_f3.py`; draft PDF untouched):
legend grows one worded entry "empirical feasibility (all-clean, 10
instances)" when `--empirical` is set; caption file gets the two-sentence
addendum on that path only (offline-only runs still write the TB3-4
caption, so t6 is unchanged); empirical series is points + CI bars
(`fmt="ko"`), not a step, and is drawn only on B1 medium / B1 heavy;
`rendered_strings()` is the paper-facing blob the f3 grep gate uses.

```
sha256sum results_tb3_offline/grid/tb3_offline_grid.csv
# 0db098e9563a72f6961e380b40448a66d15bb1ba312e5088919606b116d6c42e
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_analysis.py
# wall_s 13.090
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/fig_tb3_f3.py \
  --curves docs/tb3/tb3_acceptance_curves.csv \
  --empirical docs/tb3/tb3_empirical_ceiling.csv \
  --out figures/FIG_tb3_f3.pdf
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  -m pytest -q
# 91 passed
```

SHA-256 of committed CSVs this prompt:

```
733be3a50a7409315f24e3d60773d466411614bfdac400970068d892fd3057c1  docs/tb3/tb3_overrejection.csv
4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9  docs/tb3/tb3_soundness_adjudication.csv
866c21560c3d58a04d14d28fba8f8c1c81d34093c33aca57bb90a30bd56fb33b  docs/tb3/tb3_empirical_ceiling.csv
```

Inputs unchanged:

```
198fc60cc135ebf9e04fd7330b83e6f90cb14789cd225c473f66377818cb729c  docs/tb3/tb3_empirical_runs.csv
dd2e45999f42e555c2405477d9460c63c52e5ea6860424e3e077c398ee194844  docs/tb3/tb3_empirical_feasibility.csv
```

Commit: `TB-3 P6: over-rejection + soundness adjudication + F3 final`
Tag: `tb3-stack-p6`.

## Carried forward

TB-3 complete pending the writing phase. TB-4 (bound-slack waterfall, zero
new simulation) next. EXP-2 remains the critical-path item before the 8/30
freeze.
