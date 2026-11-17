# TB3-7b — directional identity gate + expanded sample (n=50 at deployed deadlines)

## (0) Three-line verdict

G1–G6 **PASS** on the merged 420. Adjudication census (P6b `verdict_v2`,
280 rows): **257 consistent + 23 energy_scope_miss + 0 TIMING_VIOLATION**.
Strict R1 on every run was an empirical regularity of the first 60
instances, over-registered in P7 as a global data gate; the amendment
keeps a directional gate (positive `dev` fails) and strict identity only
on accepted-instance runs, where the soundness claim has force.

Gate block v2 (verbatim from `results_tb3_emp_p7/collect_gates_v2.txt`):

```
=== TB3-7b empirical gates ===
G1 completeness 420/420  duplicates=0  missing=0  extra=0  PASS
G2 sum(rt_bound_violations)=0.0  PASS
G3 sum(n_miss_full_fidelity_late)=0.0  PASS
crash totals A1..A4=[0.0, 0.0, 0.0, 0.0]  (reported, not gated)
G4 provenance cross-check 420/420  PASS
G5 directional identity (dev<=+1e-09) n_pos=0/420  max_signed_dev=0.0  PASS
G6 strict identity on accepted-instance runs n_checked=69 n_fail=0  PASS
INFO strict-R1 census 407/420  n_fail=13  max_|dev|=0.40000000000000013  (not gated)
INFO detection-limit: aggregate accounting cannot exclude a zero-channel episode cancelled by a multi-job episode within the same run; medium is structurally immune (d_v equals the mission deadline there, so such an episode would raise FFL, which is zero), and every accepted-instance run satisfies strict R1, where the claim has force.
INFO strict-R1 failure table (cell, scale, instance, ckpt, tmr, N3, dev):
  B1_CDR:heavy s=1.0 i16 c100 tmr=0.6 N3=1.0 dev=-0.3999999999999999
  B1_CDR:heavy s=1.0 i49 c100 tmr=0.8 N3=1.0 dev=-0.20000000000000018
  B1_CDR:heavy s=1.0 i11 c300 tmr=0.8 N3=1.0 dev=-0.20000000000000018
  B1_CDR:heavy s=1.0 i12 c300 tmr=0.2 N3=0.4 dev=-0.20000000000000007
  B1_CDR:heavy s=1.0 i32 c300 tmr=0.4 N3=0.6 dev=-0.20000000000000018
  B1_CDR:heavy s=1.0 i49 c300 tmr=0.4 N3=0.6 dev=-0.20000000000000018
  B1_CDR:heavy s=1.0 i49 c500 tmr=0.2 N3=0.4 dev=-0.20000000000000007
  B1_CDR:medium s=1.0 i16 c100 tmr=0.4 N3=0.8 dev=-0.40000000000000013
  B1_CDR:medium s=1.0 i49 c100 tmr=0.6 N3=0.8 dev=-0.19999999999999996
  B1_CDR:medium s=1.0 i11 c300 tmr=0.6 N3=0.8 dev=-0.19999999999999996
  B1_CDR:medium s=1.0 i32 c300 tmr=0.2 N3=0.4 dev=-0.20000000000000007
  B1_CDR:medium s=1.0 i49 c300 tmr=0.2 N3=0.4 dev=-0.20000000000000007
  B1_CDR:medium s=1.0 i49 c500 tmr=0.4 N3=0.6 dev=-0.20000000000000018
GATES PASS
```

Echo asserts: seeds 10..49 at scale 1.0 are all offline-rejected (160
adjudication rows, `accept_e2e=1` count 0); every rejected row is
`consistent`; every empirically feasible instance at deployed deadlines
is uncertified (holds on the n=50 sample). Grid SHA
`0db098e9563a72f6961e380b40448a66d15bb1ba312e5088919606b116d6c42e`.

## (a) Feasibility (strict all-clean primary; majority sensitivity)

`feasible_e2e_all` = all 3 checkpoints × 5 episodes have
`task_miss_rate == 0`. Intervals are exact two-sided 95% Clopper-Pearson
(primary). Relaxed-scale rows are the original n=10; deployed rows are
n=50.

| cell | s | n | strict | strict frac [CP 95%] | majority | majority frac [CP 95%] |
|---|---|---|---|---|---|---|
| B1_CDR:medium | 1.0 | 50 | 13 | 0.26 [0.146, 0.403] | 24 | 0.48 [0.337, 0.626] |
| B1_CDR:medium | 1.4 | 10 | 3 | 0.30 [0.067, 0.652] | 5 | 0.50 [0.187, 0.813] |
| B1_CDR:medium | 1.6 | 10 | 3 | 0.30 [0.067, 0.652] | 5 | 0.50 [0.187, 0.813] |
| B1_CDR:heavy | 1.0 | 50 | 2 | 0.04 [0.005, 0.137] | 12 | 0.24 [0.131, 0.382] |
| B1_CDR:heavy | 1.8 | 10 | 3 | 0.30 [0.067, 0.652] | 5 | 0.50 [0.187, 0.813] |
| B1_CDR:heavy | 2.0 | 10 | 3 | 0.30 [0.067, 0.652] | 5 | 0.50 [0.187, 0.813] |

Deployed-deadline all-clean seeds: medium
{2, 3, 4, 10, 12, 17, 21, 24, 26, 29, 31, 33, 36}; heavy {2, 33}.

## (b) Over-rejection

`over_rejection_hat = n_rej_feas / n_feasible_e2e_all`. CP primary;
bootstrap (B=10000, rng 20260825) secondary. Path-sum = S0; per-node
refinement = S1.

| cell | test | s | n | n_feas | n_rej_feas | hat | CP 95% | boot 95% | n_undefined |
|---|---|---|---|---|---|---|---|---|---|
| B1_CDR:medium | S0 | 1.0 | 50 | 13 | 13 | 1.00 | [0.753, 1.00] | [1.00, 1.00] | 0 |
| B1_CDR:medium | S1 | 1.0 | 50 | 13 | 13 | 1.00 | [0.753, 1.00] | [1.00, 1.00] | 0 |
| B1_CDR:medium | S0 | 1.4 | 10 | 3 | 1 | 0.33 | [0.008, 0.906] | [0.00, 1.00] | 292 |
| B1_CDR:medium | S1 | 1.4 | 10 | 3 | 2 | 0.67 | [0.094, 0.992] | [0.00, 1.00] | 292 |
| B1_CDR:medium | S0 | 1.6 | 10 | 3 | 0 | 0.00 | [0.00, 0.708] | [0.00, 0.00] | 292 |
| B1_CDR:medium | S1 | 1.6 | 10 | 3 | 0 | 0.00 | [0.00, 0.708] | [0.00, 0.00] | 292 |
| B1_CDR:heavy | S0 | 1.0 | 50 | 2 | 2 | 1.00 | [0.158, 1.00] | [1.00, 1.00] | 1339 |
| B1_CDR:heavy | S1 | 1.0 | 50 | 2 | 2 | 1.00 | [0.158, 1.00] | [1.00, 1.00] | 1339 |
| B1_CDR:heavy | S0 | 1.8 | 10 | 3 | 1 | 0.33 | [0.008, 0.906] | [0.00, 1.00] | 292 |
| B1_CDR:heavy | S1 | 1.8 | 10 | 3 | 2 | 0.67 | [0.094, 0.992] | [0.00, 1.00] | 292 |
| B1_CDR:heavy | S0 | 2.0 | 10 | 3 | 0 | 0.00 | [0.00, 0.708] | [0.00, 0.00] | 292 |
| B1_CDR:heavy | S1 | 2.0 | 10 | 3 | 0 | 0.00 | [0.00, 0.708] | [0.00, 0.00] | 292 |

At deployed deadlines both tests reject every empirically feasible
instance (13 of 13 medium, 2 of 2 heavy).

## (c) Extended frontier lower bounds

`s*_test` from the raw grid at deployed deadlines. `s*_emp` left-censored
at the smallest tested scale where `feasible_e2e_all` holds; here every
row is feasible at scale 1.0. 15 instances × 2 tests = 30 rows.

| cell | instance | test | s*_test | s_emp_ub | ratio lb | censored at |
|---|---|---|---|---|---|---|
| B1_CDR:medium | 2 | S0 | 1.5220 | 1.0 | 1.5220 | 1.0 |
| B1_CDR:medium | 2 | S1 | 1.4728 | 1.0 | 1.4728 | 1.0 |
| B1_CDR:medium | 3 | S0 | 1.2855 | 1.0 | 1.2855 | 1.0 |
| B1_CDR:medium | 3 | S1 | 1.3829 | 1.0 | 1.3829 | 1.0 |
| B1_CDR:medium | 4 | S0 | 1.3463 | 1.0 | 1.3463 | 1.0 |
| B1_CDR:medium | 4 | S1 | 1.4629 | 1.0 | 1.4629 | 1.0 |
| B1_CDR:medium | 10 | S0 | 1.5157 | 1.0 | 1.5157 | 1.0 |
| B1_CDR:medium | 10 | S1 | 1.4820 | 1.0 | 1.4820 | 1.0 |
| B1_CDR:medium | 12 | S0 | 1.3511 | 1.0 | 1.3511 | 1.0 |
| B1_CDR:medium | 12 | S1 | 1.6244 | 1.0 | 1.6244 | 1.0 |
| B1_CDR:medium | 17 | S0 | 1.7144 | 1.0 | 1.7144 | 1.0 |
| B1_CDR:medium | 17 | S1 | 1.6018 | 1.0 | 1.6018 | 1.0 |
| B1_CDR:medium | 21 | S0 | 1.5350 | 1.0 | 1.5350 | 1.0 |
| B1_CDR:medium | 21 | S1 | 1.5302 | 1.0 | 1.5302 | 1.0 |
| B1_CDR:medium | 24 | S0 | 1.4847 | 1.0 | 1.4847 | 1.0 |
| B1_CDR:medium | 24 | S1 | 1.5526 | 1.0 | 1.5526 | 1.0 |
| B1_CDR:medium | 26 | S0 | 1.6290 | 1.0 | 1.6290 | 1.0 |
| B1_CDR:medium | 26 | S1 | 1.5919 | 1.0 | 1.5919 | 1.0 |
| B1_CDR:medium | 29 | S0 | 1.5590 | 1.0 | 1.5590 | 1.0 |
| B1_CDR:medium | 29 | S1 | 1.5852 | 1.0 | 1.5852 | 1.0 |
| B1_CDR:medium | 31 | S0 | 1.7200 | 1.0 | 1.7200 | 1.0 |
| B1_CDR:medium | 31 | S1 | 1.5946 | 1.0 | 1.5946 | 1.0 |
| B1_CDR:medium | 33 | S0 | 1.5027 | 1.0 | 1.5027 | 1.0 |
| B1_CDR:medium | 33 | S1 | 1.4655 | 1.0 | 1.4655 | 1.0 |
| B1_CDR:medium | 36 | S0 | 1.5924 | 1.0 | 1.5924 | 1.0 |
| B1_CDR:medium | 36 | S1 | 1.5703 | 1.0 | 1.5703 | 1.0 |
| B1_CDR:heavy | 2 | S0 | 1.9025 | 1.0 | 1.9025 | 1.0 |
| B1_CDR:heavy | 2 | S1 | 1.8410 | 1.0 | 1.8410 | 1.0 |
| B1_CDR:heavy | 33 | S0 | 1.8784 | 1.0 | 1.8784 | 1.0 |
| B1_CDR:heavy | 33 | S1 | 1.8319 | 1.0 | 1.8319 | 1.0 |

## (d) Paper-facing paragraph

At deployed deadlines the path-sum test and the per-node travel
refinement accept 0 of 200 offline instances on each B1 cell. Of 50
empirically evaluated instances, 13 of 50 on B1 medium (exact 95%
Clopper-Pearson interval [0.146, 0.403]) and 2 of 50 on B1 heavy
([0.005, 0.137]) are all-clean feasible under frozen-policy replay (3
checkpoints × 5 episodes). Neither test accepts any of those feasible
instances: the tests do not certify the missions the policy completes.
Over-rejection on that feasible subset is 13 of 13 (medium) and 2 of 2
(heavy). F3 error bars are the same Clopper-Pearson intervals; the
deployed-deadline point aggregates 50 instances and the two relaxed
points 10 each.

## (e) Miss-composition note and scoped P6b correction

13 runs have `dev < −tol` (strict-R1 failures): **7 heavy, 6 medium**,
all at scale 1.0, seeds {11, 12, 16, 32, 49}, FFL = N1 = N2 = rt = 0.
Dev granularity is 0.2 = one job / five episodes; max `|dev|` = 0.4
implies two extra N3 jobs in that run, so at most three jobs lost in a
single episode (one baseline miss job plus two extras). That is a
fork-join cascade: an upstream N3 death leaves successors unreachable,
or two agents exhaust in one episode. Positive `dev` (zero-channel miss)
occurs **zero** times.

The P6b sentence “fail only through **single** energy-unreached jobs”
holds on **accepted-instance** runs (G6, 69/69 strict R1), not
population-wide. F3 caption drops “single” accordingly.

## (f) Audit chain

STOP record `docs/tb3/TB3-7_report.md` committed verbatim, SHA-256
`3a39bc6a60f33b2f5916cfba00a37f0329b7ceabd99eb89c9426b4f18486a45a`.
p5b/p6/p6b CSVs unchanged:

```
198fc60cc135ebf9e04fd7330b83e6f90cb14789cd225c473f66377818cb729c  docs/tb3/tb3_empirical_runs.csv
dd2e45999f42e555c2405477d9460c63c52e5ea6860424e3e077c398ee194844  docs/tb3/tb3_empirical_feasibility.csv
866c21560c3d58a04d14d28fba8f8c1c81d34093c33aca57bb90a30bd56fb33b  docs/tb3/tb3_empirical_ceiling.csv
733be3a50a7409315f24e3d60773d466411614bfdac400970068d892fd3057c1  docs/tb3/tb3_overrejection.csv
4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9  docs/tb3/tb3_soundness_adjudication.csv
2be50bcdc3fd0a86560ad4bed9f64170459cd823c4cb6ef10e13a97ce8acfe88  docs/tb3/tb3_soundness_adjudication_v2.csv
```

This prompt (no simulation rerun):

```
abca80446e59186f1f7a7177b0f9d723911d85ea43becbe7b6ded24d2acddf2b  docs/tb3/tb3_empirical_runs_p7.csv
2d0419ac87643fffeedadea8ce6041c2baaf4d4e5610e503a388fbbc9e88b8c2  docs/tb3/tb3_empirical_feasibility_p7.csv
51741457bb707687086b579c67fc36d4e54b59cb1b4e1c982b8f81a0de2695a2  docs/tb3/tb3_empirical_ceiling_p7.csv
bf8230e8c3bd33da59555d2edddad19a1bf8377fefa26c7ad030cd44feaf5a98  docs/tb3/tb3_overrejection_p7.csv
4a712189d22c0edde4f6fc1e8caf49c6a24eee3d03366448cc3043de37f2cb65  docs/tb3/tb3_soundness_adjudication_p7.csv
bfaf7f73999b16d7672d71981f5a1147ad5e209649197d97659af3ca5f412b93  docs/tb3/tb3_frontier_lb_p7.csv
2424a054a8fc5192bc98eea9615c430336b912a4bf177184e9712d782830b3c1  figures/FIG_tb3_f3.pdf
a75140b8acb05cf3143a259e0632472e46043cbb47aeb3b98ddf6980a5ea47f9  figures/FIG_tb3_f3_caption.txt
```

Commands:

```
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_empirical_collect.py \
  --results_dir results_tb3_emp,results_tb3_emp_p7 \
  --plans default,p7 \
  --grid results_tb3_offline/grid/tb3_offline_grid.csv \
  --out docs/tb3 2>&1 | tee results_tb3_emp_p7/collect_gates_v2.txt
# collect_wall_s 8.517; GATES PASS
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_analysis.py --p7 \
  --runs docs/tb3/tb3_empirical_runs_p7.csv \
  --feas docs/tb3/tb3_empirical_feasibility_p7.csv \
  --grid results_tb3_offline/grid/tb3_offline_grid.csv \
  --out docs/tb3
# analysis_wall_s 11.623
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/fig_tb3_f3.py \
  --curves docs/tb3/tb3_acceptance_curves.csv \
  --empirical docs/tb3/tb3_empirical_ceiling_p7.csv \
  --out figures/FIG_tb3_f3.pdf
# f3_wall_s 7.142
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 -m pytest -q
# 107 passed
```

Sweep wall (unchanged evidence): 2026-08-25 04:47:53 → 05:24:53, 240/240
rc=0, mean `wall_s` 9.247. Base HEAD `9c15b31` (`tb3-stack-p7`).

Commit: `TB-3 P7b: directional identity gate + expanded sample (n=50 at deployed deadlines) + CP intervals + F3`
Tag: `tb3-stack-p7b`.
TB-4 / EXP-2 were not started.

## P7c amendment (presentation only)

Stale strings that survived P7b:
- in-figure legend: `empirical feasibility (all-clean, 10 instances)`
- caption: `among 10 shared instances` (alongside the 50/10 sentence)
- y-label: `offline acceptance ratio`

Edits (no data, no analysis, no simulation):
1. legend label → `empirical feasibility (all-clean)`
2. shared y-axis label → `fraction of instances`
3. caption sidecar replaced (no “10 instances”; “50 instances” once; bars remain exact 95% Clopper-Pearson)

SHA-256 after regenerating `figures/FIG_tb3_f3.pdf` from unchanged `tb3_empirical_ceiling_p7.csv`:

```
564b5921887d2055d0586f9607d4e8413311199fa6ac88ad84a93dc5b12900ee  figures/FIG_tb3_f3.pdf
c138ca333a51f29c57354de6e532e8aba3b5b3592b470a53c05a1e867e1dbf57  figures/FIG_tb3_f3_caption.txt
```
