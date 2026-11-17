# TB3-5 — empirical anchor data + feasibility table

180/180 frozen-policy replays finished clean in 27 min 56 s
(2026-08-24 09:12:44 → 09:40:40). GATES PASS. Spectrum-wide
`sum(n_miss_full_fidelity_late)=0` is the Theorem 1(ii-b) runtime
validation now holding on 60 randomized instances × 900 episodes
(jittered WCETs, resampled locations, jittered batteries), not only
the registered geometries; `sum(rt_bound_violations)=0` likewise.

Base HEAD was `ee0078e` (`tb3-stack-p5`). No source files were edited.
TB3-6 is not started.

## (a) Feasibility per (cell, β) — 10 shared instances

`feasible_e2e_all` = all 3 checkpoint seeds × 5 episodes have
`task_miss_rate == 0` (the primary D4 headline). Majority is ≥2/3.
Bitmap = instance_seeds 0..9, `1` = feasible_e2e_all.

| cell | β | e2e_all /10 | e2e_maj /10 | perjob_all /10 | mean epfrac | bitmap |
|---|---|---|---|---|---|---|
| B1_CDR:medium | 1.0 | 3 | 4 | 3 | 0.8200 | 0011100000 |
| B1_CDR:medium | 1.4 | 3 | 5 | 3 | 0.8400 | 0011100000 |
| B1_CDR:medium | 1.6 | 3 | 5 | 3 | 0.8400 | 0011100000 |
| B1_CDR:heavy | 1.0 | 1 | 3 | 1 | 0.6867 | 0010000000 |
| B1_CDR:heavy | 1.8 | 3 | 5 | 3 | 0.8267 | 0011100000 |
| B1_CDR:heavy | 2.0 | 3 | 5 | 3 | 0.8267 | 0011100000 |

B1 medium: the empirical all-clean ceiling is **3/10 at every signed-off
β**, always instances {2,3,4}. Raising the deadline from 1.0 to 1.6 does
not add all-clean instances; majority only moves 4→5. **β=1.0 is the
deployed-deadline over-rejection headline candidate**: 3/10 instances
are empirically feasible_all while both offline tests accept **0/200**
on the grid (and 0/10 on this shared pool).

B1 heavy: the ceiling is **1/10 at β=1.0** (instance 2 only) and **3/10
at 1.8 and 2.0** (instances {2,3,4}). **β=1.0 is again the over-rejection
headline candidate** (1/10 feasible_all, 0/10 offline accept). Mean
epfrac 0.69 at deployed deadlines says most episodes still succeed even
when the strict all-clean bit is off.

## (b) Offline-join preview (counts only; no rates, no CIs)

2×2 of (offline `accept_e2e`, `feasible_e2e_all`) over the 10 shared
instances. Mapping: S0 = path-sum test, S1 = per-node travel refinement.

Derived columns:
- `n_acc_infeas` = accept AND NOT feasible_all: soundness-violation
  **CANDIDATES** under the strict all-clean definition (equivalently:
  an accepted instance with at least one miss episode among 15).
- `n_rej_feas` = reject AND feasible_all: over-rejection numerator preview.

| cell | β | test | n_acc_feas | n_acc_infeas | n_rej_feas | n_rej_infeas |
|---|---|---|---|---|---|---|
| B1_CDR:medium | 1.0 | S0 | 0 | 0 | 3 | 7 |
| B1_CDR:medium | 1.0 | S1 | 0 | 0 | 3 | 7 |
| B1_CDR:medium | 1.4 | S0 | 2 | 0 | 1 | 7 |
| B1_CDR:medium | 1.4 | S1 | 1 | 0 | 2 | 7 |
| B1_CDR:medium | 1.6 | S0 | 3 | **6** | 0 | 1 |
| B1_CDR:medium | 1.6 | S1 | 3 | **5** | 0 | 2 |
| B1_CDR:heavy | 1.0 | S0 | 0 | 0 | 1 | 9 |
| B1_CDR:heavy | 1.0 | S1 | 0 | 0 | 1 | 9 |
| B1_CDR:heavy | 1.8 | S0 | 2 | **1** | 1 | 6 |
| B1_CDR:heavy | 1.8 | S1 | 1 | 0 | 2 | 7 |
| B1_CDR:heavy | 2.0 | S0 | 3 | **6** | 0 | 1 |
| B1_CDR:heavy | 2.0 | S1 | 3 | **5** | 0 | 2 |

**FINDING:** `n_acc_infeas > 0` on B1 medium β=1.6 (S0: 6, S1: 5), B1
heavy β=1.8 S0 (1), and B1 heavy β=2.0 (S0: 6, S1: 5). Adjudicate in
TB3-6. First interpretation path (TB3-4): the published test's
parking-corner start-set assumption, not a harness bug. Do not treat
these as confirmed soundness violations until the per-episode statement
and CIs exist.

Over-rejection preview (`n_rej_feas`) is concentrated at β=1.0 (medium
3, heavy 1) and shrinks as β grows, matching the offline tests starting
to accept.

## (c) Gate block (verbatim) and SHA-256

```
=== TB3-5 empirical gates ===
completeness 180/180  duplicates=0  missing=0  extra=0  PASS
sum(rt_bound_violations)=0.0  PASS
sum(n_miss_full_fidelity_late)=0.0  PASS
crash totals A1..A4=[0.0, 0.0, 0.0, 0.0]  (reported, not gated)
provenance cross-check 180/180  PASS
GATES PASS
```

PRE == POST (collector re-run, frozen inputs, byte-identical CSVs):

```
198fc60cc135ebf9e04fd7330b83e6f90cb14789cd225c473f66377818cb729c  docs/tb3/tb3_empirical_runs.csv
dd2e45999f42e555c2405477d9460c63c52e5ea6860424e3e077c398ee194844  docs/tb3/tb3_empirical_feasibility.csv
```

Consistency (throwaway pandas block):

```
PASS runs CSV has 180 rows
PASS every (cell, beta) has exactly 10 rows and n_runs == 3
PASS B1 cells at beta=1.0: all offline S0/S1 reject (0/10, matches grid 0/200)
PASS sum(rt_bound_violations)==0 and sum(n_miss_full_fidelity_late)==0
```

## (d) Inventory

- Run dirs: **180** (`ls -d results_tb3_emp/tb1_*`). The results root also
  holds `sweep_tb3_progress.log`, `sweep_tb3_console.log`, and
  `collect_gates.txt` (this prompt's tee). `du -sh results_tb3_emp`: **8.0M**.
- Exp-name scheme:
  `tb1_<bench>_<stress>_k1p0_seed<ckpt>[_b<beta>]_i<inst>`
  (`_b*` omitted at β=1.0).
- Logs: `results_tb3_emp/sweep_tb3_progress.log`,
  `results_tb3_emp/sweep_tb3_console.log`.
- Plan: 2 cells × 3 β × 10 instance_seeds × 3 ckpts `{100,300,500}` =
  180 runs; `eval_episodes=5`; `eval_seed_base=ckpt seed`; κ=1.0;
  subround sidecar off. 900 episodes vs the ~300 episode-equivalent
  spec (3× replication because measured cost is low).
- Sweep wall: start `2026-08-24 09:12:44`, last completion
  `2026-08-24 09:40:40` (span **0:27:56**). Per-run `wall_s` min **9.176** /
  mean **9.294** / max **9.536**. `grep -c ' rc=0 '` = **180**. No TIMEOUT
  on any run. One `# REFUSAL` comment is interleaved at log line 13
  (`planned experiment dirs already exist`) from a concurrent `--dry-run`
  against the live `results_tb3_emp/` after dirs existed; the 180
  completion lines are all `rc=0` and the sweep did not abort.
- `results_tb3_emp/` and `results_tb3_emp_smoke/` were not otherwise
  modified or rerun.

Commands this prompt:

```
git rev-parse --short HEAD   # ee0078e
sha256sum docs/tb3/tb3_empirical_runs.csv docs/tb3/tb3_empirical_feasibility.csv
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_empirical_collect.py --results_dir results_tb3_emp \
  --grid results_tb3_offline/grid/tb3_offline_grid.csv --out docs/tb3 \
  2>&1 | tee results_tb3_emp/collect_gates.txt
sha256sum ...   # POST == PRE
# throwaway pandas block for tables 2.1 / 2.2 / consistency
```

Commit: `TB-3 P5b: empirical anchor data + feasibility table`
Tag: `tb3-stack-p5b` (hash: `git rev-parse tb3-stack-p5b`).

## Carried forward

TB3-6 = over-rejection P(reject|feasible) with bootstrap CIs + per-episode
soundness statement + frontier ratio β*_test/β*_emp + F3 final with the
black empirical ceiling series (`fig_tb3_f3.py --empirical` input schema
already defined). After TB3-6, the remaining critical-path item before
the 8/30 freeze is EXP-2.

Stop. Do not start TB3-6.
