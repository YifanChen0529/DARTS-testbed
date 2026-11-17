# TB3-7 — STOP (protocol rule 5): R1 failed on the merged 420

## (0) Merged verdict

Sweep: **240/240** rc=0, 0 REFUSAL, wall 04:47:53 → 05:24:53 (36 min 60 s;
sum `wall_s` = 2219.376, mean 9.247 s). Completeness, FFL, rt, and
provenance all PASS. **R1 fails: 407/420, n_fail=13,
max_|dev|=0.40000000000000013.** Protocol rule 5: STOP, report verbatim,
no reruns. Analysis `--p7`, F3 regeneration, pytest-as-success, and tag
`tb3-stack-p7b` were **not** run.

Gate block (verbatim from `results_tb3_emp_p7/collect_gates.txt`):

```
=== TB3-5 empirical gates ===
completeness 420/420  duplicates=0  missing=0  extra=0  PASS
sum(rt_bound_violations)=0.0  PASS
sum(n_miss_full_fidelity_late)=0.0  PASS
crash totals A1..A4=[0.0, 0.0, 0.0, 0.0]  (reported, not gated)
provenance cross-check 420/420  PASS
R1 identity 407/420  n_fail=13  max_|dev|=0.40000000000000013  FAIL
GATES FAIL
```

All 13 R1 failures are **new p7 runs** (plan=`p7`, deadline scale 1.0,
instance seeds in {11, 12, 16, 32, 49}). The original 180 default-plan
runs remain R1-clean. FFL = N1 = N2 = rt = 0 on every failing row; N3
job-count exceeds the binary episode-miss rate (`dev = tmr − jobcount <
0`), i.e. at least one miss episode dropped **two or more** jobs. That
is the R1 finding the P6b amendment said would stay TIMING_VIOLATION
(the dual of a jobcount=0 miss episode).

FAILING RUNS (verbatim):

```
B1_CDR:heavy  s=1.0 i16c100 tmr=0.6 FFL=0 N1=0 N2=0 N3=1.0 resid=1.4 dev=-0.4
B1_CDR:heavy  s=1.0 i49c100 tmr=0.8 FFL=0 N1=0 N2=0 N3=1.0 resid=2.2 dev=-0.2
B1_CDR:heavy  s=1.0 i11c300 tmr=0.8 FFL=0 N1=0 N2=0 N3=1.0 resid=2.2 dev=-0.2
B1_CDR:heavy  s=1.0 i12c300 tmr=0.2 FFL=0 N1=0 N2=0 N3=0.4 resid=0.4 dev=-0.2
B1_CDR:heavy  s=1.0 i32c300 tmr=0.4 FFL=0 N1=0 N2=0 N3=0.6 resid=1.0 dev=-0.2
B1_CDR:heavy  s=1.0 i49c300 tmr=0.4 FFL=0 N1=0 N2=0 N3=0.6 resid=1.0 dev=-0.2
B1_CDR:heavy  s=1.0 i49c500 tmr=0.2 FFL=0 N1=0 N2=0 N3=0.4 resid=0.4 dev=-0.2
B1_CDR:medium s=1.0 i16c100 tmr=0.4 FFL=0 N1=0 N2=0 N3=0.8 resid=0.8 dev=-0.4
B1_CDR:medium s=1.0 i49c100 tmr=0.6 FFL=0 N1=0 N2=0 N3=0.8 resid=1.6 dev=-0.2
B1_CDR:medium s=1.0 i11c300 tmr=0.6 FFL=0 N1=0 N2=0 N3=0.8 resid=1.6 dev=-0.2
B1_CDR:medium s=1.0 i32c300 tmr=0.2 FFL=0 N1=0 N2=0 N3=0.4 resid=0.4 dev=-0.2
B1_CDR:medium s=1.0 i49c300 tmr=0.2 FFL=0 N1=0 N2=0 N3=0.4 resid=0.4 dev=-0.2
B1_CDR:medium s=1.0 i49c500 tmr=0.4 FFL=0 N1=0 N2=0 N3=0.6 resid=1.0 dev=-0.2
```

`(a)` feasibility table, `(b)` over-rejection table, `(c)` frontier
lower bounds, and `(d)` paper-facing paragraph were **not** produced.

## (e) Audit-chain note

p5b/p6/p6b CSVs were not rewritten. SHAs after the failed collect match
the pre-collect snapshot:

```
198fc60cc135ebf9e04fd7330b83e6f90cb14789cd225c473f66377818cb729c  docs/tb3/tb3_empirical_runs.csv
dd2e45999f42e555c2405477d9460c63c52e5ea6860424e3e077c398ee194844  docs/tb3/tb3_empirical_feasibility.csv
866c21560c3d58a04d14d28fba8f8c1c81d34093c33aca57bb90a30bd56fb33b  docs/tb3/tb3_empirical_ceiling.csv
733be3a50a7409315f24e3d60773d466411614bfdac400970068d892fd3057c1  docs/tb3/tb3_overrejection.csv
4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9  docs/tb3/tb3_soundness_adjudication.csv
2be50bcdc3fd0a86560ad4bed9f64170459cd823c4cb6ef10e13a97ce8acfe88  docs/tb3/tb3_soundness_adjudication_v2.csv
```

Collector still wrote diagnostic `docs/tb3/tb3_empirical_runs_p7.csv`
(420 rows) and `docs/tb3/tb3_empirical_feasibility_p7.csv` (140 rows)
before returning FAIL. They are **not** paper-facing; do not treat them
as a P7b deliverable.

Commands:

```
grep -cE ' rc=0 ' results_tb3_emp_p7/sweep_tb3_progress.log   # 240
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_empirical_collect.py \
  --results_dir results_tb3_emp,results_tb3_emp_p7 \
  --plans default,p7 \
  --grid results_tb3_offline/grid/tb3_offline_grid.csv \
  --out docs/tb3 2>&1 | tee results_tb3_emp_p7/collect_gates.txt
# collect_wall_s 8.465; rc=1
```

HEAD remains `9c15b31` (`tb3-stack-p7`). No `tb3-stack-p7b` tag.
TB-4 / EXP-2 were not started.
