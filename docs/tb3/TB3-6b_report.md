# TB3-6b — adjudication amendment (episode-binary miss semantics)

## (0) Three-line verdict

R1 holds on **180/180** runs (`n_fail=0`, `max_|dev|=0`). v2 census:
**97 consistent + 23 energy_scope_miss + 0 TIMING_VIOLATION**. The P6
identity treated `task_miss_rate` as a job count; `evaluate_agents`
writes it as a per-episode mean of a **binary** end-to-end miss
(`alg/train_darts.py:1321`) while FFL/N1/N2/N3 are per-episode means of
**job** counts (`:1369–1376`, construction `:904–922`). The 23 v1
TIMING_VIOLATION rows are that mis-specification's mechanical artifact;
they remain in `tb3_soundness_adjudication.csv` and as `verdict_v1`.

Base HEAD was `c5162a1` (`tb3-stack-p6`). v1 CSV SHA unchanged
`4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9`.
`FIG_tb3_f3.pdf` was not regenerated: the caption is a sidecar
`.txt` and is not baked into the PDF text (`pdftotext` shows axis
labels and legend only).

R1 derivation: with a binary per-episode e2e flag and 5 eval episodes,
`residual = sum_ep (4*flag_ep − jobcount_ep)/5`. Then
`residual − 3 * task_miss_rate = sum_ep (flag_ep − jobcount_ep)/5`, so
R1 holds iff every miss episode has exactly one missed job. A
jobcount=0 miss episode would contribute residual step 0.8 and fail
R1 — a real timing finding. None occurred. N1 = N2 = 0 on every v2
row; `n_miss_episodes == n3_jobs` on the 23 flips.

## (a) v1 → v2 flip table (the 23 rows)

| cell | s | test | instance | n_miss_episodes | n3_jobs | verdict_v1 | verdict_v2 |
|---|---|---|---|---|---|---|---|
| B1_CDR:medium | 1.6 | S0 | 0 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S0 | 1 | 3 | 3 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S0 | 5 | 8 | 8 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S0 | 6 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S0 | 8 | 4 | 4 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S0 | 9 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S1 | 0 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S1 | 5 | 8 | 8 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S1 | 6 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S1 | 8 | 4 | 4 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:medium | 1.6 | S1 | 9 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 1.8 | S0 | 8 | 4 | 4 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 0 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 1 | 3 | 3 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 5 | 8 | 8 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 6 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 8 | 4 | 4 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S0 | 9 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S1 | 0 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S1 | 5 | 8 | 8 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S1 | 6 | 1 | 1 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S1 | 8 | 4 | 4 | TIMING_VIOLATION | energy_scope_miss |
| B1_CDR:heavy | 2.0 | S1 | 9 | 2 | 2 | TIMING_VIOLATION | energy_scope_miss |

All 23 have `r1_holds=1`. All 97 v1 `consistent` rows stay `consistent`.

## (b) Corrected adjudication paragraph (paper record)

Zero timing violations across all accepted instances. Every residual
miss on an accepted-but-not-all-clean instance is a single N3
energy-unreached job per miss episode (R1; N1 = N2 = FFL = 0;
`rt_bound_violations` = 0). No episode overruns the mission deadline
with all jobs individually on time. Theorem 1(ii-b) spectrum-wide
`sum(n_miss_full_fidelity_late)=0` stands. The response-time test
contains no energy term; energy admission is owned by the runtime
energy predicate and reserve sizing.

The P6 `verdict` column is retained as the historical mis-specified
rule. Paper text uses `verdict_v2`.

## (c) Files, SHAs, commands, commit

```
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  scripts/tb3_analysis.py
# wall_s 12.792; v1 CSV not rewritten
conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$PWD python3 \
  -m pytest -q
# 96 passed
```

SHA-256:

```
4de308a69c4a2aef868825b0b93b493720bc6c1c8736678ccb813e76c2836ed9  docs/tb3/tb3_soundness_adjudication.csv
2be50bcdc3fd0a86560ad4bed9f64170459cd823c4cb6ef10e13a97ce8acfe88  docs/tb3/tb3_soundness_adjudication_v2.csv
a88791d1d28f8f643ca84873f784191a7478d6eb125f8cf57a268b0cd625956c  figures/FIG_tb3_f3_caption.txt
```

PDF unchanged (git `4f8721068352db7d8c74f284d39be1691b28ae95`, same as
`tb3-stack-p6`). Caption sidecar updated; symbol gates pass (no
`beta` / `β` / `S0` / `S1`).

Commit: `TB-3 P6b: adjudication amendment (episode-binary miss semantics), zero timing violations`
Tag: `tb3-stack-p6b`.
