# Recovered N3 / FFL breakdown for corrupted B1 cells

Canonical header source: `results_a1a2b1_20260521/exp_1/B1/energy_b02/darts_log.csv`
Method: line-by-line parse of mixed-schema CSVs; keep rows with exactly 63 fields;
        re-parse via canonical header; aggregate across recovered rows × seeds.

| Stress | Method | n_files | n_recovered_rows | task_miss% | N1 drop | N2 fallback-insuf. | N3 unreached | FF late |
|---|---|---|---|---|---|---|---|---|
| light | darts | 10/10 | 50 | 2.8 | 0.000 | 0.000 | 0.028 | 0.000 |
| light | darts_nocert | 10/10 | 50 | 2.2 | 0.000 | 0.000 | 0.022 | 0.000 |
| light | cea_edf | 10/10 | 50 | 3.2 | 0.000 | 0.000 | 0.032 | 0.000 |
| light | sadcher | 10/10 | 50 | 2.0 | — | — | — | — |
| medium | darts | 10/10 | 50 | 9.6 | 0.000 | 0.000 | 0.096 | 0.000 |
| medium | darts_nocert | 10/10 | 50 | 9.4 | 0.000 | 0.000 | 0.094 | 0.000 |
| medium | cea_edf | 10/10 | 50 | 32.8 | 0.000 | 0.000 | 0.328 | 0.000 |
| medium | sadcher | 10/10 | 50 | 31.4 | — | — | — | — |
| heavy | darts | 10/10 | 50 | 23.0 | 0.000 | 0.000 | 0.220 | 0.000 |
| heavy | darts_nocert | 10/10 | 50 | 22.6 | 0.000 | 0.000 | 0.212 | 0.000 |
| heavy | cea_edf | 10/10 | 50 | 83.0 | 0.000 | 0.000 | 0.818 | 0.000 |
| heavy | sadcher | 10/10 | 50 | 56.4 | — | — | — | — |
