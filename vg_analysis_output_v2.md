# §V.G T1 — Auction-Latency Bound Tightness

- **δ_obs,max** (global max of `max_t_auc_ms` across 185 cells): **88.50 ms**
- **Suggested δ_auc** (next multiple of 5 above δ_obs,max): **90 ms**

  *Adjust upward (e.g., to 1.2× δ_obs,max) for a more honest WCET margin; paper §V.G T1 expects δ_auc/δ_obs ≈ 1.2-2.5.*

- **|A_k|=1 P99** (B2/medium DARTS, n=150 samples): **23.38 ms**
- **|A_k|=4 P99** (B4/medium DARTS, n=150 samples): **86.30 ms**

### Bound-to-observation ratio per benchmark (using suggested δ_auc)

| Benchmark | Ā | T̄_auc = Ā·δ_auc | P99 observed | T̄_auc / P99 |
|---|---|---|---|---|
| B1_CDR | 2 | 180 ms | 26.21 ms | **6.87×** |
| B2_LML | 3 | 270 ms | 23.38 ms | **11.55×** |
| B3_IQI | 1 | 90 ms | 23.95 ms | **3.76×** |
| B4_DEM | 4 | 360 ms | 86.30 ms | **4.17×** |

### B1 priority — auction latency vs. n (linear-growth check)

| n | Stress dir | n_seeds | P99 (ms) | mean (ms) |
|---|---|---|---|---|
| 2 | medium_n2 | 3 | 24.85 | 12.93 |
| 4 | medium | 3 | 26.21 | 10.90 |
| 6 | medium_n6 | 3 | 59.53 | 29.63 |
| 8 | medium_n8 | 3 | 78.30 | 38.05 |

# §V.G T3 — R_measured_G per (benchmark, stress, method)

Convergence window: last 1000 episodes. Uses `task_time` column.

| Benchmark | Stress | Method | n_seeds | R_measured (ms) | ± std |
|---|---|---|---|---|---|
| B1 | light | darts | 10 | 109677.8 | 1741.5 |
| B1 | light | darts_nocert | 10 | 108245.0 | 1473.5 |
| B1 | light | darts_no_incentive | 10 | 107835.1 | 3497.4 |
| B1 | light | cea_edf | 10 | 121542.9 | 972.3 |
| B1 | light | sadcher | 10 | 130005.3 | 31747.8 |
| B1 | medium | darts | 10 | 103381.5 | 1468.0 |
| B1 | medium | darts_nocert | 10 | 104330.0 | 1313.8 |
| B1 | medium | darts_no_incentive | 10 | 105149.3 | 2660.3 |
| B1 | medium | cea_edf | 10 | 117113.6 | 937.5 |
| B1 | medium | sadcher | 10 | 118507.3 | 18877.3 |
| B1 | heavy | darts | 10 | 100540.4 | 996.9 |
| B1 | heavy | darts_nocert | 10 | 100221.0 | 1024.6 |
| B1 | heavy | darts_no_incentive | 10 | 100477.6 | 1373.6 |
| B1 | heavy | cea_edf | 10 | 107469.7 | 650.6 |
| B1 | heavy | sadcher | 6 | 104415.9 | 5123.1 |
| B2_LML | medium | darts | 10 | 94403.1 | 945.1 |
| B2_LML | medium | cea_edf | 10 | 122981.4 | 1726.7 |
| B2_LML | medium | sadcher | 10 | 125066.2 | 14069.1 |
| B3_IQI | medium | darts | 10 | 121692.9 | 729.5 |
| B3_IQI | medium | cea_edf | 10 | 141606.8 | 2149.3 |
| B3_IQI | medium | sadcher | 10 | 139625.1 | 16536.2 |
| B4_DEM | medium | darts | 10 | 57059.4 | 1505.3 |
| B4_DEM | medium | cea_edf | 10 | 49758.8 | 1569.2 |
| B4_DEM | medium | sadcher | 10 | 63083.4 | 3641.6 |

# Table II / D1 — Structural Miss Breakdown

Convergence-window mean (last 1000 episodes) of per-episode counts.
Higher is worse for all four columns. SADCHER omitted: it does not classify misses.

| Benchmark | Stress | Method | task_miss% | N1 drop | N2 fallback-insuf. | N3 unreached | FF late | Schema |
|---|---|---|---|---|---|---|---|---|
| B1 | light | darts | 2.0 | 0.00 | 0.00 | — | — | legacy |
| B1 | light | darts_nocert | 1.8 | 0.00 | 0.00 | — | — | legacy |
| B1 | light | darts_no_incentive | 1.8 | 0.00 | 0.00 | 0.02 | 0.00 | new |
| B1 | light | cea_edf | 1.7 | 0.00 | 0.00 | — | — | legacy |
| B1 | medium | darts | 10.4 | 0.00 | 0.00 | — | — | legacy |
| B1 | medium | darts_nocert | 8.2 | 0.00 | 0.00 | — | — | legacy |
| B1 | medium | darts_no_incentive | 7.5 | 0.00 | 0.00 | 0.08 | 0.00 | new |
| B1 | medium | cea_edf | 15.4 | 0.00 | 0.00 | — | — | legacy |
| B1 | heavy | darts | 18.8 | 0.00 | 0.00 | — | — | legacy |
| B1 | heavy | darts_nocert | 19.6 | 0.00 | 0.00 | — | — | legacy |
| B1 | heavy | darts_no_incentive | 23.6 | 0.00 | 0.00 | 0.23 | 0.00 | new |
| B1 | heavy | cea_edf | 57.6 | 0.00 | 0.00 | — | — | legacy |
| B1 | energy_b04 | darts | 10.1 | 0.00 | 0.00 | 0.10 | 0.00 | new |
| B1 | energy_b04 | darts_nocert | 8.0 | 0.00 | 0.00 | 0.08 | 0.00 | new |
| B1 | energy_b04 | cea_edf | 36.0 | 0.00 | 0.00 | 0.36 | 0.00 | new |
| B1 | energy_b03 | darts | 5.9 | 0.00 | 0.00 | 0.06 | 0.00 | new |
| B1 | energy_b03 | darts_nocert | 7.2 | 0.00 | 0.00 | 0.07 | 0.00 | new |
| B1 | energy_b03 | cea_edf | 42.4 | 0.00 | 0.00 | 0.42 | 0.00 | new |
| B1 | energy_b02 | darts | 9.9 | 0.00 | 0.00 | 0.10 | 0.00 | new |
| B1 | energy_b02 | darts_nocert | 10.4 | 0.00 | 0.00 | 0.10 | 0.00 | new |
| B1 | energy_b02 | cea_edf | 49.6 | 0.00 | 0.00 | 0.49 | 0.00 | new |
| B2_LML | medium | darts | 2.4 | 0.00 | 0.00 | — | — | legacy |
| B2_LML | medium | cea_edf | 11.3 | 0.00 | 0.00 | — | — | legacy |
| B3_IQI | medium | darts | 1.4 | 0.00 | 0.00 | — | — | legacy |
| B3_IQI | medium | cea_edf | 13.8 | 0.00 | 0.00 | — | — | legacy |
| B4_DEM | medium | darts | 0.6 | 0.00 | 0.00 | — | — | legacy |
| B4_DEM | medium | cea_edf | 0.3 | 0.00 | 0.00 | — | — | legacy |

=== END vg_analysis.py ===
