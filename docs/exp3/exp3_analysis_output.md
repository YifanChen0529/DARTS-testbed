## GATE ADJUDICATION

Registrations quoted from `docs/exp3/exp3_gates.md`: G-V strict (`valid == 1` on 100% of recorded rows); G-C strict (`rounds <= l_t * m` on every recorded row); G-R strict (`ref_agree == 1` on every recorded row); G-S informational (first-half vs second-half median `t_total_ms` in trial order, flag beyond about 10%); G-F report-only (overhead floor; per-config loopback compute floor context).

### G-V (strict) PASS: valid==1 on 1600/1600 recorded rows.

### G-C (strict) PASS: rounds <= l_t*m on 1600/1600 recorded rows.

### G-R (strict) PASS: ref_agree==1 on 1600/1600 recorded rows.

### G-S (informational)
Per configuration, first-half vs second-half median t_total_ms (trial order). Flag if |percent delta| exceeds about 10%.
- m4_lt1 m=4 l_t=1 n=200: first-half median 27.2470 ms, second-half median 26.4210 ms, delta -3.03% (second half lower).
- m4_ltceil m=4 l_t=1 n=200: first-half median 25.7316 ms, second-half median 26.4370 ms, delta 2.74% (second half higher).
- m8_lt1 m=8 l_t=1 n=200: first-half median 24.9632 ms, second-half median 24.9020 ms, delta -0.25% (second half lower).
- m8_ltceil m=8 l_t=2 n=200: first-half median 31.2856 ms, second-half median 31.2324 ms, delta -0.17% (second half lower).
- m16_lt1 m=16 l_t=1 n=200: first-half median 20.7668 ms, second-half median 21.2346 ms, delta 2.25% (second half higher).
- m16_ltceil m=16 l_t=4 n=200: first-half median 38.0794 ms, second-half median 38.5691 ms, delta 1.29% (second half higher).
- m32_lt1 m=32 l_t=1 n=200: first-half median 20.9113 ms, second-half median 20.7057 ms, delta -0.98% (second half lower).
- m32_ltceil m=32 l_t=8 n=200: first-half median 68.5806 ms, second-half median 65.1785 ms, delta -4.96% (second half lower).

### G-F (report-only)
Overhead floor from 20 null-trial median t_total_ms: 16.2674 ms.
Cross-check against t_overhead_floor_ms column: 16.2674 ms (agreement within 0.1 ms).
Per-config loopback compute-floor context (no threshold):
- m4_lt1: median summed per-node compute 0.9296 ms; min t_total_ms 19.0686 ms.
- m4_ltceil: median summed per-node compute 0.9446 ms; min t_total_ms 18.0198 ms.
- m8_lt1: median summed per-node compute 1.0472 ms; min t_total_ms 17.9361 ms.
- m8_ltceil: median summed per-node compute 1.9182 ms; min t_total_ms 18.9545 ms.
- m16_lt1: median summed per-node compute 1.1725 ms; min t_total_ms 17.9410 ms.
- m16_ltceil: median summed per-node compute 7.4980 ms; min t_total_ms 23.6505 ms.
- m32_lt1: median summed per-node compute 1.6714 ms; min t_total_ms 18.6941 ms.
- m32_ltceil: median summed per-node compute 65.4971 ms; min t_total_ms 38.3596 ms.
## PER-CONFIG TABLE

All wall-time columns are measured on the loopback tier: all four agent processes executed on a single Jetson-class node over local loopback; per-round compute, serialization, and synchronization are measured; the physical wireless channel was not traversed, so any cross-machine network term is calibrated rather than measured.

| tasks | bundle length | n | rounds mean | rounds P50 | rounds P95 | rounds P99 | rounds max | cap (l_t*m) | max/cap | t_total mean | t_total P50 | t_total P95 | t_total P99 | t_total max | rounds P99/median | t_total P99/median | msgs mean | bytes mean | m x 9.6 ms | m x 23.38 ms | m x 30.0 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | one | 200 | 2.325 | 2.000 | 3.000 | 4.000 | 4.000 | 4 | 1.000 | 27.787 | 26.617 | 35.163 | 38.723 | 39.758 | 2.000 | 1.455 | 39.9 | 2808.7 | 38.4 | 93.52 | 120.0 |
| 4 | ceiling of m over four | 200 | 2.310 | 2.000 | 3.000 | 4.000 | 4.000 | 4 | 1.000 | 27.317 | 26.074 | 33.747 | 37.834 | 43.140 | 2.000 | 1.451 | 39.7 | 2795.7 | 38.4 | 93.52 | 120.0 |
| 8 | one | 200 | 1.835 | 2.000 | 3.000 | 3.000 | 3.000 | 8 | 0.375 | 24.652 | 24.963 | 30.947 | 33.256 | 37.724 | 1.500 | 1.332 | 34.0 | 4849.1 | 76.8 | 187.04 | 240.0 |
| 8 | ceiling of m over four | 200 | 3.065 | 3.000 | 5.000 | 5.000 | 6.000 | 16 | 0.375 | 31.948 | 31.248 | 42.724 | 46.721 | 53.734 | 1.667 | 1.495 | 48.8 | 5629.0 | 76.8 | 187.04 | 240.0 |
| 16 | one | 200 | 1.425 | 1.000 | 2.000 | 3.000 | 3.000 | 16 | 0.188 | 22.650 | 21.045 | 28.506 | 32.620 | 33.315 | 3.000 | 1.550 | 29.1 | 8394.7 | 153.6 | 374.08 | 480.0 |
| 16 | ceiling of m over four | 200 | 4.060 | 4.000 | 6.000 | 7.000 | 8.000 | 64 | 0.125 | 38.899 | 38.437 | 52.560 | 57.201 | 57.726 | 1.750 | 1.488 | 60.7 | 12355.0 | 153.6 | 374.08 | 480.0 |
| 32 | one | 200 | 1.245 | 1.000 | 2.000 | 2.000 | 3.000 | 32 | 0.094 | 22.078 | 20.790 | 27.358 | 30.816 | 34.129 | 2.000 | 1.482 | 26.9 | 15551.6 | 307.2 | 748.16 | 960.0 |
| 32 | ceiling of m over four | 200 | 5.080 | 5.000 | 8.000 | 9.000 | 9.000 | 256 | 0.035 | 67.744 | 67.352 | 95.243 | 101.482 | 113.452 | 1.800 | 1.507 | 73.0 | 27473.4 | 307.2 | 748.16 | 960.0 |

Footnotes (anchor provenance, T0 R5):
- 9.6 ms mean per-subround at frontier size 1 exists in the paper supplementary only, no repo artifact (T0 R5).
- 23.38 ms P99 from vg_analysis_output.md line 8.
- 30.0 ms calibrated envelope from vg_t3_combined.md line 3.
