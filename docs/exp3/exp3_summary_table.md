# Convergence-latency summary

All wall-time columns are measured on the loopback tier: all four agent processes executed on a single Jetson-class node over local loopback; per-round compute, serialization, and synchronization are measured; the physical wireless channel was not traversed, so any cross-machine network term is calibrated rather than measured.

| tasks | bundle length | recorded trials | rounds to convergence, mean | rounds to convergence, 50th percentile | rounds to convergence, 95th percentile | rounds to convergence, 99th percentile | rounds to convergence, maximum | round cap (bundle length times number of tasks) | maximum rounds divided by the round cap | convergence wall time, mean (milliseconds) | convergence wall time, 50th percentile (milliseconds) | convergence wall time, 95th percentile (milliseconds) | convergence wall time, 99th percentile (milliseconds) | convergence wall time, maximum (milliseconds) | rounds to convergence, 99th percentile over median | convergence wall time, 99th percentile over median | messages sent, mean | bytes sent, mean | tasks times 9.6 milliseconds | tasks times 23.38 milliseconds | tasks times 30.0 milliseconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | one | 200 | 2.325 | 2.000 | 3.000 | 4.000 | 4.000 | 4 | 1.000 | 27.787 | 26.617 | 35.163 | 38.723 | 39.758 | 2.000 | 1.455 | 39.9 | 2808.7 | 38.4 | 93.52 | 120.0 |
| 4 | ceiling of m over four | 200 | 2.310 | 2.000 | 3.000 | 4.000 | 4.000 | 4 | 1.000 | 27.317 | 26.074 | 33.747 | 37.834 | 43.140 | 2.000 | 1.451 | 39.7 | 2795.7 | 38.4 | 93.52 | 120.0 |
| 8 | one | 200 | 1.835 | 2.000 | 3.000 | 3.000 | 3.000 | 8 | 0.375 | 24.652 | 24.963 | 30.947 | 33.256 | 37.724 | 1.500 | 1.332 | 34.0 | 4849.1 | 76.8 | 187.04 | 240.0 |
| 8 | ceiling of m over four | 200 | 3.065 | 3.000 | 5.000 | 5.000 | 6.000 | 16 | 0.375 | 31.948 | 31.248 | 42.724 | 46.721 | 53.734 | 1.667 | 1.495 | 48.8 | 5629.0 | 76.8 | 187.04 | 240.0 |
| 16 | one | 200 | 1.425 | 1.000 | 2.000 | 3.000 | 3.000 | 16 | 0.188 | 22.650 | 21.045 | 28.506 | 32.620 | 33.315 | 3.000 | 1.550 | 29.1 | 8394.7 | 153.6 | 374.08 | 480.0 |
| 16 | ceiling of m over four | 200 | 4.060 | 4.000 | 6.000 | 7.000 | 8.000 | 64 | 0.125 | 38.899 | 38.437 | 52.560 | 57.201 | 57.726 | 1.750 | 1.488 | 60.7 | 12355.0 | 153.6 | 374.08 | 480.0 |
| 32 | one | 200 | 1.245 | 1.000 | 2.000 | 2.000 | 3.000 | 32 | 0.094 | 22.078 | 20.790 | 27.358 | 30.816 | 34.129 | 2.000 | 1.482 | 26.9 | 15551.6 | 307.2 | 748.16 | 960.0 |
| 32 | ceiling of m over four | 200 | 5.080 | 5.000 | 8.000 | 9.000 | 9.000 | 256 | 0.035 | 67.744 | 67.352 | 95.243 | 101.482 | 113.452 | 1.800 | 1.507 | 73.0 | 27473.4 | 307.2 | 748.16 | 960.0 |

Footnotes: 9.6 ms mean per-subround at frontier size 1 exists in the paper supplementary only, no repo artifact (T0 R5). 23.38 ms P99 from vg_analysis_output.md line 8. 30.0 ms calibrated envelope from vg_t3_combined.md line 3.
