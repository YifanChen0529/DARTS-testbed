# §V.G T3 — Combined R̄^off_G vs R_measured_G

- δ_auc = 30.0 ms
- H = 1000.0 ms, J̄ = 0.0 ms
- u_i,max = 0.22 m/s, h = 50.0 ms
- start_pos = (0.0, 0.0) m (worst-case agent start)

| Benchmark | Stress | Method | R̄^off_G (ms) | R_measured_G (ms) ± std | n_seeds | Ratio (off / meas) |
|---|---|---|---|---|---|---|
| B1_CDR | light | darts | 240250 | 109678 ± 1742 | 10 | **2.19×** |
| B1_CDR | light | darts_nocert | 240250 | 108245 ± 1474 | 10 | **2.22×** |
| B1_CDR | light | darts_no_incentive | 240250 | 107835 ± 3497 | 10 | **2.23×** |
| B1_CDR | light | cea_edf | 240250 | 121543 ± 972 | 10 | **1.98×** |
| B1_CDR | light | sadcher | 240250 | 130005 ± 31748 | 10 | **1.85×** |
| B1_CDR | medium | darts | 240250 | 103382 ± 1468 | 10 | **2.32×** |
| B1_CDR | medium | darts_nocert | 240250 | 104330 ± 1314 | 10 | **2.30×** |
| B1_CDR | medium | darts_no_incentive | 240250 | 105149 ± 2660 | 10 | **2.28×** |
| B1_CDR | medium | cea_edf | 240250 | 117114 ± 938 | 10 | **2.05×** |
| B1_CDR | medium | sadcher | 240250 | 118507 ± 18877 | 10 | **2.03×** |
| B1_CDR | heavy | darts | 240250 | 100540 ± 997 | 10 | **2.39×** |
| B1_CDR | heavy | darts_nocert | 240250 | 100221 ± 1025 | 10 | **2.40×** |
| B1_CDR | heavy | darts_no_incentive | 240250 | 100478 ± 1374 | 10 | **2.39×** |
| B1_CDR | heavy | cea_edf | 240250 | 107470 ± 651 | 10 | **2.24×** |
| B1_CDR | heavy | sadcher | 240250 | 104416 ± 5123 | 6 | **2.30×** |
| B2_LML | medium | darts | 255513 | 94403 ± 945 | 10 | **2.71×** |
| B2_LML | medium | cea_edf | 255513 | 122981 ± 1727 | 10 | **2.08×** |
| B2_LML | medium | sadcher | 255513 | 125066 ± 14069 | 10 | **2.04×** |
| B3_IQI | medium | darts | 255513 | 121693 ± 730 | 10 | **2.10×** |
| B3_IQI | medium | cea_edf | 255513 | 141607 ± 2149 | 10 | **1.80×** |
| B3_IQI | medium | sadcher | 255513 | 139625 ± 16536 | 10 | **1.83×** |
| B4_DEM | medium | darts | 74810 | 57059 ± 1505 | 10 | **1.31×** |
| B4_DEM | medium | cea_edf | 74810 | 49759 ± 1569 | 10 | **1.50×** |
| B4_DEM | medium | sadcher | 74810 | 63083 ± 3642 | 10 | **1.19×** |
