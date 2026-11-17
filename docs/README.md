# docs

Records of the four experiments added in the camera-ready version of the paper. The
files are kept as they were written during development. They are not paper text: some
use internal labels, and statements about pending decisions describe the state at the
time of writing. The supplementary material (`supplementary.pdf` at the repository
root) is the reference description of every experiment.

`certificate_interface.md` is not an experiment record. It documents the online
certificate, its cost envelopes, and the offline test, and how each can be replaced.

| Folder | Experiment | In the paper | Contents |
|---|---|---|---|
| `exp2/` | Periodic-release validation of Theorem 2(a) | Supp. Sec. IX, Table IX, Fig. 2 | pre-registered gates with appended results, configuration table, allocation maps, result table |
| `exp3/` | CBBA convergence-latency microbenchmark | Supp. Sec. X, Table X, Fig. 3 | pre-registered gates with appended results, analysis output, summary table |
| `tb3/` | Offline acceptance and over-rejection | Supp. Sec. VIII-B, Tables VII and VIII, Fig. 1 | step reports, travel-refinement lemma, aggregate CSV files |

The envelope-scaling sweep (Fig. 7; Supp. Sec. VIII-A, Table VI) has no folder here; its
code is `scripts/sweep_tb1.py`, `scripts/tb1_analysis.py`, and `scripts/fig_tb1_tradeoff.py`.

Internal labels used in file names and reports:

- `tb1`: the envelope-scaling sweep. `tb3`: the offline acceptance study. `exp2`: the
  periodic-release experiment. `exp3`: the CBBA microbenchmark.
- In `tb3/`, S0 is the path-sum offline test of the paper (Eq. (23)) and S1 is the
  per-node travel refinement mentioned in Remark 2.

Files checked by the test suite, which must stay unchanged: the pre-registered part of
`exp2/exp2_gates.md` (everything above its RESULTS heading), the pre-registered part of
`exp3/exp3_gates.md` (everything above its RESULTS heading), `tb3/TB3-7_report.md`, and
`tb3/tb3_soundness_adjudication.csv`.
