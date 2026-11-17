# DARTS^rt

Code, data, and supplementary material for **"DARTS^rt: A Real-Time MARL Framework with
Bounded Auction Latency and Schedulability"** by Yifan Chen and Cong Liu (RTSS 2026).
The artifact passed the RTSS 2026 artifact evaluation; the evaluated version of this
repository is the git tag `rtss26-ae`.

This package reproduces every evaluation element of the paper — Figures 3–6 and
Supplementary Tables II–V — from the included raw experiment logs, and provides a
reduced-scale end-to-end demo that runs the full DARTS^rt stack (coordinator + four
agents, auction, certificates, incentive update) on commodity hardware in under a
minute. **No robot, Jetson, or network hardware is required.**

The paragraph above and Sections 1 to 8 describe the evaluated version. The camera-ready
version of the paper regroups the panels of Figures 3 to 5 and adds Figure 7 and
Supplementary Sections VIII to X. Section 9 maps the figure numbering and lists the
added experiments.

- Code: this repository. The default branch is the camera-ready version; the git tag `rtss26-ae` is the evaluated version.
- Data: `data/` (≈12 MB, 769 curated episode-log CSVs + manifests)
- Contact: Yifan Chen, UC Riverside, ychen1320@ucr.edu
- License: see `LICENSE` at the repository root
- Supplementary material: [`supplementary.pdf`](supplementary.pdf) at the repository root
- Citation: Section 10 and `CITATION.cff`

---

## 1. Requirements

| Item | Requirement |
|---|---|
| OS | Ubuntu 24.04 (any recent x86-64 Linux works; also runs on arm64) |
| Python | **>= 3.11** (evaluator target: 3.12) |
| RAM / disk | 4 GB RAM, < 1 GB disk |
| GPU / robot hardware | **None** (CPU-only; all hardware-derived measurements ship as data) |
| Network | Only for `pip install` during setup |

All results in the paper were produced on Jetson Orin Nano (arm64) hardware; the
artifact reproduces them on plain x86 from the shipped logs. Runtimes quoted below
were measured on the Orin Nano and are conservative for modern x86 machines.

## 2. Setup (≈5 minutes)

```bash
python3 -m venv ae_venv
source ae_venv/bin/activate

# Step 1: CPU-only torch from the PyTorch CPU wheel index
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Step 2: everything else (pinned)
pip install -r requirements-x86.txt
```

Sanity check:

```bash
python -c "import torch, grpc, pandas, matplotlib; print('env OK')"
make test        # fast unit tests (certificate math with hand-computed answers, ~6 s)
```

## 3. Quick demo (Tier B, ≈1 minute)

```bash
make demo
# equivalent to:
# python run_artifact_demo.py --benchmark B1 --stress medium --episodes 200 --seed 0
```

This runs a reduced-scale (200-episode) end-to-end experiment in simulator mode —
the same execution mode that produced all behavioral results in the paper — and
prints a summary (task miss rate, RT-bound violations, per-agent crash rates, CSV
path). Expected: `rt_bound_violations = 0` for the DARTS family (the Theorem 1
empirical check). Measured wall time: **43.5 s** on the Orin Nano reference machine.

Every flag is independently variable (`--benchmark B1..B4`, `--stress`, `--method
darts|darts_nocert|darts_no_incentive|cea_edf|sadcher`, `--episodes`, `--seed`,
`--eval_interval`, `--eval_batch`) — see `python run_artifact_demo.py --help`.
Reduced-scale runs demonstrate trends; the paper's exact numbers regenerate from the
shipped raw logs via the Makefile (next section).

## 4. Full reproduction (≈1 minute)

```bash
make verify all          # add PY=<python> if your interpreter is not `python3`
```

`verify` recomputes Supplementary Table III from the shipped tree and checks it
against the packaged reference grid (must print **Overall: PASS**). The eight
element targets then regenerate every evaluation element into `out/`:

| Target | Paper element | Output in `out/` |
|---|---|---|
| `fig3` | Fig. 3 (left: B1/heavy headline; right: bound tightness) | `FIG0_headline_b1heavy.pdf`, `FIG3_t3_tightness.pdf` |
| `fig4` | Fig. 4 (B1 miss-rate grid + certified coverage, 10 seeds) | `FIG1_b1_grid.pdf` |
| `fig5` | Fig. 5 (DBench-4 generalization; auction timing; agent scaling) | `FIG2_generalization_scaling.pdf` |
| `fig6` | Fig. 6 (B1/heavy admission-infeasibility heatmap) | `FIG4_d1_heatmap.pdf` |
| `tab2` | Supp. Table II (offline bound tightness) | `vg_t3_combined.md` |
| `tab3` | Supp. Table III (miss-rate grid, 5 methods × DBench-4) | `sweep_summary_*.csv` |
| `tab4` | Supp. Table IV (auction-latency analysis, §V.G T1) | `vg_analysis_output.md` |
| `tab5` | Supp. Table V (admission-state decomposition) | `recovered_n3_ffl.md` |

Measured wall time for `make verify all`: **47.1 s** total on the reference machine.

### Reproduction fidelity, in one paragraph

Every DARTS^rt value in the paper reproduces exactly from the shipped data.
Ablation and baseline cells may differ slightly from the printed values, within
cross-seed variation (see `DATA.md` §7). `make verify` recomputes the full grid
from the shipped data and must PASS. Fig. 5's bottom (timing) panels carry the
paper's measured values as explicit constants alongside the raw timing CSVs; see
`DATA.md` §8. The Table V / Fig. 6 chain reproduces the committed published
output **byte-identically** (modulo one path line).

## 5. Tests

```bash
make test        # fast: certificate unit tests (hand-computed Eq. 11 cases), summarizer fixture
make test-all    # adds the slow end-to-end demo schema test (~15 s)
```

## 6. Repository structure

```
run_artifact_demo.py        # Tier-B demo entry point (see §3)
Makefile                    # verify / fig3..fig6 / tab2..tab5 / all / demo / test / clean
DEVELOPMENT.md              # per-file annotated repository guide (development README)
requirements-x86.txt        # pinned CPU-only environment (torch installed separately)
alg/                        # DARTS^rt, baselines, training loop, stress profiles
envs/                       # simulator environment, workloads, transports (gRPC unused here)
scripts/                    # analysis + figure generators (all take explicit path flags)
  summarize_sweep.py        #   Table III summarizer      (--results_dir/--out_dir)
  vg_analysis.py            #   §V.G T1 analysis          (--new_sweep_dir/--old_results_dir)
  vg_t3_combined.py         #   bound-tightness table     (same flags + physics params)
  r_off_g.py                #   offline bound computation (no data inputs)
  recover_63col.py          #   Table V recovery          (--new_sweep_dir)
  fig0..fig4*.py            #   figure generators         (--*_dir/--input_md/--out_dir)
  ae_curate_data.py         #   builds data/ from raw result trees (already run)
  ae_spotcheck_tab3.py      #   Table III verification gate (used by `make verify`)
  ae_scan_timing.py         #   timing-column provenance scanner (audit tool)
tests/                      # pytest suite (`make test`)
data/                       # curated experiment logs + manifests (see DATA.md)
out/                        # regenerated elements (created by make)
```

The purpose of each script is documented in its module docstring. A fuller,
per-file annotated layout (development-oriented, including the Jetson/dev
environment) is in `DEVELOPMENT.md`.

## 7. Data documentation

`DATA.md` documents the full 63-column episode-log schema, directory layouts and
seed conventions, per-element provenance, the ten-cell deviation disclosure, and
the timing-column semantics (what is measured, on which hardware, in which mode).

## 8. Troubleshooting

- **`make` uses the wrong Python** → pass `PY=/path/to/python` (e.g. your venv's).
  The system `python3` may lack matplotlib.
- **`ModuleNotFoundError: torch`** → Step 1 of §2 was skipped; torch installs from
  the PyTorch CPU index, not from `requirements-x86.txt`.
- **Demo prints `[WARN]` on `rt_bound_violations`** → expected for the baselines
  (CEA-EDF / SADCHER); the Theorem-1 guarantee applies to the DARTS family only.
- **Regenerated `vg_*.md` differ slightly from the committed repo-root copies** →
  expected and documented; see `DATA.md` §7 (the committed copies are the frozen
  paper-time outputs).

## 9. Camera-ready version: figure numbering and added experiments

Sections 3 to 8 use the figure numbers of the evaluated version. The camera-ready paper
shows the same panels in a different grouping. The supplementary table numbers II to V
are unchanged.

| Camera-ready element | Regenerate with | Panel file in `out/` |
|---|---|---|
| Fig. 3, left (B1 heavy-stress miss rate) | `make fig3` | `FIG0_headline_b1heavy.pdf` |
| Fig. 3, right (B1 stress sweep and certified coverage) | `make fig4` | `FIG1_b1_grid.pdf` |
| Fig. 4(a), 4(b) (auction time by frontier size; agent scaling) | `make fig5` | `FIG2_generalization_scaling.pdf`, timing panels |
| Fig. 4(c) (offline bound versus measured response time) | `make fig3` | `FIG3_t3_tightness.pdf` |
| Fig. 5 (miss rate across DBench-4) | `make fig5` | `FIG2_generalization_scaling.pdf`, generalization panels |
| Fig. 6 (admission-infeasibility breakdown) | `make fig6` | `FIG4_d1_heatmap.pdf` |
| Supp. Tables II to V | `make tab2` to `make tab5` | as listed in Section 4 |

The camera-ready version adds four experiments. Their generated figures and tables are
part of the repository; `docs/README.md` describes the records kept for each of them.

| Camera-ready element | Experiment | Code | Generated files |
|---|---|---|---|
| Fig. 7; Supp. Sec. VIII-A, Table VI | Certificate conservatism under envelope scaling | `scripts/sweep_tb1.py`, `scripts/tb1_analysis.py`, `scripts/fig_tb1_tradeoff.py` | `figures/FIG_tb1_tradeoff.pdf` |
| Supp. Sec. VIII-B, Tables VII and VIII, Fig. 1 | Offline acceptance and over-rejection | `scripts/offline_test_harness.py`, `scripts/sweep_tb3.py`, `scripts/tb3_empirical_collect.py`, `scripts/tb3_analysis.py`, `scripts/fig_tb3_f3.py` | `figures/FIG_tb3_f3.pdf`, `docs/tb3/*.csv` |
| Supp. Sec. IX, Table IX, Fig. 2 | Periodic-release validation of Theorem 2(a) | `alg/periodic_release.py`, `scripts/sweep_exp2.py`, `scripts/thm2a_analysis.py`, `scripts/fig_exp2_response_cdf.py` | `figures/FIG_exp2_response_cdf.pdf`, `docs/exp2/exp2_supplementary_table.csv` |
| Supp. Sec. X, Table X, Fig. 3 | CBBA convergence-latency microbenchmark | `benchmarks/cbba_microbench/`, `scripts/cbba_analysis.py`, `scripts/fig_exp3_tconv_cdf.py` | `figures/FIG_exp3_tconv_cdf.pdf`, `docs/exp3/exp3_summary_table.csv` |

Directories added since the evaluated version: `benchmarks/` (the CBBA microbenchmark),
`docs/` (experiment records), and `figures/` (generated figures with their caption files).
`git diff --stat rtss26-ae` lists every difference between the two versions.

The certificate interface that the conclusion of the paper refers to (online certificate,
cost envelopes, offline test) is documented in `docs/certificate_interface.md`. The raw
result trees of the four added experiments, which are the inputs of the analysis and
figure scripts listed above, and the trained policy checkpoints are not part of the
repository. Tests that need the checkpoints are skipped with a stated reason.

Equation numbers in code comments, docstrings, and help texts follow earlier drafts of the
paper and can differ from the camera-ready numbering. For example, the offline path-sum
test is Eq. (23) of the paper and `eq. 26` in the help text of
`scripts/offline_test_harness.py`.

## 10. Citation

```bibtex
@inproceedings{chen2026dartsrt,
  author    = {Chen, Yifan and Liu, Cong},
  title     = {{DARTS}$^{\mathrm{rt}}$: A Real-Time {MARL} Framework with Bounded Auction Latency and Schedulability},
  booktitle = {2026 IEEE Real-Time Systems Symposium (RTSS)},
  year      = {2026}
}
```

`CITATION.cff` at the repository root carries the same reference.
