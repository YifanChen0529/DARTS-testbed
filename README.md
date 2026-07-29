# DARTS^rt — RTSS 2026 Artifact (Paper #117)

Artifact for **"DARTS^rt: A Real-Time MARL Framework with Bounded Auction Latency and
Schedulability"** (RTSS 2026, Paper #117).

This package reproduces every evaluation element of the paper — Figures 3–6 and
Supplementary Tables II–V — from the included raw experiment logs, and provides a
reduced-scale end-to-end demo that runs the full DARTS^rt stack (coordinator + four
agents, auction, certificates, incentive update) on commodity hardware in under a
minute. **No robot, Jetson, or network hardware is required.**

- Code: git tag `rtss26-ae` <!-- TODO(YC): confirm final tag name after advisor sign-off -->
- Data: `data/` (≈12 MB, 769 curated episode-log CSVs + manifests)
- Contact: Yifan Chen, UC Riverside <!-- TODO(YC): confirm contact email -->
- License: see `LICENSE` at the repository root

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