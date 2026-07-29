"""
Plot evaluation logs from sweep runs and per-experiment training artifacts.

- Sweep layout: ``{method}_{benchmark}_{stress}_seed{seed}_log.csv`` under ``--results_dir``,
  or nested ``{exp_tag}/darts_log.csv`` (and CEA/SADCHER equivalents).
- Theorem 1.iii: ``rho_hat`` and ``Theta_m_lower_bound`` are read from TensorBoard
  (tags ``Theorem1/rho_hat`` if logged, and ``Theorem1/Theta_m_lower_bound``) when
  ``<exp_dir>/tb`` exists (DARTS). Baselines omit TB coverage curves.
- Fallback / commitment counts: CSV columns ``n_full_fidelity``, ``n_degradable_fallback``,
  ``n_nondegradable_n1``, ``n_nondegradable_n2`` (eval rows).

Writes ``theorem1_summary.json`` into each experiment directory when using ``--scan_experiments``
or ``--write_summary_for_exp``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None  # type: ignore

# --- Metric column registry: (csv_key, tensorboard_tag, plot_title, y_axis_label) ---
# PRIMARY_COLUMNS: task miss vs rho_hat adjacency for paper-style figure groups.
PRIMARY_COLUMNS: List[Tuple[str, str, str, str]] = [
    ("task_miss_rate", "", "Task miss rate", "miss rate (-)"),
    ("rho_hat", "Theorem1/rho_hat", "Full-Fidelity Coverage Ratio rho_hat", "rho_hat (-)"),
]

COVERAGE_COLUMNS: List[Tuple[str, str, str, str]] = [
    ("rho_hat", "Theorem1/rho_hat", "Full-Fidelity Coverage Ratio rho_hat", "rho_hat (-)"),
    ("Theta_m_lower_bound", "Theorem1/Theta_m_lower_bound", "Coverage-Gradient Norm Theta_m", "||grad rho||^2 (-)"),
]

FALLBACK_COLUMNS: List[Tuple[str, str, str]] = [
    ("n_full_fidelity", "Full-Fidelity Commitments per Episode", "count"),
    ("n_degradable_fallback", "Degradable Fallback Triggers per Episode", "count"),
    ("n_nondegradable_n2", "Nondegradable Casualties (N2) per Episode", "count"),
    ("n_nondegradable_n1", "Nondegradable Drops (N1) per Episode", "count"),
]

BENCHMARKS = ("B1_CDR", "B2_LML", "B3_IQI", "B4_DEM")
METHODS_DEFAULT = ("darts", "cea_edf", "sadcher")
LOG_NAMES = ("darts_log.csv", "cea_edf_log.csv", "sadcher_log.csv")


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _strip_fieldnames(fieldnames: Optional[Sequence[str]]) -> List[str]:
    if not fieldnames:
        return []
    return [h.strip() for h in fieldnames]


def _find_log_csv(exp_dir: Path) -> Optional[Path]:
    for name in LOG_NAMES:
        p = exp_dir / name
        if p.is_file():
            return p
    return None


def _find_tb_dir(exp_dir: Path) -> Optional[Path]:
    tb = exp_dir / "tb"
    if tb.is_dir():
        return tb
    return None


def _find_config_json(exp_dir: Path) -> Optional[Path]:
    for name in ("config.json", "cea_edf_config.json", "sadcher_config.json"):
        p = exp_dir / name
        if p.is_file():
            return p
    return None


def _load_config(exp_dir: Path) -> Dict[str, Any]:
    cfg_path = _find_config_json(exp_dir)
    if cfg_path is None:
        return {}
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _v_total_from_config(cfg: Dict[str, Any]) -> int:
    tw = cfg.get("task_wcets")
    if isinstance(tw, dict) and tw:
        return len(tw)
    wid = cfg.get("workload_id")
    if wid:
        try:
            from envs.workloads import get_workload

            return len(get_workload(str(wid)).node_ids)
        except Exception:
            pass
    return 4


def _load_csv_series(csv_path: Path, column: str) -> List[Tuple[int, float]]:
    if not csv_path.is_file():
        return []
    out: List[Tuple[int, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fn = _strip_fieldnames(reader.fieldnames)
        if "episode" not in fn or column not in fn:
            return []
        # map stripped -> index for row access
        for row in reader:
            er = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            ep_s = er.get("episode", "")
            v_s = er.get(column, "")
            if ep_s in ("", "N/A") or v_s in ("", "N/A"):
                continue
            try:
                out.append((int(float(ep_s)), float(v_s)))
            except ValueError:
                continue
    out.sort(key=lambda t: t[0])
    return out


def _load_tb_series(tb_root: Path, tag: str) -> List[Tuple[int, float]]:
    if EventAccumulator is None or not tb_root.is_dir():
        return []
    try:
        ea = EventAccumulator(str(tb_root), size_guidance={}, purge_orphaned_data=False)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if tag not in tags:
            return []
        return [(int(e.step), float(e.value)) for e in ea.Scalars(tag)]
    except Exception:
        return []


def _last_n(series: List[Tuple[int, float]], n: int) -> List[Tuple[int, float]]:
    if not series:
        return []
    return series[-n:]


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def _mean_delta(series: List[Tuple[int, float]]) -> Optional[float]:
    if len(series) < 2:
        return None
    diffs = [series[i][1] - series[i - 1][1] for i in range(1, len(series))]
    return sum(diffs) / len(diffs)


def build_theorem1_summary(
    exp_dir: Path,
    log_csv: Path,
    cfg: Dict[str, Any],
    last_n: int = 100,
) -> Dict[str, Any]:
    """Aggregate last-100-episode stats + proof margins for theorem1_summary.json."""
    V = max(1, _v_total_from_config(cfg))
    tb_dir = _find_tb_dir(exp_dir)

    rho_tb = _load_tb_series(tb_dir, "Theorem1/rho_hat") if tb_dir else []
    rho_csv = _load_csv_series(log_csv, "rho_hat")
    rho_series = rho_tb if rho_tb else rho_csv
    tail_rho = [v for _, v in _last_n(rho_series, last_n)]

    theta_tb = _load_tb_series(tb_dir, "Theorem1/Theta_m_lower_bound") if tb_dir else []
    theta_csv = _load_csv_series(log_csv, "Theta_m_lower_bound")
    theta_series = theta_tb if theta_tb else theta_csv
    tail_theta = [v for _, v in _last_n(theta_series, last_n)]

    n1_series = _load_csv_series(log_csv, "n_nondegradable_n1")
    n2_series = _load_csv_series(log_csv, "n_nondegradable_n2")
    ff_series = _load_csv_series(log_csv, "n_full_fidelity")
    tail_ff = [v for _, v in _last_n(ff_series, last_n)]

    rho_mean, rho_std = _mean_std(tail_rho)
    rho_impr = _mean_delta(_last_n(rho_series, last_n))
    theta_mean, _ = _mean_std(tail_theta)

    n1_tot = sum(v for _, v in _last_n(n1_series, last_n))
    n2_tot = sum(v for _, v in _last_n(n2_series, last_n))

    frac_ff: Optional[float] = None
    if tail_ff:
        frac_ff = sum(tail_ff) / (len(tail_ff) * V)

    L = float(cfg.get("lipschitz_rho_estimate", 10.0))
    lr_eta = float(cfg.get("lr_eta", 3e-4))
    lr_phi = float(cfg.get("lr_phi", 1e-4))
    lam = float(cfg.get("lambda_rho", 1.0))
    margin_eta = 1.0 - (L * lr_eta * lam) / 0.25
    margin_phi = 1.0 - (L * lr_phi * lam) / 0.25

    return {
        "exp_dir": str(exp_dir),
        "log_csv": str(log_csv),
        "workload_cardinality_V": V,
        "rho_hat_final_mean": rho_mean,
        "rho_hat_final_std": rho_std,
        "rho_hat_improvement_per_iteration": rho_impr,
        "Theta_m_at_convergence": theta_mean,
        "n_nondegradable_n1_total": int(n1_tot),
        "n_nondegradable_n2_total": int(n2_tot),
        "fraction_full_fidelity_final": frac_ff,
        "proof_regime_eta_margin": margin_eta,
        "proof_regime_phi_margin": margin_phi,
    }


def write_theorem1_summary(exp_dir: Path, last_n: int = 100) -> Optional[Path]:
    log = _find_log_csv(exp_dir)
    if log is None:
        return None
    cfg = _load_config(exp_dir)
    summary = build_theorem1_summary(exp_dir, log, cfg, last_n=last_n)
    out = exp_dir / "theorem1_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return out


def _iter_experiment_dirs(results_root: Path) -> Iterable[Path]:
    if not results_root.is_dir():
        return
    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue
        if _find_log_csv(p) is not None:
            yield p


def _nested_exp_dir(results_dir: Path, method: str, benchmark: str, stress: str, seed: int) -> Path:
    return results_dir / f"{method}_{benchmark}_{stress}_seed{seed}"


def _resolve_log_csv(results_dir: Path, method: str, benchmark: str, stress: str, seed: int) -> Path:
    """Prefer nested ``<method>_<bench>_<stress>_seed<seed>/`` log; else flat sweep CSV filename."""
    nested = _nested_exp_dir(results_dir, method, benchmark, stress, seed)
    found = _find_log_csv(nested)
    if found is not None:
        return found
    return results_dir / f"{method}_{benchmark}_{stress}_seed{seed}_log.csv"


def plot_miss_rate_grid(
    benchmark: str,
    stress_levels: Sequence[str],
    methods: Sequence[str],
    results_dir: Path,
    seed: int,
    out_path: Path,
) -> None:
    assert plt is not None
    grid: Dict[Tuple[str, str], Optional[float]] = {}
    for m in methods:
        for s in stress_levels:
            lg = _resolve_log_csv(results_dir, m, benchmark, s, seed)
            ser = _load_csv_series(lg, "task_miss_rate") if lg.is_file() else []
            grid[(m, s)] = ser[-1][1] if ser else None

    nrows, ncols = len(methods), len(stress_levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    fig.suptitle(f"Benchmark {benchmark}: task_miss_rate (last eval row)")
    for i, m in enumerate(methods):
        for j, s in enumerate(stress_levels):
            ax = axes[i][j]
            val = grid.get((m, s))
            ax.bar([0], [0.0 if val is None else val], width=0.5, color="C0")
            ax.set_ylim(0.0, max(1.0, (val or 0.0) * 1.15))
            ax.set_xticks([])
            if i == 0:
                ax.set_title(s)
            if j == 0:
                ax.set_ylabel(m, rotation=90)
            ax.text(0, 0.02, "N/A" if val is None else f"{val:.3f}", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_theorem1_coverage_figure(
    benchmark: str,
    stress: str,
    methods: Sequence[str],
    results_dir: Path,
    seed: int,
    out_path: Path,
) -> None:
    """One figure per (benchmark, stress): rows = coverage metrics, cols = methods."""
    assert plt is not None
    nrows = len(COVERAGE_COLUMNS)
    ncols = len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.0 * nrows), squeeze=False)
    fig.suptitle(f"Theorem1_Coverage — {benchmark} / {stress}")

    for j, method in enumerate(methods):
        exp_dir = _nested_exp_dir(results_dir, method, benchmark, stress, seed)
        log_csv = _resolve_log_csv(results_dir, method, benchmark, stress, seed)
        tb_dir = _find_tb_dir(exp_dir)

        for i, (csv_key, tb_tag, title, ylab) in enumerate(COVERAGE_COLUMNS):
            ax = axes[i][j]
            series = _load_tb_series(tb_dir, tb_tag) if tb_dir else []
            if not series:
                series = _load_csv_series(log_csv, csv_key) if log_csv.is_file() else []
            if series:
                xs, ys = zip(*series)
                ax.plot(xs, ys, lw=1.2)
            else:
                ax.text(0.5, 0.5, "no series\n(TB or CSV)", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            if csv_key == "rho_hat":
                ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(ylab)
            ax.set_title(f"{method}\n{title}" if i == 0 else title, fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("episode / step")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_defer_fallback_stacked(
    benchmark: str,
    stress: str,
    methods: Sequence[str],
    results_dir: Path,
    seed: int,
    out_path: Path,
) -> None:
    """Stacked area: four commitment classes vs episode, one subplot per method."""
    assert plt is not None
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 2.8 * n_methods), squeeze=False)
    fig.suptitle(f"Defer_Fallback_Classification (stacked) — {benchmark} / {stress}")

    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]
    labels = [fc[1] for fc in FALLBACK_COLUMNS]

    for mi, method in enumerate(methods):
        ax = axes[mi][0]
        log_csv = _resolve_log_csv(results_dir, method, benchmark, stress, seed)
        if not log_csv.is_file():
            ax.set_visible(False)
            continue
        by_ep: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for idx, (col, _, _) in enumerate(FALLBACK_COLUMNS):
            for ep, v in _load_csv_series(log_csv, col):
                by_ep[ep][idx] = float(v)
        if not by_ep:
            ax.text(0.5, 0.5, "no fallback columns in CSV", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(method)
            continue
        xs_sorted = sorted(by_ep.keys())
        ys_stack = [[by_ep[e][k] for e in xs_sorted] for k in range(len(FALLBACK_COLUMNS))]
        ax.stackplot(xs_sorted, *ys_stack, labels=labels, colors=colors, alpha=0.85)
        ax.set_ylabel(f"{method}\ncount")
        ax.set_xlabel("episode (eval checkpoints)")
        if mi == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DARTS / sweep visualization: miss rates, Theorem 1 coverage, defer/fallback stacks, summaries."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="B1_CDR",
        choices=list(BENCHMARKS),
        help="Benchmark ID for sweep-style paths and figure names.",
    )
    parser.add_argument(
        "--stress_levels",
        type=str,
        default="light,medium,heavy",
        help="Comma-separated stress keys (column / combo order).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="darts,cea_edf,sadcher",
        help="Comma-separated methods.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root containing nested exp dirs or flat sweep CSVs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed suffix for sweep / nested dir names.")
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="results/vis",
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--all_benchmarks",
        action="store_true",
        help="Generate figures for every benchmark in BENCHMARKS.",
    )
    parser.add_argument(
        "--scan_experiments",
        action="store_true",
        help="Scan results_dir for subdirs with *log.csv and write theorem1_summary.json in each.",
    )
    parser.add_argument(
        "--write_summary_for_exp",
        type=str,
        default="",
        help="Absolute or project-relative path to one experiment directory; writes theorem1_summary.json only.",
    )
    parser.add_argument(
        "--summary_last_n",
        type=int,
        default=100,
        help="Number of tail episodes/rows for summary statistics.",
    )
    args = parser.parse_args()

    if plt is None:
        raise SystemExit("matplotlib is required (pip install matplotlib).")

    stress_levels = _parse_csv_list(args.stress_levels)
    methods = _parse_csv_list(args.methods)
    results_dir = (_PROJECT_ROOT / args.results_dir.replace("/", os.sep)).resolve()
    vis_out = (_PROJECT_ROOT / args.vis_output_dir.replace("/", os.sep)).resolve()
    vis_out.mkdir(parents=True, exist_ok=True)

    if args.write_summary_for_exp:
        exp = (_PROJECT_ROOT / args.write_summary_for_exp.replace("/", os.sep)).resolve()
        p = write_theorem1_summary(exp, last_n=args.summary_last_n)
        if p:
            print(f"Wrote {p}")
        else:
            raise SystemExit(f"No log CSV under {exp}")
        return

    if args.scan_experiments:
        n = 0
        for exp in _iter_experiment_dirs(results_dir):
            p = write_theorem1_summary(exp, last_n=args.summary_last_n)
            if p:
                print(f"Wrote {p}")
                n += 1
        print(f"Summaries written: {n}")
        return

    benchmarks = list(BENCHMARKS) if args.all_benchmarks else [args.benchmark]

    for bench in benchmarks:
        # Legacy miss-rate grid (all stress columns)
        plot_miss_rate_grid(bench, stress_levels, methods, results_dir, args.seed, vis_out / f"miss_rate_{bench}.png")

        for stress in stress_levels:
            plot_theorem1_coverage_figure(
                bench, stress, methods, results_dir, args.seed, vis_out / f"Theorem1_Coverage_{bench}_{stress}.png"
            )
            plot_defer_fallback_stacked(
                bench, stress, methods, results_dir, args.seed, vis_out / f"Defer_Fallback_Classification_{bench}_{stress}.png"
            )

    print(f"Figures written under {vis_out}")
    print(
        "Paper outline (evaluation): 5.1 Deadline Safety (Theorem 1.i) — cite task_miss_rate and "
        "n_nondegradable_n2/|V| from theorem1_summary.json; 5.2 Response-Time Bound (Theorem 1.ii) — "
        "sub*_rt_ms vs 2H + d_v - t_k; 5.3 Coverage (Theorem 1.iii) — rho_hat curve and Theta_m_at_convergence."
    )


if __name__ == "__main__":
    main()
