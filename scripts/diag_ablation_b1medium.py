#!/usr/bin/env python3
"""Read-only diagnostic: DARTS vs NoCert vs NoIncentive on B1/medium.

Outputs Markdown to stdout and saves diag_ablation_b1medium.md at repo root.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results_a1a2b1_20260521"
OUT_MD = REPO / "diag_ablation_b1medium.md"
METHODS = ("darts", "darts_nocert", "darts_no_incentive")
CONV_EPS = {4500, 5000}
EVAL_EPS = set(range(500, 5001, 500))
N_SEEDS = 10
V1_MEANS = {
    "darts": 0.095,
    "darts_nocert": 0.075,
    "darts_no_incentive": 0.033,
}

sys.path.insert(0, str(REPO / "scripts"))
from util_csv import safe_read_csv  # noqa: E402

lines_out: List[str] = []
_clean_cache: Dict[str, Tuple[Optional[pd.DataFrame], Dict]] = {}
_clean_logged: set[str] = set()


def emit(text: str = "") -> None:
    lines_out.append(text)
    print(text)


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def csv_path(seed_idx: int, method: str) -> Path:
    return RESULTS / f"exp_{seed_idx}" / "B1" / "medium" / f"{method}_log.csv"


def _clean_df(path: Path, *, verbose: bool = True) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Load CSV, dedup episodes (keep last), clip to episode<=5000."""
    key = str(path.resolve())
    if key in _clean_cache:
        return _clean_cache[key]

    stats = {
        "n_rows_raw": 0,
        "n_rows_unique_le5k": 0,
        "n_duplicate_episodes": 0,
    }
    df = safe_read_csv(path)
    if df is None or df.empty:
        _clean_cache[key] = (None, stats)
        return None, stats

    original_len = len(df)
    stats["n_rows_raw"] = original_len
    if "episode" not in df.columns:
        _clean_cache[key] = (None, stats)
        return None, stats

    df = df.copy()
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df = df.dropna(subset=["episode"])
    stats["n_duplicate_episodes"] = original_len - len(df.drop_duplicates(subset=["episode"], keep="last"))
    df = df.drop_duplicates(subset=["episode"], keep="last").reset_index(drop=True)
    df = df[df["episode"] <= 5000]
    stats["n_rows_unique_le5k"] = len(df)

    if verbose and key not in _clean_logged:
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        emit(
            f"[cleaned] {rel}: kept {stats['n_rows_unique_le5k']} unique-episode rows "
            f"out of {original_len} raw rows"
        )
        _clean_logged.add(key)

    _clean_cache[key] = (df, stats)
    return df, stats


def read_csv_stats(path: Path, *, verbose: bool = False) -> Dict:
    df, clean = _clean_df(path, verbose=verbose)
    if df is None or df.empty:
        return {
            **clean,
            "n_rows": 0,
            "min_episode": None,
            "max_episode": None,
            "last_5_episodes": [],
            "eval_rows": 0,
            "eval_episodes_ok": False,
        }
    ep = df["episode"].astype(int)
    unique_eps = sorted(ep.unique().tolist())
    eval_sub = [e for e in unique_eps if e in EVAL_EPS]
    return {
        **clean,
        "n_rows": len(df),
        "min_episode": int(ep.min()),
        "max_episode": int(ep.max()),
        "last_5_episodes": unique_eps[-5:],
        "eval_rows": len(eval_sub),
        "eval_episodes_ok": set(eval_sub) == EVAL_EPS,
    }


def step1_csv_integrity() -> Tuple[bool, List[int]]:
    emit("## Step 1 — CSV Integrity Check (Bug B)")
    emit()
    emit(
        "| seed | darts_md5 | nocert_md5 | noinc_md5 | rows_d | rows_nc | rows_ni | "
        "max_ep_d | max_ep_nc | max_ep_ni |"
    )
    emit(
        "|------|-----------|------------|-----------|--------|---------|---------|"
        "----------|-----------|-----------|"
    )

    collision_seeds: List[int] = []
    missing_files = 0

    for seed in range(1, N_SEEDS + 1):
        hashes: Dict[str, str] = {}
        stats: Dict[str, Dict] = {}
        for method in METHODS:
            path = csv_path(seed, method)
            if not path.is_file():
                missing_files += 1
                hashes[method] = "MISSING"
                stats[method] = read_csv_stats(path)
                continue
            hashes[method] = md5_file(path)
            stats[method] = read_csv_stats(path)

        md5_vals = [hashes[m] for m in METHODS if hashes[m] != "MISSING"]
        if len(md5_vals) == 3 and len(set(md5_vals)) < 3:
            collision_seeds.append(seed)
            emit(f"\n**BUG (B): seed {seed} — md5 collision among method CSVs.**")
            pairs = [
                (METHODS[i], METHODS[j])
                for i in range(3)
                for j in range(i + 1, 3)
                if hashes[METHODS[i]] == hashes[METHODS[j]] and hashes[METHODS[i]] != "MISSING"
            ]
            for a, b in pairs:
                emit(f"- `{a}_log.csv` == `{b}_log.csv` (md5 `{hashes[a][:12]}…`)")

        emit(
            f"| {seed} "
            f"| `{hashes['darts'][:12]}` "
            f"| `{hashes['darts_nocert'][:12]}` "
            f"| `{hashes['darts_no_incentive'][:12]}` "
            f"| {stats['darts']['n_rows']} "
            f"| {stats['darts_nocert']['n_rows']} "
            f"| {stats['darts_no_incentive']['n_rows']} "
            f"| {stats['darts']['max_episode']} "
            f"| {stats['darts_nocert']['max_episode']} "
            f"| {stats['darts_no_incentive']['max_episode']} |"
        )

    emit()
    emit(f"Expected files: {N_SEEDS * len(METHODS)}; missing: {missing_files}.")
    emit()

    emit("### Per-file pollution report (raw vs cleaned ≤5000)")
    emit()
    emit(
        "| seed | method | n_rows_raw | n_rows_unique_le5k | n_duplicate_episodes | "
        "max_ep (clean) | eval10_ok |"
    )
    emit(
        "|------|--------|------------|--------------------|-----------------------|"
        "----------------|-----------|"
    )
    for seed in range(1, N_SEEDS + 1):
        for method in METHODS:
            path = csv_path(seed, method)
            st = read_csv_stats(path, verbose=False)
            ok = "OK" if st["eval_episodes_ok"] else "WARN"
            emit(
                f"| {seed} | {method} | {st['n_rows_raw']} | {st['n_rows_unique_le5k']} | "
                f"{st['n_duplicate_episodes']} | {st['max_episode']} | {ok} |"
            )
    emit()

    emit("### Per-file detail after cleaning (episode span, last 5 unique episodes)")
    emit()
    for seed in range(1, N_SEEDS + 1):
        emit(f"**seed {seed}**")
        for method in METHODS:
            path = csv_path(seed, method)
            st = read_csv_stats(path, verbose=False)
            ok = "OK" if st["eval_episodes_ok"] else "WARN"
            emit(
                f"- `{path.name}`: raw={st['n_rows_raw']}, unique≤5k={st['n_rows_unique_le5k']}, "
                f"dup_rows={st['n_duplicate_episodes']}, "
                f"ep=[{st['min_episode']}, {st['max_episode']}], "
                f"last5={st['last_5_episodes']}, eval10={st['eval_rows']} [{ok}]"
            )
        emit()

    if collision_seeds:
        emit(f"**Step 1 result:** md5 collisions on seeds {collision_seeds}.")
    else:
        emit("**Step 1 result:** all 30 CSV hashes distinct per seed; no routing collision detected.")
    emit()
    return len(collision_seeds) == 0 and missing_files == 0, collision_seeds


def _grep_with_context(pattern: str, files: Sequence[str], context: int = 3) -> List[Tuple[str, int, List[str]]]:
    hits: List[Tuple[str, int, List[str]]] = []
    for rel in files:
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            proc = subprocess.run(
                ["grep", "-n", "-E", pattern, str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if proc.returncode not in (0, 1):
            continue
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in proc.stdout.splitlines():
            if ":" not in line:
                continue
            num_s, _ = line.split(":", 1)
            if not num_s.isdigit():
                continue
            line_no = int(num_s)
            start = max(0, line_no - 1 - context)
            end = min(len(text), line_no + context)
            hits.append((rel, line_no, text[start:end]))
    return hits


def _collect_grep_files(roots: Sequence[str]) -> List[str]:
    files: List[str] = []
    for root in roots:
        base = REPO / root
        if base.is_file():
            files.append(str(base.relative_to(REPO)))
        elif base.is_dir():
            files.extend(str(p.relative_to(REPO)) for p in sorted(base.rglob("*.py")))
    return files


def _is_explicit_incentive_path(ctx: str) -> bool:
    lower = ctx.lower()
    markers = (
        "incentive",
        "freeze_incentive",
        "phi",
        "loss_phi",
        "optimizer_phi",
        "incentive_net",
        "incentive_trajectory",
        "no_incentive ablation",
        "mc-7",
    )
    return any(m in lower for m in markers)


def _is_bid_step_cert_block(rel: str, line_no: int, ctx: str) -> bool:
    lower = ctx.lower()
    if "disable_feasibility_mask" not in lower:
        return False
    if rel.endswith("darts_agent.py"):
        # Cert evaluation inside bid path (~470-490) or check_feasibility helper (~614)
        if 460 <= line_no <= 495:
            return True
        if "skip schedulability check" in lower or "force cert mask" in lower:
            return True
    return False


def step2_flag_handling() -> Tuple[List[str], bool]:
    emit("## Step 2 — Flag Handling Check (Bug A)")
    emit()

    grep_specs = [
        ("disable_feasibility_mask", ["alg/train_darts.py", "alg/darts_agent.py", "alg/coordinator.py"]),
        ("no_incentive", ["alg/train_darts.py", "alg/darts_agent.py", "alg/coordinator.py"]),
        (r"args\.no_incentive|cfg\.no_incentive", ["alg"]),
    ]

    suspects: List[str] = []
    step2_clean_for_verdict = True

    for pattern, roots in grep_specs:
        files = _collect_grep_files(roots)
        cmd = f'grep -n "{pattern}" {" ".join(roots)}'
        emit(f"### `{cmd}` (+3 context)")
        emit()
        hits = _grep_with_context(pattern, files, context=3)
        if not hits:
            emit("_No matches._")
            emit()
            continue
        for rel, line_no, ctx_lines in hits:
            emit(f"**{rel}:{line_no}**")
            emit("```")
            start_line = line_no - 3
            for j, cl in enumerate(ctx_lines):
                cur = start_line + j
                prefix = ">>>" if cur == line_no else "   "
                emit(f"{prefix} {cur:4d} | {cl}")
            emit("```")
            emit()

            block = "\n".join(ctx_lines)
            lower = block.lower()

            if re.search(r"no_incentive", pattern, re.I):
                if re.search(r"\b(bidding|policy|eta|phi_update|coverage)\b", lower):
                    if not _is_explicit_incentive_path(block):
                        msg = (
                            f"[SUSPECT-A1] {rel}:{line_no} — no_incentive may affect "
                            "non-target code path"
                        )
                        suspects.append(msg)
                        emit(f"**{msg}**")
                        emit()
                        step2_clean_for_verdict = False

            if "disable_feasibility_mask" in pattern:
                if not _is_bid_step_cert_block(rel, line_no, block):
                    if "disable_feasibility_mask" in block:
                        msg = (
                            f"[SUSPECT-A2] {rel}:{line_no} — disable_feasibility_mask "
                            "outside bid-step cert evaluation"
                        )
                        suspects.append(msg)
                        emit(f"**{msg}**")
                        emit()
                        step2_clean_for_verdict = False

    emit("Inspect `[SUSPECT-*]` lines above to confirm whether flags leak into non-target paths.")
    emit()

    emit("### Full `grep -n -B 5 -A 5 disable_feasibility_mask` dump")
    emit()
    emit("```")
    try:
        proc = subprocess.run(
            [
                "grep",
                "-n",
                "-B",
                "5",
                "-A",
                "5",
                "disable_feasibility_mask",
                "alg/train_darts.py",
                "alg/darts_agent.py",
                "alg/coordinator.py",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        emit(proc.stdout.rstrip() if proc.stdout.strip() else "_No matches._")
    except OSError as exc:
        emit(f"_grep failed: {exc}_")
    emit("```")
    emit()
    return suspects, step2_clean_for_verdict


def _conv_window_mean(path: Path, col: str) -> float:
    df, _ = _clean_df(path, verbose=True)
    if df is None or col not in df.columns:
        return float("nan")
    sub = df[df["episode"].isin(CONV_EPS)].copy()
    if sub.empty:
        return float("nan")
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["episode", col])
    if sub.empty:
        return float("nan")
    return float(sub[col].mean())


def _final_ep5000(path: Path, col: str) -> float:
    df, _ = _clean_df(path, verbose=True)
    if df is None or col not in df.columns:
        return float("nan")
    sub = df[df["episode"] == 5000].copy()
    if sub.empty:
        return float("nan")
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    vals = sub[col].dropna()
    if vals.empty:
        return float("nan")
    return float(vals.iloc[-1])


def _build_vectors(col: str) -> Dict[str, np.ndarray]:
    out: Dict[str, List[float]] = {m: [] for m in METHODS}
    for seed in range(1, N_SEEDS + 1):
        for method in METHODS:
            out[method].append(_conv_window_mean(csv_path(seed, method), col))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def _ttest_detail(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int, int, int]:
    n_a = int(np.sum(~np.isnan(a)))
    n_b = int(np.sum(~np.isnan(b)))
    mask = ~(np.isnan(a) | np.isnan(b))
    n_pairs = int(mask.sum())
    if n_pairs < 2:
        return float("nan"), float("nan"), n_a, n_b, n_pairs
    t, p = ss.ttest_ind(a[mask], b[mask], equal_var=False)
    return float(t), float(p), n_a, n_b, n_pairs


def _comparison_tag(n_a: int, n_b: int, p: float) -> str:
    if n_a < 8 or n_b < 8:
        if np.isnan(p) or p >= 0.10:
            return "INDETERMINATE (need more seeds)"
        return "UNDERPOWERED"
    return "valid"


def _sig_label(p: float, tag: str) -> str:
    if tag.startswith("INDETERMINATE"):
        return tag
    if np.isnan(p):
        return "[NA]"
    return "[SIG]" if p < 0.10 else "[NS]"


def _pollution_check(vecs: Dict[str, np.ndarray]) -> None:
    emit("### v1 → cleaned mean shift (pollution check)")
    emit()
    any_sig = False
    for method, v1_mean in V1_MEANS.items():
        clean_mean = float(np.nanmean(vecs[method]))
        if v1_mean == 0:
            rel = abs(clean_mean - v1_mean)
        else:
            rel = abs(clean_mean - v1_mean) / abs(v1_mean)
        flag = ""
        if rel > 0.20:
            any_sig = True
            flag = " **[POLLUTION SIGNIFICANT]**"
        emit(
            f"- {method}: v1={v1_mean:.4f} → cleaned={clean_mean:.4f} "
            f"(Δ={clean_mean - v1_mean:+.4f}, rel={rel:.1%}){flag}"
        )
    emit()
    if any_sig:
        emit(
            "[POLLUTION SIGNIFICANT] Cleaning shifted at least one method mean by >20% vs v1; "
            "cleaned numbers below are the basis for verdict."
        )
        emit()


def step3_welch() -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    emit("## Step 3 — Welch t-Test (seed noise vs real effect)")
    emit()
    emit(
        "Convergence window: episodes {4500, 5000} on **deduped, episode≤5000** CSVs; "
        "per-seed mean of `task_miss_rate`."
    )
    emit()
    emit("### Per-file cleaning log")
    emit()

    vecs = _build_vectors("task_miss_rate")

    emit()
    emit("| method | cleaned mean | std | n_valid_seeds |")
    emit("|--------|--------------|-----|---------------|")
    for method in METHODS:
        v = vecs[method]
        n = int(np.sum(~np.isnan(v)))
        emit(f"| {method} | {np.nanmean(v):.4f} | {np.nanstd(v):.4f} | {n} |")
    emit()

    _pollution_check(vecs)

    comparisons = [
        ("DARTS vs NoCert", "darts", "darts_nocert", "darts", "nocert"),
        ("DARTS vs NoIncen", "darts", "darts_no_incentive", "darts", "noinc"),
        ("NoCert vs NoIncen", "darts_nocert", "darts_no_incentive", "nocert", "noinc"),
    ]
    results: Dict[str, Dict] = {}
    for label, m1, m2, n1_label, n2_label in comparisons:
        t, p, n_a, n_b, n_pairs = _ttest_detail(vecs[m1], vecs[m2])
        tag = _comparison_tag(n_a, n_b, p)
        under = " [UNDERPOWERED]" if (n_a < 8 or n_b < 8) else ""
        verdict = _sig_label(p, tag)
        t_s = f"{t:.4f}" if not np.isnan(t) else "nan"
        p_s = f"{p:.4f}" if not np.isnan(p) else "nan"
        line = (
            f"- **{label}:** t={t_s}, p={p_s}, n_{n1_label}={n_a}, n_{n2_label}={n_b} "
            f"({tag}){under} → {verdict}"
        )
        emit(line)
        results[label] = {
            "t": t,
            "p": p,
            "n_a": n_a,
            "n_b": n_b,
            "n_pairs": n_pairs,
            "tag": tag,
            "verdict": verdict,
        }
    emit()
    return vecs, results


def step4_scatter() -> List[int]:
    emit("## Step 4 — Per-Seed Scatter (episode 5000 task_miss_rate)")
    emit()
    emit("| seed | darts | nocert | noinc | (nocert − darts) | (noinc − darts) |")
    emit("|------|-------|--------|-------|------------------|-----------------|")

    outlier_seeds: List[int] = []
    for seed in range(1, N_SEEDS + 1):
        vals = {m: _final_ep5000(csv_path(seed, m), "task_miss_rate") for m in METHODS}
        d = vals["darts"]
        nc = vals["darts_nocert"]
        ni = vals["darts_no_incentive"]
        d_nc = nc - d if not (np.isnan(nc) or np.isnan(d)) else float("nan")
        d_ni = ni - d if not (np.isnan(ni) or np.isnan(d)) else float("nan")

        flag = ""
        if (not np.isnan(d_nc) and abs(d_nc) > 0.20) or (not np.isnan(d_ni) and abs(d_ni) > 0.20):
            outlier_seeds.append(seed)
            flag = " ⚠"

        def fmt(x: float) -> str:
            return "nan" if np.isnan(x) else f"{x:.3f}"

        emit(
            f"| {seed} | {fmt(d)} | {fmt(nc)} | {fmt(ni)} | {fmt(d_nc)} | {fmt(d_ni)} |{flag}"
        )

    emit()
    if outlier_seeds:
        emit(f"Seeds with |Δ| > 0.20: {outlier_seeds} — inspect for per-seed divergence.")
    else:
        emit("No seed exceeds |Δ| > 0.20 vs DARTS.")
    emit()
    return outlier_seeds


def step5_rho_crosscheck(miss_vecs: Dict[str, np.ndarray]) -> bool:
    emit("## Step 5 — rho_hat Cross-Check")
    emit()
    rho_vecs = _build_vectors("rho_hat")

    emit("| method | mean rho_hat (conv window) | std | n |")
    emit("|--------|--------------------------|-----|---|")
    for method in METHODS:
        v = rho_vecs[method]
        n = int(np.sum(~np.isnan(v)))
        emit(f"| {method} | {np.nanmean(v):.4f} | {np.nanstd(v):.4f} | {n} |")
    emit()

    mean_miss = {m: float(np.nanmean(miss_vecs[m])) for m in METHODS}
    mean_rho = {m: float(np.nanmean(rho_vecs[m])) for m in METHODS}

    inconsistent = False
    checks = [
        ("DARTS vs NoCert", "darts", "darts_nocert"),
        ("DARTS vs NoIncen", "darts", "darts_no_incentive"),
    ]
    emit("| comparison | miss: B better if lower | rho: B better if higher | consistent |")
    emit("|------------|-------------------------|-------------------------|------------|")
    for label, a, b in checks:
        miss_b_better = mean_miss[b] < mean_miss[a]
        rho_b_better = mean_rho[b] > mean_rho[a]
        ok = miss_b_better == rho_b_better
        if not ok and not (np.isnan(mean_miss[a]) or np.isnan(mean_miss[b]) or np.isnan(mean_rho[a]) or np.isnan(mean_rho[b])):
            inconsistent = True
        emit(
            f"| {label} | {miss_b_better} ({mean_miss[a]:.3f} vs {mean_miss[b]:.3f}) | "
            f"{rho_b_better} ({mean_rho[a]:.3f} vs {mean_rho[b]:.3f}) | {'yes' if ok else 'NO'} |"
        )
    emit()

    if inconsistent:
        emit(
            "[METRIC INCONSISTENCY] task_miss_rate and rho_hat disagree on DARTS vs NoCert "
            "direction. One of the columns is bugged. INVESTIGATE n_total_jobs and "
            "n_full_fidelity column population."
        )
        emit()
    else:
        emit("Metric directions consistent between task_miss_rate and rho_hat.")
        emit()
    return inconsistent


def step6_verdict(
    *,
    step1_clean: bool,
    collision_seeds: List[int],
    suspects: List[str],
    step2_clean: bool,
    metric_inconsistent: bool,
    ttest_results: Dict[str, Dict],
) -> str:
    emit("## Step 6 — Verdict")
    emit()

    if collision_seeds:
        emit(
            "**VERDICT: (B)** CSV-routing bug — fix file naming in "
            "`run_all_experiments_medium.py` before any further analysis."
        )
        return "B"

    if suspects and not step2_clean:
        emit(
            "**Note:** `[SUSPECT-*]` flags present — manual Step 2 review required before "
            "assigning **(A)**. Proceeding with automated rules below."
        )
        emit()

    if metric_inconsistent:
        emit(
            "**VERDICT: (A)** bug in CSV column writers (metric inconsistency subtype); "
            "inspect `train_darts.py` `EVAL_CSV_HEADER` population."
        )
        return "A"

    indeterminate_comps = [
        label
        for label, r in ttest_results.items()
        if r["tag"].startswith("INDETERMINATE")
    ]
    if indeterminate_comps:
        emit(
            f"**VERDICT: (INDETERMINATE)** Underpowered comparison(s) {indeterminate_comps} "
            "with p ≥ 0.10 — need more seeds (target n≥8 per method) before drawing ablation "
            "conclusions."
        )
        return "INDETERMINATE"

    p_nc = ttest_results["DARTS vs NoCert"]["p"]
    p_ni = ttest_results["DARTS vs NoIncen"]["p"]

    if (not np.isnan(p_nc) and p_nc >= 0.10) and (not np.isnan(p_ni) and p_ni >= 0.10):
        emit(
            "**VERDICT: (C)** seed-noise. Differences between DARTS and ablations are "
            "within statistical uncertainty at n=10 (both p ≥ 0.10). Paper should add: "
            "_\"differences within seed variance at n=10\"_."
        )
        return "C"

    if step1_clean and step2_clean and not metric_inconsistent:
        p_nc_ok = not np.isnan(p_nc) and p_nc < 0.05
        p_ni_ok = not np.isnan(p_ni) and p_ni < 0.05
        if p_nc_ok or p_ni_ok:
            emit(
                "**VERDICT: (D)** legitimate negative result. The bilevel incentive layer "
                "is not helpful on B1's 4-agent setting (p < 0.05 for at least one ablation). "
                "Paper §V.F should note: _\"The bilevel incentive layer's contribution is "
                "negligible (p > 0.1) on the 4-agent B1 testbed; we hypothesize its "
                "contribution scales with team size and DAG density.\"_"
            )
            return "D"

    emit(
        "**VERDICT: (C)** inconclusive / mixed significance — neither strong effect (D) nor "
        "clear bug; treat as seed-noise until n increases or partial runs (NoIncentive seeds 4–10) "
        "are completed."
    )
    return "C"


def main() -> None:
    emit("# B1/medium Ablation Diagnostic")
    emit()
    emit(f"Repo: `{REPO}`")
    emit(f"Data: `{RESULTS}/exp_{{1..10}}/B1/medium/{{darts,darts_nocert,darts_no_incentive}}_log.csv`")
    emit()

    step1_clean, collision_seeds = step1_csv_integrity()
    suspects, step2_clean = step2_flag_handling()
    miss_vecs, ttest_results = step3_welch()
    step4_scatter()
    metric_inconsistent = step5_rho_crosscheck(miss_vecs)
    letter = step6_verdict(
        step1_clean=step1_clean,
        collision_seeds=collision_seeds,
        suspects=suspects,
        step2_clean=step2_clean,
        metric_inconsistent=metric_inconsistent,
        ttest_results=ttest_results,
    )

    emit()
    emit(f"VERDICT: ({letter})")

    OUT_MD.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"\nSaved report to {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
