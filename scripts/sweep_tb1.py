#!/usr/bin/env python3
"""Sequential TB-1 sweep runner for one κ pane.

One process drives one pane (typically one κ value) over the manifest cells, running
``alg/train_darts.py --eval_only`` once per (κ, bench, stress, seed) and notifying after
each run. Panes are independent, so several can run concurrently under tmux; see
``scripts/sweep_tb1_launch.sh``.

Stress parameters are imported from ``alg.stress_profiles``, the same table
``run_all_experiments_medium.py`` splats into CLI flags. Numbers are never re-typed here.

Note on ``poisson_*``: those flags are inert at runtime -- the sampling release path is
dead code, and B2's releases are static. They are still passed so the TB run's config.json
carries the same provenance as the paper-era run it is compared against.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alg.stress_profiles import get_stress_profile  # noqa: E402

CONDA_ENV = "DARTS_nano"
LD_LIBRARY_PATH = "/home/yifan/miniconda3/envs/DARTS_nano/lib/cusparselt"
NTFY_TOPIC_URL = os.environ.get("NTFY_TOPIC_URL", "")  # opt-in: read from the environment

DEFAULT_MANIFEST = os.path.join(_PROJECT_ROOT, "ckpt_manifest_tb1.json")
DEFAULT_CELLS = "B1_CDR:medium,B1_CDR:heavy,B2_LML:medium"
PROGRESS_LOG_NAME = "sweep_tb1_progress.log"

# ---------------------------------------------------------------------------
# Deadline-flag invariant.
#
# DARTS_CONFIG defaults global_deadline_ms to 150000, which silently overrides B2's
# workload value of 240000 on a bare invocation. The stress-profile splat is the only
# thing keeping B2 semantics correct, so its presence is enforced rather than assumed.
#
# The expectation is per benchmark because the AE-era profiles are not uniform: B1_CDR
# declares only global_deadline_ms and lets the workload's own relative_deadline_ms
# (150000) stand, while B2_LML declares both. Requiring --relative_deadline_ms
# unconditionally would be unsatisfiable for B1 (no value exists in its profile to source
# from) and would change B1 heavy semantics away from the published baseline that
# --check-kappa1 regresses against. Declaring the expectation per benchmark keeps the
# check strict in both directions: a missing key fails, and so does an unexpected extra
# key, which would silently move a cell off its published deadline.
# ---------------------------------------------------------------------------
DEADLINE_KEYS = ("global_deadline_ms", "relative_deadline_ms")
PROFILE_DEADLINE_EXPECTATION: Dict[str, Tuple[str, ...]] = {
    "B1_CDR": ("global_deadline_ms",),
    "B2_LML": ("global_deadline_ms", "relative_deadline_ms"),
    "B3_IQI": ("global_deadline_ms", "relative_deadline_ms"),
    "B4_DEM": ("global_deadline_ms",),
}


class SweepConfigError(RuntimeError):
    """Raised for any condition that must stop the sweep before it launches."""


def kappa_str(kappa: float) -> str:
    """Canonical decimal text for a κ: 1.0 -> '1.0', 0.75 -> '0.75'.

    repr of a float keeps the trailing zero that %g would drop, so κ=1.0 and κ=1 cannot
    produce two different experiment directories for the same sweep point.
    """
    return str(float(kappa))


def kappa_tag(kappa: float) -> str:
    """0.75 -> 'k0p75', 1.0 -> 'k1p0'."""
    return "k" + kappa_str(kappa).replace(".", "p")


def beta_str(beta: float) -> str:
    return str(float(beta))


def beta_tag(beta: float) -> str:
    """0.85 -> 'b0p85'."""
    return "b" + beta_str(beta).replace(".", "p")


def is_unity_beta(beta: float) -> bool:
    return float(beta) == 1.0


def exp_name_for(
    bench: str, stress: str, kappa: float, seed: int, beta: float = 1.0
) -> str:
    name = f"tb1_{bench}_{stress}_{kappa_tag(kappa)}_seed{seed}"
    if not is_unity_beta(beta):
        name += f"_{beta_tag(beta)}"
    return name


def fmt_value(v: Any) -> str:
    """Same formatting run_all_experiments_medium._fmt_value uses for the splat."""
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def parse_kappas(raw: str) -> List[float]:
    vals = [float(x) for x in raw.split(",") if x.strip()]
    if not vals:
        raise SweepConfigError("--kappas produced an empty list")
    for v in vals:
        if v <= 0:
            raise SweepConfigError(f"κ must be positive, got {v}")
    return vals


def parse_cells(raw: str) -> List[Tuple[str, str]]:
    cells: List[Tuple[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SweepConfigError(f"--cells entry {chunk!r} is not 'BENCH:stress'")
        bench, stress = chunk.split(":", 1)
        cells.append((bench.strip(), stress.strip()))
    if not cells:
        raise SweepConfigError("--cells produced an empty list")
    return cells


def load_manifest(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise SweepConfigError(f"manifest not found: {path}")
    with open(path) as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise SweepConfigError(f"manifest {path} is not a JSON object")
    return manifest


def manifest_seeds(manifest: Dict[str, Any], bench: str, stress: str) -> List[int]:
    """Seeds present in the manifest for this cell, ascending. Hard-fails when empty."""
    prefix = f"{bench}|{stress}|"
    seeds: List[int] = []
    for key in manifest:
        if key == "_meta" or not key.startswith(prefix):
            continue
        tail = key[len(prefix):]
        if not tail.isdigit():
            raise SweepConfigError(f"manifest key {key!r} has a non-integer seed field")
        seeds.append(int(tail))
    if not seeds:
        raise SweepConfigError(
            f"manifest has no entries for cell {bench}:{stress} "
            f"(looked for keys starting {prefix!r})"
        )
    return sorted(seeds)


def manifest_entry(manifest: Dict[str, Any], bench: str, stress: str, seed: int) -> Dict[str, Any]:
    key = f"{bench}|{stress}|{seed}"
    entry = manifest.get(key)
    if entry is None:
        raise SweepConfigError(f"manifest is missing key {key!r}")
    for field in ("dir", "ep"):
        if entry.get(field) in (None, ""):
            raise SweepConfigError(f"manifest key {key!r} has no {field!r}")
    ckpt_dir = entry["dir"]
    if not os.path.isdir(ckpt_dir):
        raise SweepConfigError(f"manifest key {key!r} points at a missing dir: {ckpt_dir}")
    return entry


def assert_deadline_invariant(
    bench: str,
    stress: str,
    profile: Dict[str, Any],
    cmd: Sequence[str],
    deadline_scale: float = 1.0,
) -> None:
    """Deadline flags must come from the stress profile, exactly as the profile declares.

    Fails in both directions: a required key absent from the profile or from the built
    command, and a deadline key present in the profile that the expectation does not
    allow (which would move the cell off its published deadline).

    ``--deadline_scale`` is not a profile key. When ``deadline_scale != 1.0`` it must
    appear on the command with that value; at unity it must be absent, so a β-scaled
    run cannot silently masquerade as the paper-era environment.
    """
    expected = PROFILE_DEADLINE_EXPECTATION.get(bench)
    if expected is None:
        raise SweepConfigError(
            f"no deadline expectation registered for benchmark {bench!r}; add one to "
            "PROFILE_DEADLINE_EXPECTATION before sweeping it"
        )
    declared = tuple(k for k in DEADLINE_KEYS if k in profile)
    if declared != expected:
        raise SweepConfigError(
            f"{bench}:{stress} stress profile declares deadline keys {declared} but "
            f"{expected} was expected. A profile edit changed which deadline the run "
            "inherits; reconcile alg/stress_profiles.py with "
            "PROFILE_DEADLINE_EXPECTATION before sweeping."
        )
    for key in expected:
        flag = f"--{key}"
        if flag not in cmd:
            raise SweepConfigError(
                f"{bench}:{stress} built command is missing {flag}, which the stress "
                "profile declares. The profile splat is what keeps deadline semantics "
                "correct; refusing to launch."
            )
        value = cmd[list(cmd).index(flag) + 1]
        if value != fmt_value(profile[key]):
            raise SweepConfigError(
                f"{bench}:{stress} {flag}={value!r} does not match the profile value "
                f"{fmt_value(profile[key])!r}"
            )
    scale_flag = "--deadline_scale"
    if not is_unity_beta(deadline_scale):
        if scale_flag not in cmd:
            raise SweepConfigError(
                f"{bench}:{stress} --deadline-scale {deadline_scale} is active but "
                f"{scale_flag} is missing from the built command"
            )
        got = cmd[list(cmd).index(scale_flag) + 1]
        if got != beta_str(deadline_scale):
            raise SweepConfigError(
                f"{bench}:{stress} {scale_flag}={got!r} does not match "
                f"{beta_str(deadline_scale)!r}"
            )
    elif scale_flag in cmd:
        raise SweepConfigError(
            f"{bench}:{stress} built command has unexpected {scale_flag} at "
            "deadline_scale=1.0; β-scaled runs must not share the unsuffixed exp_name"
        )


def build_command(
    bench: str,
    stress: str,
    seed: int,
    kappa: float,
    entry: Dict[str, Any],
    profile: Dict[str, Any],
    results_dir: str,
    eval_episodes: int,
    eval_seed_base: int,
    subround_log: bool,
    deadline_scale: float = 1.0,
) -> Tuple[List[str], str]:
    inner: List[str] = [
        "python3", "-u", "alg/train_darts.py",
        "--eval_only",
        "--load_ckpt", str(entry["dir"]),
        "--load_ckpt_ep", str(int(entry["ep"])),
        "--workload", bench,
    ]
    for key, value in profile.items():
        inner.extend([f"--{key}", fmt_value(value)])
    inner.extend([
        "--random_seed", str(seed),
        "--eval_seed_base", str(eval_seed_base),
        "--cert_conservatism", kappa_str(kappa),
        "--eval_episodes", str(eval_episodes),
        "--results_dir", results_dir,
    ])
    # Provenance only: the eval_only path runs no updates, so bilevel_mode cannot change
    # the simulation, but config.json would otherwise record "legacy" (the argparse
    # default) for checkpoints that were trained under fixed_eta.
    bilevel = entry.get("bilevel_mode")
    if bilevel:
        inner.extend(["--bilevel_mode", str(bilevel)])
    if subround_log:
        inner.append("--subround_log")
    if not is_unity_beta(deadline_scale):
        inner.extend(["--deadline_scale", beta_str(deadline_scale)])
    exp_name = exp_name_for(bench, stress, kappa, seed, deadline_scale)
    inner.extend(["--exp_name", exp_name])

    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "env",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        f"PYTHONPATH={_PROJECT_ROOT}",
        *inner,
    ]
    return cmd, exp_name


def plan_runs(
    kappas: Sequence[float],
    cells: Sequence[Tuple[str, str]],
    manifest: Dict[str, Any],
    results_dir: str,
    eval_episodes: int,
    eval_seed_base_offset: int,
    subround_log: bool,
    deadline_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build and validate every command up front, so a bad cell fails before run 1."""
    ordered = sorted(kappas, key=lambda k: (k != 1.0, k))
    runs: List[Dict[str, Any]] = []
    for kappa in ordered:
        for bench, stress in cells:
            profile = get_stress_profile(bench, stress)
            if profile is None:
                raise SweepConfigError(
                    f"no stress profile for {bench}:{stress} (single-medium-only benchmark?)"
                )
            for seed in manifest_seeds(manifest, bench, stress):
                entry = manifest_entry(manifest, bench, stress, seed)
                cmd, exp_name = build_command(
                    bench, stress, seed, kappa, entry, profile, results_dir,
                    eval_episodes, seed + eval_seed_base_offset, subround_log,
                    deadline_scale,
                )
                assert_deadline_invariant(
                    bench, stress, profile, cmd, deadline_scale=deadline_scale
                )
                runs.append({
                    "kappa": kappa,
                    "bench": bench,
                    "stress": stress,
                    "seed": seed,
                    "ep": int(entry["ep"]),
                    "exp_name": exp_name,
                    "cmd": cmd,
                })
    return runs


def assert_no_stale_outputs(runs: Sequence[Dict[str, Any]], results_dir: str) -> None:
    """Refuse to start if any target dir already holds a darts_log.csv.

    There is no --resume: append-mode writers would mix the new rows into an existing
    file, which is exactly the stale-generation hazard the reference reader in
    tb1_analysis.py has to work around on the paper-era logs.
    """
    clashes = [
        os.path.join(results_dir, r["exp_name"], "darts_log.csv")
        for r in runs
        if os.path.isfile(os.path.join(results_dir, r["exp_name"], "darts_log.csv"))
    ]
    if clashes:
        raise SweepConfigError(
            "target experiment dirs already contain darts_log.csv (no --resume; move or "
            "delete them first):\n  " + "\n  ".join(clashes)
        )


def notify(pane: str, message: str, priority: str = "default") -> None:
    env = dict(os.environ)
    env.setdefault("NTFY_TOPIC_URL", NTFY_TOPIC_URL)
    try:
        subprocess.run(
            [sys.executable, os.path.join(_SCRIPT_DIR, "notify.py"),
             "--title", f"TB1 {pane}", "--message", message, "--priority", priority],
            env=env, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(f"[sweep] notify failed (continuing): {exc}\n")


def append_progress(results_dir: str, line: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, PROGRESS_LOG_NAME), "a") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kappas", required=True,
                    help="Comma-separated κ values for this pane, e.g. '1.0' or '0.5,0.75'.")
    ap.add_argument("--cells", default=DEFAULT_CELLS,
                    help=f"Comma-separated BENCH:stress cells (default: {DEFAULT_CELLS}).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="TB-1 checkpoint manifest (v2: 'bench|stress|seed' -> {dir, ep}).")
    ap.add_argument("--results-dir", required=True,
                    help="Output root passed to train_darts --results_dir. Use a FRESH dir.")
    ap.add_argument("--eval-episodes", type=int, default=100,
                    help="Eval episodes per run (default 100; ~0.05 s/episode on CPU).")
    ap.add_argument("--eval-seed-base-offset", type=int, default=0,
                    help="Added to the cell seed to form --eval_seed_base (default 0).")
    ap.add_argument("--subround-log", action="store_true", default=False,
                    help="Pass --subround_log to every run (per-subround sidecar).")
    ap.add_argument("--deadline-scale", type=float, default=1.0,
                    help="β: when != 1.0, pass --deadline_scale to train_darts and suffix "
                         "exp_name with _b<beta> (dot as 'p'). Default 1.0 (no flag, no suffix).")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="Print every command and exit without running anything.")
    ap.add_argument("--pane", default=None,
                    help="Pane label for notifications (default: derived from --kappas).")
    args = ap.parse_args()

    try:
        kappas = parse_kappas(args.kappas)
        cells = parse_cells(args.cells)
        if args.deadline_scale <= 0:
            raise SweepConfigError(
                f"--deadline-scale must be positive, got {args.deadline_scale}"
            )
        manifest = load_manifest(args.manifest)
        runs = plan_runs(
            kappas, cells, manifest, args.results_dir, args.eval_episodes,
            args.eval_seed_base_offset, args.subround_log,
            deadline_scale=args.deadline_scale,
        )
    except SweepConfigError as exc:
        sys.stderr.write(f"[sweep] FATAL: {exc}\n")
        return 2

    pane = args.pane or ",".join(kappa_tag(k) for k in sorted(kappas, key=lambda k: (k != 1.0, k)))
    total = len(runs)

    if args.dry_run:
        print(f"# TB-1 sweep plan: pane={pane} runs={total} results_dir={args.results_dir}")
        print(f"# κ order (1.0 first, then ascending): "
              f"{[k for k in sorted(kappas, key=lambda k: (k != 1.0, k))]}")
        for i, r in enumerate(runs, 1):
            print(f"\n# [{i}/{total}] {r['exp_name']} (ep{r['ep']})")
            print(" ".join(r["cmd"]))
        return 0

    try:
        assert_no_stale_outputs(runs, args.results_dir)
    except SweepConfigError as exc:
        sys.stderr.write(f"[sweep] FATAL: {exc}\n")
        return 2

    os.makedirs(args.results_dir, exist_ok=True)
    append_progress(args.results_dir, f"# pane={pane} total={total} start={time.strftime('%F %T')}")
    notify(pane, f"pane start: {total} runs -> {args.results_dir}")

    n_fail = 0
    sweep_t0 = time.time()
    for i, r in enumerate(runs, 1):
        t0 = time.time()
        print(f"\n===== [{i}/{total}] {r['exp_name']} =====", flush=True)
        print(" ".join(r["cmd"]), flush=True)
        proc = subprocess.run(r["cmd"], cwd=_PROJECT_ROOT)
        rc = proc.returncode
        elapsed = time.time() - t0
        mmss = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        if rc != 0:
            n_fail += 1
        append_progress(
            args.results_dir,
            f"{time.strftime('%F %T')} {r['exp_name']} rc={rc} "
            f"({i}/{total}, {mmss}) kappa={r['kappa']} seed={r['seed']} ep={r['ep']}",
        )
        notify(
            pane,
            f"{r['exp_name']} rc={rc} ({i}/{total}, {mmss})",
            priority=("high" if rc != 0 else "default"),
        )
        if rc != 0:
            sys.stderr.write(f"[sweep] {r['exp_name']} exited rc={rc}; continuing.\n")

    total_s = time.time() - sweep_t0
    summary = (f"pane done: {total - n_fail}/{total} ok, {n_fail} failed, "
               f"{int(total_s // 60):02d}:{int(total_s % 60):02d}")
    append_progress(args.results_dir, f"# {summary}")
    notify(pane, summary, priority=("high" if n_fail else "default"))
    print(f"\n[sweep] {summary}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
