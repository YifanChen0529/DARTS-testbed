#!/usr/bin/env python3
"""TB-3 empirical anchor sweep: frozen-policy replay on seeded instances.

The runner itself is bare base python3. Inner eval commands are conda-wrapped
via ``scripts.sweep_tb1.build_command``. Do not wrap this file in ``conda run``.

Plan (signed-off, do not renegotiate): B1_CDR:medium x {1.0, 1.4, 1.6} and
B1_CDR:heavy x {1.0, 1.8, 2.0}; instance_seeds 0..9; ckpt seeds {100, 300, 500}
pinned at ep 2500; eval_episodes=5; kappa=1.0; method darts; subround sidecar
off. 2 x 3 x 10 x 3 = 180 runs.

NTFY_TOPIC_URL must be exported in the launching shell; ``sweep_tb1.notify``
inherits it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import scripts.sweep_tb1 as _tb1  # noqa: E402
from scripts.sweep_tb1 import (  # noqa: E402
    DEFAULT_MANIFEST,
    SweepConfigError,
    append_progress as _tb1_append_progress,
    assert_deadline_invariant,
    beta_str,
    build_command,
    get_stress_profile,
    is_unity_beta,
    load_manifest,
    manifest_entry,
    notify as _tb1_notify,
)

TB3_EMP_PLAN = {
    "B1_CDR:medium": (1.0, 1.4, 1.6),
    "B1_CDR:heavy": (1.0, 1.8, 2.0),
}
TB3_EMP_INSTANCE_SEEDS = tuple(range(10))
TB3_EMP_PLAN_P7 = {
    "B1_CDR:medium": (1.0,),
    "B1_CDR:heavy": (1.0,),
}
TB3_EMP_INSTANCE_SEEDS_P7 = tuple(range(10, 50))
TB3_EMP_CKPT_SEEDS = (100, 300, 500)
TB3_EMP_EVAL_EPISODES = 5
TB3_EMP_KAPPA = 1.0
TB3_EMP_PINNED_EP = 2500
TB3_EMP_N_RUNS = 180
TB3_EMP_N_RUNS_P7 = 240
TB3_EMP_N_BLOCKS = 6
TB3_BLOCK_SIZE = 30  # 10 instances x 3 ckpts
RUN_TIMEOUT_S = 1800
PROGRESS_LOG_NAME = "sweep_tb3_progress.log"
SMOKE_RESULTS_DIR = "results_tb3_emp_smoke"
SMOKE_RESULTS_DIR_P7 = "results_tb3_emp_p7_smoke"
SMOKE_WALL_NAME = "smoke_wall_s.txt"
PANE = "TB3-5"
PANE_P7 = "TB3-7"
DEFAULT_RESULTS_DIR = "results_tb3_emp"
DEFAULT_RESULTS_DIR_P7 = "results_tb3_emp_p7"
EXP_NAME_RE = re.compile(
    r"^tb1_(B1_CDR)_(medium|heavy)_k1p0_seed(\d+)(?:_(b\d+p\d+))?_i(\d+)$"
)

PLAN_SPECS = {
    "default": {
        "plan": TB3_EMP_PLAN,
        "seeds": TB3_EMP_INSTANCE_SEEDS,
        "n_runs": TB3_EMP_N_RUNS,
        "results_dir": DEFAULT_RESULTS_DIR,
        "smoke_dir": SMOKE_RESULTS_DIR,
        "pane": PANE,
        "label": "TB3-5",
    },
    "p7": {
        "plan": TB3_EMP_PLAN_P7,
        "seeds": TB3_EMP_INSTANCE_SEEDS_P7,
        "n_runs": TB3_EMP_N_RUNS_P7,
        "results_dir": DEFAULT_RESULTS_DIR_P7,
        "smoke_dir": SMOKE_RESULTS_DIR_P7,
        "pane": PANE_P7,
        "label": "TB3-7",
    },
}


def append_progress(results_dir: str, line: str) -> None:
    """Reuse sweep_tb1.append_progress with the TB-3 log filename."""
    old = _tb1.PROGRESS_LOG_NAME
    _tb1.PROGRESS_LOG_NAME = PROGRESS_LOG_NAME
    try:
        _tb1_append_progress(results_dir, line)
    finally:
        _tb1.PROGRESS_LOG_NAME = old


def notify(message: str, priority: str = "default", pane: str = PANE) -> None:
    _tb1_notify(pane, message, priority=priority)


def parse_cell(cell: str) -> Tuple[str, str]:
    if ":" not in cell:
        raise SweepConfigError(f"cell {cell!r} is not 'BENCH:stress'")
    bench, stress = cell.split(":", 1)
    return bench.strip(), stress.strip()


def parse_beta_tag(tag: str) -> float:
    """'b1p8' -> 1.8."""
    raw = tag[1:] if tag.startswith("b") else tag
    return float(raw.replace("p", "."))


def parse_exp_name(name: str) -> Dict[str, Any]:
    m = EXP_NAME_RE.match(name)
    if not m:
        raise SweepConfigError(f"cannot parse exp_name {name!r}")
    bench, stress, ckpt_s, btag, inst_s = m.groups()
    beta = 1.0 if btag is None else parse_beta_tag(btag)
    return {
        "cell": f"{bench}:{stress}",
        "bench": bench,
        "stress": stress,
        "beta": beta,
        "ckpt_seed": int(ckpt_s),
        "instance_seed": int(inst_s),
        "exp_name": name,
    }


def _count_flag(cmd: Sequence[str], flag: str) -> int:
    return sum(1 for t in cmd if t == flag)


def post_edit_command(
    cmd: List[str],
    tb1_name: str,
    instance_seed: int,
) -> Tuple[List[str], str]:
    """Replace --exp_name with the TB-3 name and append --instance_seed."""
    out = list(cmd)
    exp_name = f"{tb1_name}_i{instance_seed}"
    if "--exp_name" not in out:
        raise SweepConfigError("build_command returned no --exp_name")
    idx = out.index("--exp_name")
    out[idx + 1] = exp_name
    out.extend(["--instance_seed", str(int(instance_seed))])
    if _count_flag(out, "--exp_name") != 1:
        raise SweepConfigError("final argv must contain exactly one --exp_name")
    if _count_flag(out, "--instance_seed") != 1:
        raise SweepConfigError("final argv must contain exactly one --instance_seed")
    return out, exp_name


def build_tb3_command(
    bench: str,
    stress: str,
    ckpt_seed: int,
    instance_seed: int,
    beta: float,
    entry: Dict[str, Any],
    profile: Dict[str, Any],
    results_dir: str,
    eval_episodes: int,
) -> Tuple[List[str], str]:
    cmd, tb1_name = build_command(
        bench,
        stress,
        ckpt_seed,
        TB3_EMP_KAPPA,
        entry,
        profile,
        results_dir,
        eval_episodes,
        ckpt_seed,  # eval_seed_base = ckpt_seed + 0
        False,  # subround_log off
        deadline_scale=beta,
    )
    cmd, exp_name = post_edit_command(cmd, tb1_name, instance_seed)
    assert_deadline_invariant(bench, stress, profile, cmd, deadline_scale=beta)
    return cmd, exp_name


def _gate_manifest_ep(
    manifest: Dict[str, Any], emp_plan: Dict[str, Tuple[float, ...]]
) -> None:
    """Every planned (cell, ckpt_seed) must exist with pinned ep 2500."""
    for cell in emp_plan:
        bench, stress = parse_cell(cell)
        for seed in TB3_EMP_CKPT_SEEDS:
            entry = manifest_entry(manifest, bench, stress, seed)
            ep = int(entry["ep"])
            if ep != TB3_EMP_PINNED_EP:
                raise SweepConfigError(
                    f"manifest {bench}|{stress}|{seed} has ep={ep}, "
                    f"want pinned {TB3_EMP_PINNED_EP}"
                )


def plan_runs(
    manifest: Dict[str, Any],
    results_dir: str,
    eval_episodes: int,
    *,
    emp_plan: Optional[Dict[str, Tuple[float, ...]]] = None,
    instance_seeds: Optional[Sequence[int]] = None,
    n_runs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Cells in plan order, then beta asc, instance_seed asc, ckpt_seed asc."""
    emp_plan = TB3_EMP_PLAN if emp_plan is None else emp_plan
    instance_seeds = TB3_EMP_INSTANCE_SEEDS if instance_seeds is None else instance_seeds
    n_runs = TB3_EMP_N_RUNS if n_runs is None else n_runs
    _gate_manifest_ep(manifest, emp_plan)
    runs: List[Dict[str, Any]] = []
    for cell, betas in emp_plan.items():
        bench, stress = parse_cell(cell)
        profile = get_stress_profile(bench, stress)
        if profile is None:
            raise SweepConfigError(f"no stress profile for {cell}")
        for beta in betas:
            for instance_seed in instance_seeds:
                for ckpt_seed in TB3_EMP_CKPT_SEEDS:
                    entry = manifest_entry(manifest, bench, stress, ckpt_seed)
                    cmd, exp_name = build_tb3_command(
                        bench,
                        stress,
                        ckpt_seed,
                        instance_seed,
                        beta,
                        entry,
                        profile,
                        results_dir,
                        eval_episodes,
                    )
                    runs.append(
                        {
                            "cell": cell,
                            "bench": bench,
                            "stress": stress,
                            "beta": float(beta),
                            "instance_seed": int(instance_seed),
                            "ckpt_seed": int(ckpt_seed),
                            "ep": int(entry["ep"]),
                            "exp_name": exp_name,
                            "cmd": cmd,
                        }
                    )
    if len(runs) != n_runs:
        raise SweepConfigError(
            f"plan has {len(runs)} runs, signed-off total is {n_runs}"
        )
    names = [r["exp_name"] for r in runs]
    if len(set(names)) != len(names):
        raise SweepConfigError("duplicate exp_name in plan")
    return runs


def assert_fresh_results_dir(runs: Sequence[Dict[str, Any]], results_dir: str) -> None:
    """Refuse if any planned exp dir already exists (no --resume)."""
    clashes = [
        os.path.join(results_dir, r["exp_name"])
        for r in runs
        if os.path.exists(os.path.join(results_dir, r["exp_name"]))
    ]
    if clashes:
        raise SweepConfigError(
            "REFUSAL: planned experiment dirs already exist (no --resume; "
            "move or delete them first):\n  " + "\n  ".join(clashes)
        )


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}:{m:02d}"


def _read_smoke_wall_s(smoke_dir: str = SMOKE_RESULTS_DIR) -> Optional[float]:
    path = os.path.join(_PROJECT_ROOT, smoke_dir, SMOKE_WALL_NAME)
    if not os.path.isfile(path):
        return None
    try:
        return float(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _write_smoke_wall_s(results_dir: str, wall_s: float) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, SMOKE_WALL_NAME), "w") as fh:
        fh.write(f"{wall_s:.3f}\n")


def _run_one(cmd: Sequence[str]) -> Tuple[str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            list(cmd), cwd=_PROJECT_ROOT, timeout=RUN_TIMEOUT_S
        )
        rc: Any = proc.returncode
        tag = str(rc)
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
        tag = "TIMEOUT"
    elapsed = time.time() - t0
    return tag, elapsed


def execute_runs(
    runs: Sequence[Dict[str, Any]],
    results_dir: str,
    *,
    per_run_s_hint: Optional[float],
    pane: str = PANE,
    label: str = "TB3-5",
) -> int:
    os.makedirs(results_dir, exist_ok=True)
    total = len(runs)
    n_blocks = max(1, (total + TB3_BLOCK_SIZE - 1) // TB3_BLOCK_SIZE)
    hint = per_run_s_hint if per_run_s_hint and per_run_s_hint > 0 else 75.0
    eta0 = _fmt_hms(hint * total)
    start_msg = (
        f"{total} runs, {n_blocks} blocks, ETA {eta0} from smoke per-run wall time"
    )
    append_progress(
        results_dir,
        f"# pane={pane} total={total} start={time.strftime('%F %T')} {start_msg}",
    )
    notify(start_msg, pane=pane)

    n_fail = 0
    walls: List[float] = []
    sweep_t0 = time.time()
    block_idx = 0
    prev_key: Optional[Tuple[str, float]] = None

    for i, r in enumerate(runs, 1):
        key = (r["cell"], r["beta"])
        if prev_key is not None and key != prev_key:
            block_idx += 1
            done = i - 1
            elapsed = time.time() - sweep_t0
            mean_s = (sum(walls) / len(walls)) if walls else hint
            eta = _fmt_hms(mean_s * (total - done))
            notify(
                f"block {block_idx}/{n_blocks} done, {done}/{total}, "
                f"elapsed {_fmt_hms(elapsed)}, ETA {eta}",
                pane=pane,
            )
        prev_key = key

        print(f"\n===== [{i}/{total}] {r['exp_name']} =====", flush=True)
        print(" ".join(r["cmd"]), flush=True)
        tag, elapsed = _run_one(r["cmd"])
        walls.append(elapsed)
        ok = tag == "0"
        if not ok:
            n_fail += 1
            notify(
                f"HIGH: {r['exp_name']} rc={tag} ({i}/{total})",
                priority="high",
                pane=pane,
            )
        append_progress(
            results_dir,
            f"{time.strftime('%F %T')} {r['exp_name']} rc={tag} "
            f"wall_s={elapsed:.3f} ({i}/{total})",
        )
        if not ok:
            sys.stderr.write(
                f"[sweep_tb3] {r['exp_name']} rc={tag}; continuing.\n"
            )

    if prev_key is not None:
        block_idx += 1
        elapsed = time.time() - sweep_t0
        notify(
            f"block {block_idx}/{n_blocks} done, {total}/{total}, "
            f"elapsed {_fmt_hms(elapsed)}, ETA 0:00",
            pane=pane,
        )

    n_ok = total - n_fail
    summary = (
        f"{label} sweep done: {n_ok}/{total} ok, {n_fail} failed -> run Part B"
    )
    append_progress(results_dir, f"# {summary}")
    notify(summary, priority=("high" if n_fail else "default"), pane=pane)
    print(f"\n[sweep_tb3] {summary}")
    return 1 if n_fail else 0


def _fatal(exc: BaseException, results_dir: Optional[str]) -> int:
    first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    msg = f"REFUSAL: {first}"
    sys.stderr.write(f"[sweep_tb3] FATAL: {exc}\n")
    if results_dir:
        try:
            append_progress(results_dir, f"# {msg}")
        except OSError:
            pass
    notify(msg, priority="high")
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results_dir",
        "--results-dir",
        dest="results_dir",
        default=None,
        help="Default results_tb3_emp (default plan) or results_tb3_emp_p7 (--plan p7).",
    )
    ap.add_argument(
        "--plan",
        choices=tuple(PLAN_SPECS),
        default="default",
        help="default = 180-run TB3-5 plan (byte-identical). p7 = seeds 10..49 at scale 1.0 (240 runs).",
    )
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the planned commands after the manifest + fresh-dir gates; exit 0.",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="First plan entry only, eval_episodes=2, results_tb3_emp_smoke/.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="First N runs (debugging only).",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec = PLAN_SPECS[args.plan]
    results_dir = args.results_dir or spec["results_dir"]
    try:
        if args.smoke and args.dry_run:
            raise SweepConfigError("--smoke and --dry-run are mutually exclusive")
        if args.smoke:
            results_dir = spec["smoke_dir"]
        manifest = load_manifest(args.manifest)
        eval_episodes = 2 if args.smoke else TB3_EMP_EVAL_EPISODES
        runs = plan_runs(
            manifest,
            results_dir,
            eval_episodes,
            emp_plan=spec["plan"],
            instance_seeds=spec["seeds"],
            n_runs=spec["n_runs"],
        )
        assert_fresh_results_dir(runs if not args.smoke else runs[:1], results_dir)
        if args.smoke:
            runs = runs[:1]
        elif args.limit is not None:
            if args.limit < 0:
                raise SweepConfigError(f"--limit must be >= 0, got {args.limit}")
            runs = runs[: args.limit]
    except SweepConfigError as exc:
        return _fatal(exc, results_dir)
    except Exception as exc:  # noqa: BLE001 — runner-level: ntfy then nonzero
        sys.stderr.write(traceback.format_exc())
        return _fatal(exc, results_dir)

    if args.dry_run:
        print(
            f"# {spec['label']} empirical plan: runs={len(runs)} "
            f"results_dir={results_dir} eval_episodes={TB3_EMP_EVAL_EPISODES}"
        )
        print(
            "# order: cells in TB3_EMP_PLAN, beta asc, instance_seed asc, "
            "ckpt_seed asc"
        )
        for i, r in enumerate(runs, 1):
            print(f"\n# [{i}/{len(runs)}] {r['exp_name']} (ep{r['ep']})")
            print(" ".join(r["cmd"]))
        return 0

    try:
        hint = _read_smoke_wall_s(spec["smoke_dir"])
        rc = execute_runs(
            runs,
            results_dir,
            per_run_s_hint=hint,
            pane=spec["pane"],
            label=spec["label"],
        )
        if args.smoke and runs:
            # last progress line carries wall_s; also persist for the full sweep ETA
            log = os.path.join(results_dir, PROGRESS_LOG_NAME)
            wall = hint
            if os.path.isfile(log):
                for line in open(log):
                    if " wall_s=" in line:
                        try:
                            wall = float(line.split("wall_s=", 1)[1].split()[0])
                        except (IndexError, ValueError):
                            pass
            if wall is not None:
                _write_smoke_wall_s(results_dir, wall)
        return rc
    except SweepConfigError as exc:
        return _fatal(exc, results_dir)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        return _fatal(exc, results_dir)


if __name__ == "__main__":
    sys.exit(main())
