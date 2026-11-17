"""TB-3 empirical-anchor runner + collector (Phase 1 tooling).

p1–p6: dry-run plan, flag audit, sweep_tb1.py byte-identity, collector
unit tests, feasibility bits, numpy/torch-free import of the runner.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from scripts.csv_schema_tb import MAIN_CSV_HEADER_TB76
from scripts.sweep_tb1 import (
    DEFAULT_MANIFEST,
    PROFILE_DEADLINE_EXPECTATION,
    assert_deadline_invariant,
    beta_str,
    get_stress_profile,
    is_unity_beta,
    load_manifest,
)
from scripts.sweep_tb3 import (
    TB3_EMP_CKPT_SEEDS,
    TB3_EMP_EVAL_EPISODES,
    TB3_EMP_INSTANCE_SEEDS,
    TB3_EMP_INSTANCE_SEEDS_P7,
    TB3_EMP_N_RUNS,
    TB3_EMP_N_RUNS_P7,
    TB3_EMP_PINNED_EP,
    TB3_EMP_PLAN,
    TB3_EMP_PLAN_P7,
    parse_exp_name,
)
from scripts.tb3_empirical_collect import (
    CollectError,
    collect,
    e2e_clean,
    expected_fields,
    majority,
    perjob_clean,
    walk_runs,
)


def _parse_dry_run(stdout: str):
    names = []
    cmds = []
    for line in stdout.splitlines():
        if line.startswith("# [") and "]" in line:
            rest = line.split("]", 1)[1].strip()
            names.append(rest.split()[0])
        elif line.startswith("conda "):
            cmds.append(line.split())
    return names, cmds


def _require_pinned_checkpoints():
    """Skip the calling test on a machine without the pinned policy checkpoints.

    scripts/sweep_tb3.py builds its plan only if every checkpoint directory named in
    ckpt_manifest_tb1.json exists. Trained checkpoints are not part of the repository,
    so elsewhere the dry run stops before it prints a plan.
    """
    manifest = load_manifest(DEFAULT_MANIFEST)
    for cell in (*TB3_EMP_PLAN, *TB3_EMP_PLAN_P7):
        bench, stress = cell.split(":")
        for seed in TB3_EMP_CKPT_SEEDS:
            ckpt_dir = manifest[f"{bench}|{stress}|{seed}"]["dir"]
            if not os.path.isdir(ckpt_dir):
                pytest.skip(f"pinned policy checkpoint is not on this machine: {ckpt_dir}")


def _require_git_checkout():
    """Skip the calling test when the project root is not the top level of a git work tree.

    The test compares files with what git tracks. A downloaded archive of the
    repository carries no git metadata, so there is nothing to compare with.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-prefix"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        pytest.skip("git is not installed")
    if proc.returncode != 0 or proc.stdout.strip():
        pytest.skip("the project root is not the top level of a git work tree")


def test_p1_plan_shape(tmp_path):
    _require_pinned_checkpoints()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_tb3.py",
            "--dry-run",
            "--results_dir",
            str(tmp_path),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    names, cmds = _parse_dry_run(proc.stdout)
    assert len(names) == TB3_EMP_N_RUNS == 180
    assert len(cmds) == 180
    assert len(set(names)) == 180
    for name in names:
        meta = parse_exp_name(name)
        assert name.endswith(f"_i{meta['instance_seed']}")
        if is_unity_beta(meta["beta"]):
            assert "_b" not in name, name
        else:
            tag = "b" + beta_str(meta["beta"]).replace(".", "p")
            assert f"_{tag}_" in name
        assert "_i" in name

    expected = []
    for cell, betas in TB3_EMP_PLAN.items():
        for beta in betas:
            for inst in TB3_EMP_INSTANCE_SEEDS:
                for ckpt in TB3_EMP_CKPT_SEEDS:
                    expected.append((cell, float(beta), inst, ckpt))
    got = [
        (
            parse_exp_name(n)["cell"],
            parse_exp_name(n)["beta"],
            parse_exp_name(n)["instance_seed"],
            parse_exp_name(n)["ckpt_seed"],
        )
        for n in names
    ]
    assert got == expected


def _flag_val(cmd: list, flag: str):
    return cmd[cmd.index(flag) + 1]


def test_p2_flag_audit_sampled(tmp_path):
    _require_pinned_checkpoints()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_tb3.py",
            "--dry-run",
            "--results_dir",
            str(tmp_path),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    names, cmds = _parse_dry_run(proc.stdout)
    first, last = cmds[0], cmds[-1]
    unity = next(c for n, c in zip(names, cmds) if parse_exp_name(n)["beta"] == 1.0)
    b18 = next(
        c for n, c in zip(names, cmds) if abs(parse_exp_name(n)["beta"] - 1.8) < 1e-12
    )

    def audit(cmd, *, beta, inst, ckpt, stress):
        assert cmd.count("--exp_name") == 1
        assert cmd.count("--instance_seed") == 1
        assert "--eval_only" in cmd
        assert _flag_val(cmd, "--eval_episodes") == str(TB3_EMP_EVAL_EPISODES)
        assert _flag_val(cmd, "--eval_seed_base") == str(ckpt)
        assert _flag_val(cmd, "--random_seed") == str(ckpt)
        assert _flag_val(cmd, "--cert_conservatism") == "1.0"
        assert _flag_val(cmd, "--instance_seed") == str(inst)
        assert _flag_val(cmd, "--load_ckpt_ep") == str(TB3_EMP_PINNED_EP)
        assert _flag_val(cmd, "--workload") == "B1_CDR"
        assert "--subround_log" not in cmd
        assert PROFILE_DEADLINE_EXPECTATION["B1_CDR"] == ("global_deadline_ms",)
        assert "--global_deadline_ms" in cmd
        assert "--relative_deadline_ms" not in cmd
        profile = get_stress_profile("B1_CDR", stress)
        assert _flag_val(cmd, "--global_deadline_ms") == str(profile["global_deadline_ms"])
        if is_unity_beta(beta):
            assert "--deadline_scale" not in cmd
        else:
            assert _flag_val(cmd, "--deadline_scale") == beta_str(beta)
        assert_deadline_invariant("B1_CDR", stress, profile, cmd, deadline_scale=beta)

    m0 = parse_exp_name(names[0])
    audit(
        first,
        beta=m0["beta"],
        inst=m0["instance_seed"],
        ckpt=m0["ckpt_seed"],
        stress=m0["stress"],
    )
    mN = parse_exp_name(names[-1])
    audit(
        last,
        beta=mN["beta"],
        inst=mN["instance_seed"],
        ckpt=mN["ckpt_seed"],
        stress=mN["stress"],
    )
    mu = parse_exp_name(next(n for n in names if parse_exp_name(n)["beta"] == 1.0))
    audit(unity, beta=1.0, inst=mu["instance_seed"], ckpt=mu["ckpt_seed"], stress=mu["stress"])
    m18 = parse_exp_name(
        next(n for n in names if abs(parse_exp_name(n)["beta"] - 1.8) < 1e-12)
    )
    audit(b18, beta=1.8, inst=m18["instance_seed"], ckpt=m18["ckpt_seed"], stress=m18["stress"])
    assert abs(m0["beta"] - 1.0) < 1e-12 and m0["cell"] == "B1_CDR:medium"
    assert abs(mN["beta"] - 2.0) < 1e-12 and mN["cell"] == "B1_CDR:heavy"
    assert mN["instance_seed"] == 9 and mN["ckpt_seed"] == 500


def test_p3_sweep_tb1_byte_identity():
    _require_git_checkout()
    disk = Path(_PROJECT_ROOT, "scripts", "sweep_tb1.py").read_bytes()
    git = subprocess.check_output(
        ["git", "show", "HEAD:scripts/sweep_tb1.py"], cwd=_PROJECT_ROOT
    )
    assert hashlib.sha256(disk).hexdigest() == hashlib.sha256(git).hexdigest()


def _write_eval_row(path: Path, *, task_miss_rate: float, late: float, n1=0.0, n2=0.0, n3=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: 0.0 for k in MAIN_CSV_HEADER_TB76}
    row["episode"] = TB3_EMP_PINNED_EP
    row["task_miss_rate"] = task_miss_rate
    row["n_miss_full_fidelity_late"] = late
    row["n_miss_n1_drop"] = n1
    row["n_miss_n2_fallback_insufficient"] = n2
    row["n_miss_n3_unreached"] = n3
    row["rt_bound_violations"] = 0.0
    row["rho_hat"] = 0.5
    row["max_rt_ms"] = 10.0
    row["episode_span_ms"] = 1000.0
    row["n_ff_missed"] = 0.0
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MAIN_CSV_HEADER_TB76)
        w.writeheader()
        w.writerow(row)


def _write_valid_config(path: Path, cell: str, beta: float, instance_seed: int, ckpt_seed: int):
    _bench, stress = cell.split(":")
    fields = expected_fields(cell, instance_seed)
    wcets = {k: float(v) for k, v in fields.task_wcets_ms.items()}
    locs = {k: [float(p[0]), float(p[1])] for k, p in fields.task_locations_m.items()}
    cfg = {
        "instance_seed": instance_seed,
        "workload_task_wcets_ms": wcets,
        "workload_task_locations_m": locs,
        "task_wcets": dict(wcets),
        "workload_relative_deadline_ms": 150000.0 * beta,
        "global_deadline_ms": {"medium": 150000.0, "heavy": 120000.0}[stress] * beta,
        "cert_conservatism": 1.0,
        "load_ckpt_ep": TB3_EMP_PINNED_EP,
        "eval_seed_base": ckpt_seed,
        "eval_only": True,
        "deadline_scale": float(beta),
        "eval_episodes": TB3_EMP_EVAL_EPISODES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def _exp_name(cell: str, beta: float, inst: int, ckpt: int) -> str:
    bench, stress = cell.split(":")
    name = f"tb1_{bench}_{stress}_k1p0_seed{ckpt}"
    if not is_unity_beta(beta):
        name += f"_b{beta_str(beta).replace('.', 'p')}"
    return f"{name}_i{inst}"


def test_p4_collector_fabricated(tmp_path):
    cell, beta, inst = "B1_CDR:medium", 1.0, 0
    keys = [(cell, beta, inst, s) for s in (100, 300, 500)]
    for ckpt, miss in ((100, 0.0), (300, 0.1)):
        name = _exp_name(cell, beta, inst, ckpt)
        d = tmp_path / name
        _write_eval_row(d / "darts_log.csv", task_miss_rate=miss, late=0.0)
        _write_valid_config(d / "config.json", cell, beta, inst, ckpt)

    result = collect(
        tmp_path,
        expect_n=3,
        expected=keys,
        grid_path=None,
        out_dir=tmp_path / "out",
    )
    assert result["ok"] is False
    joined = "\n".join(result["gate_lines"])
    assert "missing=" in joined and "FAIL" in joined
    assert len(result["runs"]) == 2
    by_ckpt = {r["ckpt_seed"]: r for r in result["runs"]}
    assert by_ckpt[100]["e2e_clean"] == 1
    assert by_ckpt[300]["e2e_clean"] == 0

    dirty = tmp_path / _exp_name(cell, beta, inst, 100) / "config.json"
    cfg = json.loads(dirty.read_text(encoding="utf-8"))
    k = next(iter(cfg["workload_task_wcets_ms"]))
    cfg["workload_task_wcets_ms"][k] = float(cfg["workload_task_wcets_ms"][k]) + 1.0
    dirty.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(CollectError, match="workload_task_wcets_ms"):
        walk_runs(tmp_path)


def test_p5_feasibility_semantics():
    assert e2e_clean(0.0) is True
    assert e2e_clean(0) is True
    assert e2e_clean(0.0001) is False
    assert e2e_clean(1.0) is False
    assert perjob_clean((0.0, 0.0, 0.0, 0.0)) is True
    assert perjob_clean((0, 0, 0, 0)) is True
    assert perjob_clean((0.0, 0.0, 0.0, 1.0)) is False
    assert perjob_clean((0.0, 0.0, 0.0)) is False
    assert majority([True, True, False]) is True
    assert majority([True, False, False]) is False
    assert majority([True, True]) is False


def test_p6_runner_import_is_numpy_torch_free():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.sweep_tb3; "
            "assert 'numpy' not in sys.modules and 'torch' not in sys.modules",
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


GOLD_DEFAULT_DRYRUN = Path(_PROJECT_ROOT, "tests", "gold_tb3_default_dryrun.txt")


def test_q1_plan_p7_dry_run(tmp_path):
    _require_pinned_checkpoints()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_tb3.py",
            "--plan",
            "p7",
            "--dry-run",
            "--results_dir",
            str(tmp_path),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    names, cmds = _parse_dry_run(proc.stdout)
    assert len(names) == TB3_EMP_N_RUNS_P7 == 240
    assert len(cmds) == 240
    assert len(set(names)) == 240
    parsed = [parse_exp_name(n) for n in names]
    assert parsed[-1]["instance_seed"] == 49
    assert "_i49" in names[-1]
    assert all(p["instance_seed"] in TB3_EMP_INSTANCE_SEEDS_P7 for p in parsed)
    assert all("_b" not in n for n in names)
    expected = []
    for cell, betas in TB3_EMP_PLAN_P7.items():
        for beta in betas:
            for inst in TB3_EMP_INSTANCE_SEEDS_P7:
                for ckpt in TB3_EMP_CKPT_SEEDS:
                    expected.append((cell, float(beta), inst, ckpt))
    got = [
        (p["cell"], p["beta"], p["instance_seed"], p["ckpt_seed"]) for p in parsed
    ]
    assert got == expected

    default = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_tb3.py",
            "--plan",
            "default",
            "--dry-run",
            "--results_dir",
            str(tmp_path / "d"),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    d_names, _ = _parse_dry_run(default.stdout)
    assert set(names).isdisjoint(set(d_names))


def test_q2_default_plan_byte_identity():
    _require_pinned_checkpoints()
    gold = GOLD_DEFAULT_DRYRUN.read_bytes()
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_tb3.py",
            "--plan",
            "default",
            "--dry-run",
            "--results_dir",
            "/tmp/tb3_p7_dry_capture",
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    assert proc.stdout == gold


def test_q3_collector_merge_missing_p7_run(tmp_path):
    from scripts.tb3_empirical_collect import collect

    d_default = tmp_path / "default"
    d_p7 = tmp_path / "p7"
    cell = "B1_CDR:medium"
    _write_eval_row(
        d_default / _exp_name(cell, 1.0, 0, 100) / "darts_log.csv",
        task_miss_rate=0.0,
        late=0.0,
    )
    _write_valid_config(
        d_default / _exp_name(cell, 1.0, 0, 100) / "config.json",
        cell,
        1.0,
        0,
        100,
    )
    _write_eval_row(
        d_p7 / _exp_name(cell, 1.0, 10, 100) / "darts_log.csv",
        task_miss_rate=0.0,
        late=0.0,
    )
    _write_valid_config(
        d_p7 / _exp_name(cell, 1.0, 10, 100) / "config.json",
        cell,
        1.0,
        10,
        100,
    )
    expected = [
        (cell, 1.0, 0, 100),
        (cell, 1.0, 10, 100),
        (cell, 1.0, 10, 300),
    ]
    result = collect(
        [d_default, d_p7],
        expect_n=3,
        expected=expected,
        plans=("default", "p7"),
        out_dir=tmp_path / "out",
    )
    assert result["ok"] is False
    joined = "\n".join(result["gate_lines"])
    assert "missing=" in joined
    assert any(r.get("plan") == "default" for r in result["runs"])
    assert any(r.get("plan") == "p7" for r in result["runs"])
    assert (tmp_path / "out" / "tb3_empirical_runs_p7.csv").is_file()
    with (tmp_path / "out" / "tb3_empirical_runs_p7.csv").open() as fh:
        header = fh.readline()
    assert "plan" in header
    assert parse_exp_name("tb1_B1_CDR_medium_k1p0_seed100_i49")["instance_seed"] == 49


def _gate_run(**overrides):
    row = {
        "cell": "B1_CDR:medium",
        "beta": 1.0,
        "instance_seed": 0,
        "ckpt_seed": 100,
        "task_miss_rate": 0.0,
        "n_miss_full_fidelity_late": 0.0,
        "n_miss_n1_drop": 0.0,
        "n_miss_n2_fallback_insufficient": 0.0,
        "n_miss_n3_unreached": 0.0,
        "rt_bound_violations": 0.0,
        "crash_A1": 0.0,
        "crash_A2": 0.0,
        "crash_A3": 0.0,
        "crash_A4": 0.0,
    }
    row.update(overrides)
    return row


def test_r1_positive_dev_fails_g5_cancelled_pair_passes():
    from scripts.tb3_analysis import DETECTION_LIMIT, g5_holds_run
    from scripts.tb3_empirical_collect import gate_block

    assert "zero-channel episode cancelled by a multi-job episode" in DETECTION_LIMIT
    pos = _gate_run(task_miss_rate=0.2)
    assert not g5_holds_run(pos)
    key = ("B1_CDR:medium", 1.0, 0, 100)
    lines, ok = gate_block([pos], expect_n=1, expected=[key], feas=[])
    joined = "\n".join(lines)
    assert ok is False
    g5_line = next(ln for ln in lines if ln.startswith("G5 "))
    assert "FAIL" in g5_line

    cancelled = _gate_run(task_miss_rate=0.4, n_miss_n3_unreached=0.4)
    assert g5_holds_run(cancelled)
    lines, ok = gate_block([cancelled], expect_n=1, expected=[key], feas=[])
    joined = "\n".join(lines)
    assert ok is True
    g5_line = next(ln for ln in lines if ln.startswith("G5 "))
    assert "PASS" in g5_line
    assert "detection-limit" in joined
    assert "zero-channel episode cancelled" in joined


def test_r2_accepted_instance_nonzero_dev_fails_g6():
    from scripts.tb3_empirical_collect import gate_block

    run = _gate_run(task_miss_rate=0.2, n_miss_n3_unreached=0.4)
    feas = [
        {
            "cell": "B1_CDR:medium",
            "beta": 1.0,
            "instance_seed": 0,
            "s0_accept_e2e": 1,
            "s1_accept_e2e": 0,
        }
    ]
    key = ("B1_CDR:medium", 1.0, 0, 100)
    lines, ok = gate_block([run], expect_n=1, expected=[key], feas=feas)
    assert ok is False
    g5_line = next(ln for ln in lines if ln.startswith("G5 "))
    g6_line = next(ln for ln in lines if ln.startswith("G6 "))
    assert "PASS" in g5_line
    assert "FAIL" in g6_line

