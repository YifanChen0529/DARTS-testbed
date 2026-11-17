#!/usr/bin/env python3
"""TB-3 offline schedulability harness (paper eq. 26), evaluated per mission
instance against ``beta * D``.

The tests here are the paper's pre-deployment schedulability test (eq. 26),
evaluated per mission instance against ``beta * D``. S0's tau_bar is the raw
workspace diameter / u with NO 1.1 safety margin and NO 1.5 obstacle factor
(the deployed certificate applies both, ``envs/turtlebot3_spatial_env.py:276-294``),
exactly as ``scripts/r_off_g.py``. The ``xi_source`` switch exists because B3's
runtime WCET vector differs from its declared one (T0 H3). ``start_pos=(0,0)``
is the conservative parking corner used by the published T3 table.

Torch-free at import time: do not import ``envs.turtlebot3_spatial_env``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alg.tb3_instance import (  # noqa: E402
    NOMINAL_CELLS,
    MissionInstance,
    apply_instance_seed,
    iter_profiled_cells,
    nominal_instance,
    parse_cell,
)
from scripts.csv_schema_tb import (  # noqa: E402
    TB3_CURVES_HEADER,
    TB3_FRONTIER_HEADER,
    TB3_OFFLINE_HEADER,
)
from scripts.r_off_g import (  # noqa: E402
    compute_r_off_g,
    dag_width,
    enumerate_paths,
    euclid,
    workspace_diameter_m,
)

DEFAULT_BETA_GRID = (
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0,
)
# Spec text said "13 points"; the literal P0 grid above has 12. Recorded in
# docs/tb3/TB3-1_report.md.

# TB-3 publishes under the declared-task-set convention (matching the published
# T3 table). ``runtime`` remains available for the informational H3 line.
XI_SOURCE_DEFAULT = "declared"
TESTS_DEFAULT = "S0,S1"

# Canonical TB-3 offline grid. Supersedes the design-doc "13 points" (whose
# literal enumeration had 12); 1.8 resolves the B1 medium/heavy gap; 2.2/2.4
# cover B1/heavy S0 saturation, whose declared-xi upper support is
# (3*2110 + 1.2*70000 + 3*(sqrt(200)/0.22*1000)) / 120000 = 2.3598.
BETA_GRID_TB3 = (
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4,
)
GRID_CELLS_TB3 = (
    "B1_CDR:medium",
    "B1_CDR:heavy",
    "B2_LML:medium",
    "B3_IQI:medium",
    "B4_DEM:medium",
)
GRID_SEEDS_TB3 = range(0, 200)

PUBLISHED_R_INT: Dict[Tuple[str, str], int] = {
    ("B1_CDR", "light"): 240250,
    ("B1_CDR", "medium"): 240250,
    ("B1_CDR", "heavy"): 240250,
    ("B2_LML", "medium"): 255513,
    ("B3_IQI", "medium"): 255513,
    ("B4_DEM", "medium"): 74810,
}
PUBLISHED_DARTS_MEAN: Dict[Tuple[str, str], float] = {
    ("B1_CDR", "light"): 109677.8,
    ("B1_CDR", "medium"): 103381.5,
    ("B1_CDR", "heavy"): 100540.4,
    ("B2_LML", "medium"): 94403.1,
    ("B3_IQI", "medium"): 121692.9,
    ("B4_DEM", "medium"): 57059.4,
}
PUBLISHED_RATIO: Dict[Tuple[str, str], float] = {
    ("B1_CDR", "light"): 2.19,
    ("B1_CDR", "medium"): 2.32,
    ("B1_CDR", "heavy"): 2.39,
    ("B2_LML", "medium"): 2.71,
    ("B3_IQI", "medium"): 2.10,
    ("B4_DEM", "medium"): 1.31,
}
MEAS_TOL_MS = 0.5
H3_B3_RUNTIME_R = 290513.275291
H3_B3_RUNTIME_RATIO = 2.39


# ---------------------------------------------------------------------------
# Envelopes / bound result / registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Envelopes:
    """Makefile constants (``Makefile:24-35``). ``map_size_m`` is for S1 (TB3-3)."""

    delta_auc_ms: float = 30.0
    H_ms: float = 1000.0
    j_bar_ms: float = 0.0
    u_mps: float = 0.22
    h_comm_ms: float = 50.0
    start_pos: Tuple[float, float] = (0.0, 0.0)
    map_size_m: float = 10.0


PUBLISHED_ENVELOPES = Envelopes()


@dataclass
class BoundResult:
    test: str
    sound: bool
    R_G_off_ms: float
    per_node_ms: Dict[str, float]
    A_bar: int
    T_bar_auc_ms: float
    tau_bar_ms: float
    longest_path: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


BoundFn = Callable[[MissionInstance, Envelopes, str], BoundResult]


@dataclass(frozen=True)
class BoundTest:
    name: str
    sound: bool
    fn: BoundFn
    note: str


def s0_path_sum(
    inst: MissionInstance, env: Envelopes, xi_source: str
) -> BoundResult:
    """S0: paper path-sum, bit-exact port of ``scripts/r_off_g.py:90-106``."""
    wl = inst.workload
    xis = inst.task_wcets(xi_source)
    diam_m = workspace_diameter_m(wl.task_locations_m, env.start_pos)
    tau_bar_ms = (diam_m / env.u_mps) * 1000.0
    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    T_bar_auc_ms = A_bar * env.delta_auc_ms
    r_off_v: Dict[str, float] = {}
    for v in wl.node_ids:
        xi_v = xis[v]
        r_off_v[v] = 2.0 * env.H_ms + T_bar_auc_ms + (
            xi_v + env.h_comm_ms + tau_bar_ms
        )
    paths = enumerate_paths(list(wl.node_ids), dict(wl.predecessors))
    best_path, best_sum = None, -1.0
    for path in paths:
        s = sum(env.j_bar_ms + r_off_v[v] for v in path)
        if s > best_sum:
            best_sum, best_path = s, path
    return BoundResult(
        test="S0",
        sound=True,
        R_G_off_ms=best_sum,
        per_node_ms=r_off_v,
        A_bar=A_bar,
        T_bar_auc_ms=T_bar_auc_ms,
        tau_bar_ms=tau_bar_ms,
        longest_path=list(best_path),
        meta={"diameter_m": diam_m, "xi_source": xi_source},
    )


def s1_eccentricity(
    inst: MissionInstance, env: Envelopes, xi_source: str
) -> BoundResult:
    """S1: per-node corner eccentricity travel; see ``docs/tb3/S1_lemma.tex``."""
    wl = inst.workload
    xis = inst.task_wcets(xi_source)
    L = env.map_size_m
    corners = ((0.0, 0.0), (L, 0.0), (0.0, L), (L, L))
    tau_hat_ms: Dict[str, float] = {}
    for v in wl.node_ids:
        loc_v = wl.task_locations_m[v]
        diam_v = max(euclid(c, loc_v) for c in corners)
        tau_hat_ms[v] = (diam_v / env.u_mps) * 1000.0
    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    T_bar_auc_ms = A_bar * env.delta_auc_ms
    r_off_v: Dict[str, float] = {}
    for v in wl.node_ids:
        xi_v = xis[v]
        r_off_v[v] = 2.0 * env.H_ms + T_bar_auc_ms + (
            xi_v + env.h_comm_ms + tau_hat_ms[v]
        )
    paths = enumerate_paths(list(wl.node_ids), dict(wl.predecessors))
    best_path, best_sum = None, -1.0
    for path in paths:
        s = sum(env.j_bar_ms + r_off_v[v] for v in path)
        if s > best_sum:
            best_sum, best_path = s, path
    return BoundResult(
        test="S1",
        sound=True,
        R_G_off_ms=best_sum,
        per_node_ms=r_off_v,
        A_bar=A_bar,
        T_bar_auc_ms=T_bar_auc_ms,
        tau_bar_ms=max(tau_hat_ms.values()),
        longest_path=list(best_path),
        meta={"tau_hat_ms": dict(tau_hat_ms), "xi_source": xi_source},
    )


def _s0_w_informational(
    inst: MissionInstance, env: Envelopes, xi_source: str
) -> Tuple[float, float]:
    """Start-set-corrected S0: tau = diam(W)/u. Not a registered test."""
    L = env.map_size_m
    diam_w_m = euclid((0.0, 0.0), (L, L))
    tau_w_ms = (diam_w_m / env.u_mps) * 1000.0
    wl = inst.workload
    xis = inst.task_wcets(xi_source)
    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    T_bar_auc_ms = A_bar * env.delta_auc_ms
    r_off_v: Dict[str, float] = {}
    for v in wl.node_ids:
        xi_v = xis[v]
        r_off_v[v] = 2.0 * env.H_ms + T_bar_auc_ms + (
            xi_v + env.h_comm_ms + tau_w_ms
        )
    paths = enumerate_paths(list(wl.node_ids), dict(wl.predecessors))
    best_sum = -1.0
    for path in paths:
        s = sum(env.j_bar_ms + r_off_v[v] for v in path)
        if s > best_sum:
            best_sum = s
    return tau_w_ms, best_sum


def _s4_stub(inst: MissionInstance, env: Envelopes, xi_source: str) -> BoundResult:
    raise NotImplementedError("S4 oracle ceiling lands in TB3-6/TB-4")


TESTS: Dict[str, BoundTest] = {
    "S0": BoundTest(
        name="S0",
        sound=True,
        fn=s0_path_sum,
        note="path-sum over workspace-diameter tau_bar; bit-exact r_off_g.py:90-106",
    ),
    "S1": BoundTest(
        name="S1",
        sound=True,
        fn=s1_eccentricity,
        note="per-node corner eccentricity; soundness: docs/tb3/S1_lemma.tex",
    ),
    "S4": BoundTest(
        name="S4",
        sound=False,
        fn=_s4_stub,
        note="S4 oracle ceiling lands in TB3-6/TB-4",
    ),
}


@dataclass
class Row:
    inst: MissionInstance
    result: BoundResult
    xi_source: str
    accept_e2e: bool
    accept_rel: bool
    load_e2e: float
    load_rel: float


def evaluate(
    inst: MissionInstance,
    test_names: Sequence[str],
    env: Envelopes,
    xi_source: str,
) -> List[Row]:
    """Run named tests. Deadlines on ``inst`` are already beta-scaled; do not
    multiply by ``inst.beta`` again (``MissionInstance.scaled`` folded it in)."""
    rows: List[Row] = []
    for name in test_names:
        if name not in TESTS:
            raise KeyError(f"unknown test {name!r}; known: {sorted(TESTS)}")
        spec = TESTS[name]
        result = spec.fn(inst, env, xi_source)
        d_e2e = inst.global_deadline_ms
        d_rel = inst.relative_deadline_ms
        r = result.R_G_off_ms
        rows.append(
            Row(
                inst=inst,
                result=result,
                xi_source=xi_source,
                accept_e2e=(r <= d_e2e),
                accept_rel=(r <= d_rel),
                load_e2e=(r / d_e2e) if d_e2e else float("inf"),
                load_rel=(r / d_rel) if d_rel else float("inf"),
            )
        )
    return rows


def _xi_sources(flag: str) -> List[str]:
    if flag == "both":
        return ["declared", "runtime"]
    if flag in ("declared", "runtime"):
        return [flag]
    raise ValueError(f"--xi-source must be declared|runtime|both, got {flag!r}")


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_tests(raw: str) -> List[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        raise ValueError("--tests produced an empty list")
    return names


def _parse_instance_seeds(
    raw: Optional[str], single: Optional[int]
) -> Optional[List[int]]:
    """Parse ``--instance-seed N`` and/or ``--instance-seeds 0..9,12``.

    Returns None when neither flag is set (nominal instance).
    """
    seeds: List[int] = []
    if single is not None:
        seeds.append(int(single))
    if raw:
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ".." in tok:
                left, right = tok.split("..", 1)
                lo, hi = int(left.strip()), int(right.strip())
                if hi < lo:
                    raise ValueError(f"empty instance-seed range {tok!r}")
                seeds.extend(range(lo, hi + 1))
            else:
                seeds.append(int(tok))
    if not seeds:
        return None
    seen = set()
    out: List[int] = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def print_envelopes(env: Envelopes) -> None:
    print(
        "# Envelopes in force: "
        f"δ_auc={env.delta_auc_ms} ms, H={env.H_ms} ms, "
        f"J̄={env.j_bar_ms} ms, u={env.u_mps} m/s, h={env.h_comm_ms} ms, "
        f"start_pos={env.start_pos}, map_size_m={env.map_size_m}"
    )


def row_to_csv_dict(row: Row, env: Envelopes) -> Dict[str, Any]:
    inst = row.inst
    res = row.result
    seed = "nominal" if inst.instance_seed is None else str(inst.instance_seed)
    return {
        "benchmark": inst.benchmark,
        "stress": inst.stress,
        "instance_seed": seed,
        "beta": inst.beta,
        "test": res.test,
        "sound": res.sound,
        "xi_source": row.xi_source,
        "D_e2e_ms": inst.global_deadline_ms,
        "D_rel_ms": inst.relative_deadline_ms,
        "R_G_off_ms": res.R_G_off_ms,
        "accept_e2e": int(row.accept_e2e),
        "accept_rel": int(row.accept_rel),
        "load_e2e": row.load_e2e,
        "load_rel": row.load_rel,
        "A_bar": res.A_bar,
        "T_bar_auc_ms": res.T_bar_auc_ms,
        "tau_bar_ms": res.tau_bar_ms,
        "longest_path": "->".join(res.longest_path),
        "delta_auc_ms": env.delta_auc_ms,
        "H_ms": env.H_ms,
        "j_bar_ms": env.j_bar_ms,
        "u_mps": env.u_mps,
        "h_comm_ms": env.h_comm_ms,
        "start_x": env.start_pos[0],
        "start_y": env.start_pos[1],
    }


def write_csv(path: Path, rows: Sequence[Row], env: Envelopes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TB3_OFFLINE_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_csv_dict(row, env))


# Grid CSV float fields: fixed ``.16g`` so a rerun is byte-identical.
_GRID_FLOAT_FIELDS = frozenset(
    {
        "beta",
        "D_e2e_ms",
        "D_rel_ms",
        "R_G_off_ms",
        "load_e2e",
        "load_rel",
        "T_bar_auc_ms",
        "tau_bar_ms",
        "delta_auc_ms",
        "H_ms",
        "j_bar_ms",
        "u_mps",
        "h_comm_ms",
        "start_x",
        "start_y",
    }
)
_CURVES_FLOAT_FIELDS = frozenset({"beta", "accept_frac_e2e", "accept_frac_rel"})
_FRONTIER_FLOAT_FIELDS = frozenset(
    {
        "beta_star_min",
        "beta_star_p25",
        "beta_star_median",
        "beta_star_p75",
        "beta_star_max",
    }
)


def _fmt_float_g(value: Any) -> str:
    return format(float(value), ".16g")


def _fmt_grid_cell(key: str, value: Any) -> str:
    if key in _GRID_FLOAT_FIELDS:
        return _fmt_float_g(value)
    if key == "sound":
        return "True" if value else "False"
    return str(value)


def write_grid_csv(path: Path, dicts: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=TB3_OFFLINE_HEADER, lineterminator="\n"
        )
        writer.writeheader()
        for d in dicts:
            writer.writerow({k: _fmt_grid_cell(k, d[k]) for k in TB3_OFFLINE_HEADER})


def _write_header_csv(
    path: Path, header: Sequence[str], rows: Sequence[Dict[str, Any]], float_fields
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for d in rows:
            out = {}
            for k in header:
                v = d[k]
                out[k] = _fmt_float_g(v) if k in float_fields else str(v)
            writer.writerow(out)


def _percentile_linear(xs: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (numpy default). ``p`` in [0, 100]."""
    if not xs:
        raise ValueError("percentile of empty sample")
    s = sorted(float(x) for x in xs)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    w = k - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def aggregate_grid(
    dicts: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build curves (cell, test, beta) and frontier (cell, test) rows.

    ``n_S1_gt_S0`` is a per-cell flip count, copied onto both test rows.
    """
    from collections import defaultdict

    n_by: Dict[Tuple[str, str, float], int] = defaultdict(int)
    n_e2e: Dict[Tuple[str, str, float], int] = defaultdict(int)
    n_rel: Dict[Tuple[str, str, float], int] = defaultdict(int)
    load_e2e_at_1: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    r_by: Dict[Tuple[str, int, str], float] = {}
    cell_order: List[str] = []
    test_order: List[str] = []
    beta_order: List[float] = []
    seen_cell = set()
    seen_test = set()
    seen_beta = set()

    for d in dicts:
        cell = f"{d['benchmark']}:{d['stress']}"
        test = str(d["test"])
        beta = float(d["beta"])
        key = (cell, test, beta)
        n_by[key] += 1
        n_e2e[key] += int(d["accept_e2e"])
        n_rel[key] += int(d["accept_rel"])
        if cell not in seen_cell:
            seen_cell.add(cell)
            cell_order.append(cell)
        if test not in seen_test:
            seen_test.add(test)
            test_order.append(test)
        if beta not in seen_beta:
            seen_beta.add(beta)
            beta_order.append(beta)
        seed = int(d["instance_seed"])
        r_by.setdefault((cell, seed, test), float(d["R_G_off_ms"]))
        if abs(beta - 1.0) <= 1e-12:
            load_e2e_at_1[(cell, test)].append(float(d["load_e2e"]))

    curves: List[Dict[str, Any]] = []
    for cell in cell_order:
        for test in test_order:
            for beta in beta_order:
                key = (cell, test, beta)
                if key not in n_by:
                    continue
                n = n_by[key]
                curves.append(
                    {
                        "cell": cell,
                        "test": test,
                        "beta": beta,
                        "n_instances": n,
                        "n_accept_e2e": n_e2e[key],
                        "accept_frac_e2e": n_e2e[key] / n,
                        "n_accept_rel": n_rel[key],
                        "accept_frac_rel": n_rel[key] / n,
                    }
                )

    flips: Dict[str, int] = {}
    for cell in cell_order:
        seeds = sorted({s for (c, s, _t) in r_by if c == cell})
        n_flip = 0
        for seed in seeds:
            r0 = r_by.get((cell, seed, "S0"))
            r1 = r_by.get((cell, seed, "S1"))
            if r0 is not None and r1 is not None and r1 > r0:
                n_flip += 1
        flips[cell] = n_flip

    frontier: List[Dict[str, Any]] = []
    for cell in cell_order:
        for test in test_order:
            xs = load_e2e_at_1.get((cell, test), [])
            if not xs:
                continue
            frontier.append(
                {
                    "cell": cell,
                    "test": test,
                    "n_instances": len(xs),
                    "beta_star_min": _percentile_linear(xs, 0),
                    "beta_star_p25": _percentile_linear(xs, 25),
                    "beta_star_median": _percentile_linear(xs, 50),
                    "beta_star_p75": _percentile_linear(xs, 75),
                    "beta_star_max": _percentile_linear(xs, 100),
                    "n_S1_gt_S0": flips[cell],
                }
            )
    return curves, frontier


def _load_vg_t3():
    """Lazy load; ``vg_t3_combined.py`` execs ``r_off_g.py`` at import and
    binds ``NEW_SWEEP``/``OLD_RESULTS`` as module globals that ``measured_r_g``
    closes over by name — patch those after load, do not copy its body."""
    path = Path(_PROJECT_ROOT) / "scripts" / "vg_t3_combined.py"
    spec = importlib.util.spec_from_file_location("vg_t3_combined_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_anchor(args: argparse.Namespace) -> int:
    env = PUBLISHED_ENVELOPES
    print_envelopes(env)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_fail = 0

    print()
    print("## (a) tripwire vs scripts/r_off_g.compute_r_off_g (xi_source=declared)")
    for bench, stress in iter_profiled_cells():
        inst = nominal_instance(bench, stress)
        got = s0_path_sum(inst, env, "declared")
        ref = compute_r_off_g(
            bench, stress,
            env.delta_auc_ms, env.H_ms, env.j_bar_ms,
            env.u_mps, env.h_comm_ms, env.start_pos,
        )
        ok_r = got.R_G_off_ms == ref["R_off_G_ms"]
        ok_path = got.longest_path == list(ref["longest_path"])
        ok_a = got.A_bar == ref["A_bar"]
        ok_tau = got.tau_bar_ms == ref["tau_bar_ms"]
        ok = ok_r and ok_path and ok_a and ok_tau
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(
            f"  [{status}] {bench}:{stress}  "
            f"R={got.R_G_off_ms} vs {ref['R_off_G_ms']}  "
            f"path={got.longest_path} vs {ref['longest_path']}  "
            f"A_bar={got.A_bar} vs {ref['A_bar']}  "
            f"tau={got.tau_bar_ms} vs {ref['tau_bar_ms']}"
        )

    print()
    print("## (b) published T3 table (declared xi, method=darts)")
    vg = _load_vg_t3()
    vg.NEW_SWEEP = Path(args.new_sweep_dir).expanduser().resolve()
    vg.OLD_RESULTS = Path(args.old_results_dir).expanduser().resolve()
    print(f"  NEW_SWEEP={vg.NEW_SWEEP}")
    print(f"  OLD_RESULTS={vg.OLD_RESULTS}")
    for cell in NOMINAL_CELLS:
        bench, stress = parse_cell(cell)
        inst = nominal_instance(bench, stress)
        got = s0_path_sum(inst, env, "declared")
        want_r = PUBLISHED_R_INT[(bench, stress)]
        ok_int = round(got.R_G_off_ms) == want_r
        mean_r, std_r, n = vg.measured_r_g(bench, stress, "darts")
        want_m = PUBLISHED_DARTS_MEAN[(bench, stress)]
        want_ratio = PUBLISHED_RATIO[(bench, stress)]
        ok_n = n == 10
        ok_m = mean_r is not None and abs(mean_r - want_m) <= MEAS_TOL_MS
        ratio = (got.R_G_off_ms / mean_r) if mean_r else float("nan")
        ok_ratio = mean_r is not None and round(ratio, 2) == want_ratio
        ok = ok_int and ok_n and ok_m and ok_ratio
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        meas_s = "None" if mean_r is None else f"{mean_r:.1f}±{(std_r or 0):.1f}"
        print(
            f"  [{status}] {cell}  round(R)={round(got.R_G_off_ms)} want={want_r}  "
            f"meas={meas_s} (n={n} want 10, target {want_m})  "
            f"ratio={ratio:.2f} want={want_ratio:.2f}"
        )

    print()
    print("## (c) T0 H3, not published")
    inst_b3 = nominal_instance("B3_IQI", "medium")
    r_rt = s0_path_sum(inst_b3, env, "runtime")
    meas_b3, _, n_b3 = vg.measured_r_g("B3_IQI", "medium", "darts")
    ratio_rt = (r_rt.R_G_off_ms / meas_b3) if meas_b3 else float("nan")
    print(
        f"  B3 xi_source=runtime  R_G_off={r_rt.R_G_off_ms:.6f}  "
        f"(expect {H3_B3_RUNTIME_R})  "
        f"ratio vs {meas_b3} (n={n_b3}) = {ratio_rt:.2f}  "
        f"(expect {H3_B3_RUNTIME_RATIO})"
    )

    if n_fail:
        print(f"\nanchor: {n_fail} FAIL")
        return 1
    print("\nanchor: all (a)/(b) PASS")
    return 0


def cmd_frontier(args: argparse.Namespace) -> int:
    env = PUBLISHED_ENVELOPES
    print_envelopes(env)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    betas = _parse_floats(args.beta_grid) if args.beta_grid else list(DEFAULT_BETA_GRID)
    sources = _xi_sources(args.xi_source)
    tests = _parse_tests(args.tests)
    all_rows: List[Row] = []

    for test_name in tests:
        print()
        print(f"## {test_name}")
        print("| cell | xi | R_G_off | D_e2e | D_rel | beta_star_e2e | beta_star_rel |")
        print("|---|---|---|---|---|---|---|")
        for cell in NOMINAL_CELLS:
            bench, stress = parse_cell(cell)
            nom = nominal_instance(bench, stress)
            for xi in sources:
                bound = TESTS[test_name].fn(nom, env, xi)
                r = bound.R_G_off_ms
                bs_e2e = r / nom.global_deadline_ms
                bs_rel = r / nom.relative_deadline_ms
                print(
                    f"| {cell} | {xi} | {r:.6f} | {nom.global_deadline_ms:.0f} | "
                    f"{nom.relative_deadline_ms:.0f} | {bs_e2e:.4f} | {bs_rel:.4f} |"
                )
                for beta in betas:
                    scaled = nom.scaled(beta)
                    rows = evaluate(scaled, [test_name], env, xi)
                    all_rows.extend(rows)

        print()
        print(f"accept_e2e on the beta grid, test={test_name} (1=accept):")
        header = "cell xi " + " ".join(f"{b:g}" for b in betas)
        print(header)
        for cell in NOMINAL_CELLS:
            bench, stress = parse_cell(cell)
            nom = nominal_instance(bench, stress)
            for xi in sources:
                bits = []
                for beta in betas:
                    row = evaluate(nom.scaled(beta), [test_name], env, xi)[0]
                    bits.append(str(int(row.accept_e2e)))
                print(f"{cell} {xi} " + " ".join(bits))

    if getattr(args, "show_s0w", False):
        print()
        print("## informational S0-W (not a registered test)")
        print(
            "# tau = diam(W)/u with W=[0,L]^2; every start pose in W is admissible, "
            "so the published (0,0)-corner start set under-covers the deployment."
        )
        nom_b1 = nominal_instance("B1_CDR", "medium")
        tau_w, r_w = _s0_w_informational(nom_b1, env, "declared")
        print(
            f"# tau_W = {tau_w:.4f} ms/node "
            f"({env.map_size_m * (2.0 ** 0.5):.7f}/{env.u_mps} s)"
        )
        print(
            f"# B1_CDR declared: R_G_off = {r_w:.1f} ms  (informational; not in TESTS)"
        )

    csv_path = out_dir / "frontier.csv"
    md_path = out_dir / "frontier.md"
    write_csv(csv_path, all_rows, env)
    with md_path.open("w") as fh:
        fh.write("# TB-3 offline frontier (nominal instances)\n\n")
        for test_name in tests:
            fh.write(f"## {test_name}\n\n")
            fh.write(
                "| cell | xi | R_G_off | D_e2e | D_rel | beta_star_e2e | beta_star_rel |\n"
            )
            fh.write("|---|---|---|---|---|---|---|\n")
            for cell in NOMINAL_CELLS:
                bench, stress = parse_cell(cell)
                nom = nominal_instance(bench, stress)
                for xi in sources:
                    bound = TESTS[test_name].fn(nom, env, xi)
                    r = bound.R_G_off_ms
                    fh.write(
                        f"| {cell} | {xi} | {r:.6f} | {nom.global_deadline_ms:.0f} | "
                        f"{nom.relative_deadline_ms:.0f} | "
                        f"{r / nom.global_deadline_ms:.4f} | "
                        f"{r / nom.relative_deadline_ms:.4f} |\n"
                    )
            fh.write("\n")
        fh.write(
            "P0 beta grid is 12 points "
            "(0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,2.0); "
            "the spec's \"13 points\" does not match this literal.\n"
        )
    print(f"\nwrote {csv_path} and {md_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    env = PUBLISHED_ENVELOPES
    print_envelopes(env)
    bench, stress = parse_cell(args.cell)
    nom = nominal_instance(bench, stress)
    seeds = _parse_instance_seeds(args.instance_seeds, args.instance_seed)
    bases: List[MissionInstance]
    if seeds is None:
        bases = [nom]
    else:
        bases = [apply_instance_seed(nom, s) for s in seeds]
    betas = _parse_floats(args.beta)
    tests = _parse_tests(args.tests)
    sources = _xi_sources(args.xi_source)
    out_dir = Path(args.out)
    all_rows: List[Row] = []
    print()
    print(
        "seed beta test xi R_G_off D_e2e D_rel accept_e2e accept_rel load_e2e load_rel"
    )
    for base in bases:
        seed_label = "nominal" if base.instance_seed is None else str(base.instance_seed)
        for beta in betas:
            inst = base.scaled(beta)
            for xi in sources:
                rows = evaluate(inst, tests, env, xi)
                all_rows.extend(rows)
                for row in rows:
                    print(
                        f"{seed_label} {inst.beta:g} {row.result.test} {xi} "
                        f"{row.result.R_G_off_ms:.6f} "
                        f"{inst.global_deadline_ms:.1f} {inst.relative_deadline_ms:.1f} "
                        f"{int(row.accept_e2e)} {int(row.accept_rel)} "
                        f"{row.load_e2e:.4f} {row.load_rel:.4f}"
                    )
    write_csv(out_dir / "eval.csv", all_rows, env)
    print(f"wrote {out_dir / 'eval.csv'}")
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Acceptance-ratio grid: one instance per (cell, seed), then every beta.

    Row order: cells in ``GRID_CELLS_TB3`` (or ``--cells``), seeds ascending,
    betas ascending, tests in ``--tests`` order (canonical: S0 then S1).
    """
    t0 = time.perf_counter()
    env = PUBLISHED_ENVELOPES
    print_envelopes(env)

    cells_raw = args.cells or ",".join(GRID_CELLS_TB3)
    cells = [c.strip() for c in cells_raw.split(",") if c.strip()]
    seeds = _parse_instance_seeds(args.seeds, None)
    if seeds is None:
        seeds = list(GRID_SEEDS_TB3)
    betas = (
        _parse_floats(args.beta_grid) if args.beta_grid else list(BETA_GRID_TB3)
    )
    tests = _parse_tests(args.tests)
    if args.xi_source == "both":
        raise ValueError("grid writes one xi_source; pass declared or runtime")
    xi = args.xi_source

    dicts: List[Dict[str, Any]] = []
    for cell in cells:
        bench, stress = parse_cell(cell)
        nom = nominal_instance(bench, stress)
        for seed in seeds:
            inst0 = apply_instance_seed(nom, seed)
            for beta in betas:
                inst = inst0.scaled(beta)
                for row in evaluate(inst, tests, env, xi):
                    dicts.append(row_to_csv_dict(row, env))

    out_dir = Path(args.out)
    raw_path = out_dir / "tb3_offline_grid.csv"
    write_grid_csv(raw_path, dicts)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    wall_s = time.perf_counter() - t0
    print(f"wrote {raw_path} ({len(dicts)} rows)")
    print(f"sha256 {digest}")
    print(f"wall_s {wall_s:.3f}")

    curves, frontier = aggregate_grid(dicts)
    curves_path = Path(args.curves)
    frontier_path = Path(args.frontier)
    _write_header_csv(curves_path, TB3_CURVES_HEADER, curves, _CURVES_FLOAT_FIELDS)
    _write_header_csv(
        frontier_path, TB3_FRONTIER_HEADER, frontier, _FRONTIER_FLOAT_FIELDS
    )
    print(f"wrote {curves_path} ({len(curves)} rows)")
    print(f"wrote {frontier_path} ({len(frontier)} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="TB-3 offline schedulability harness (eq. 26 / S0)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_anchor = sub.add_parser("anchor", help="S0 tripwire + published T3 table.")
    p_anchor.add_argument(
        "--new_sweep_dir",
        default="data/raw/behavioral/sweep_a1a2b1",
    )
    p_anchor.add_argument(
        "--old_results_dir",
        default="data/raw/behavioral/legacy_medium",
    )
    p_anchor.add_argument("--out", default="results_tb3_offline/anchor")
    p_anchor.set_defaults(func=cmd_anchor)

    p_front = sub.add_parser("frontier", help="Nominal-instance beta* and accept grid.")
    p_front.add_argument(
        "--xi-source",
        dest="xi_source",
        default=XI_SOURCE_DEFAULT,
        choices=("declared", "runtime", "both"),
    )
    p_front.add_argument(
        "--beta-grid",
        dest="beta_grid",
        default=",".join(str(b) for b in DEFAULT_BETA_GRID),
    )
    p_front.add_argument(
        "--tests",
        default=TESTS_DEFAULT,
        help="Comma-separated registry names (default S0,S1).",
    )
    p_front.add_argument(
        "--show-s0w",
        dest="show_s0w",
        action="store_true",
        default=False,
        help="Print the informational start-set-corrected S0-W line (not a registered test).",
    )
    p_front.add_argument("--out", default="results_tb3_offline/frontier")
    p_front.set_defaults(func=cmd_frontier)

    p_eval = sub.add_parser("eval", help="One (cell, beta-list, tests) unit.")
    p_eval.add_argument("--cell", required=True)
    p_eval.add_argument("--beta", required=True)
    p_eval.add_argument("--tests", default=TESTS_DEFAULT)
    p_eval.add_argument(
        "--xi-source",
        dest="xi_source",
        default=XI_SOURCE_DEFAULT,
        choices=("declared", "runtime", "both"),
    )
    p_eval.add_argument("--instance-seed", dest="instance_seed", type=int, default=None)
    p_eval.add_argument(
        "--instance-seeds",
        dest="instance_seeds",
        default=None,
        help="Comma list and/or a..b ranges (e.g. 0..4 or 0,2,5..7). TB3-4 loops here.",
    )
    p_eval.add_argument("--out", default="results_tb3_offline/eval")
    p_eval.set_defaults(func=cmd_eval)

    p_grid = sub.add_parser(
        "grid",
        help="Offline acceptance grid (cells x seeds x betas x tests).",
    )
    p_grid.add_argument(
        "--xi-source",
        dest="xi_source",
        default=XI_SOURCE_DEFAULT,
        choices=("declared", "runtime", "both"),
    )
    p_grid.add_argument("--tests", default=TESTS_DEFAULT)
    p_grid.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cells (default: GRID_CELLS_TB3).",
    )
    p_grid.add_argument(
        "--seeds",
        default=None,
        help="Comma list and/or a..b ranges (default: 0..199).",
    )
    p_grid.add_argument(
        "--beta-grid",
        dest="beta_grid",
        default=None,
        help="Comma-separated betas (default: BETA_GRID_TB3).",
    )
    p_grid.add_argument("--out", default="results_tb3_offline/grid")
    p_grid.add_argument(
        "--curves",
        default="docs/tb3/tb3_acceptance_curves.csv",
        help="Committed aggregate curves CSV.",
    )
    p_grid.add_argument(
        "--frontier",
        default="docs/tb3/tb3_frontier_summary.csv",
        help="Committed per-(cell, test) frontier summary CSV.",
    )
    p_grid.set_defaults(func=cmd_grid)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
