#!/usr/bin/env python3
"""Theorem 2(a) offline admission test (27) + ladder solver.

Standalone, x86-safe. Does not import alg.train_darts. Helpers from
``scripts/r_off_g.py`` are imported read-only (that file stays byte-frozen).

Test (27): for each agent i,
    delta_sum^i = sum_{v alloc to i} c_bar_v / (D_v - Lambda_lat)  <= 1
with c_bar_v = xi_v + h + tau_bar and Lambda_lat = 2H + A_bar * delta_auc.

xi_v is the RUNTIME-EFFECTIVE WCET from DAGGraph.from_workload(workload,
DARTS_CONFIG()), which reproduces the B1-merge on B3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alg.config_darts import DARTS_CONFIG
from envs.dag_task import DAGGraph
from envs.workloads import get_workload
from envs.turtlebot3_spatial_env import MAX_VELOCITY_MPS, TurtleBot3Env
from scripts.r_off_g import dag_width, euclid, workspace_diameter_m

H_MS = 1000.0
H_COMM_MS = 50.0
DELTA_AUC_MS = 30.0
J_BAR_MS = 0.0
U_MPS = 0.22
START_POS = (0.0, 0.0)
OBSTACLE_PENALTY = 1.5
SAFETY_MARGIN = 1.1
DEFAULT_OBSTACLES = (
    (3.0, 3.0, 1.5, 1.5),
    (5.0, 6.0, 1.0, 2.0),
    (7.0, 2.0, 1.2, 1.2),
)


def _identity_alloc(node_ids: Sequence[str], num_agents: int = 4) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, nid in enumerate(node_ids):
        out[str(nid)] = f"agent_{(i % num_agents) + 1}"
    return out


def _load_alloc_map(path: Optional[str], node_ids: Sequence[str]) -> Dict[str, str]:
    if not path:
        return _identity_alloc(node_ids)
    with open(path, "r") as f:
        raw = json.load(f)
    return {str(k): str(v) for k, v in raw.items()}


def _effective_and_declared_xi(workload_id: str) -> Tuple[Dict[str, float], Dict[str, float], Any, Any]:
    wl = get_workload(workload_id)
    cfg = DARTS_CONFIG()
    dag = DAGGraph.from_workload(wl, cfg)
    declared = {nid: float(wl.task_wcets_ms[nid]) for nid in wl.node_ids}
    effective = {nid: float(dag.nodes[nid].wcet) for nid in wl.node_ids}
    return declared, effective, wl, dag


def _tau_bar_conventions_ms(task_locations: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Return (tau_raw_ms, tau_impl_ms).

    Raw: workspace-diameter / u, matching scripts/r_off_g.py.
    Impl: mirror turtlebot3_spatial_env.py:276-294 (distance *= 1.5 on the first
    intersecting default obstacle) then :322-353 (travel_time_ms * 1.1).
    For tau_bar we take the max pairwise travel (no xi term).
    """
    diam_m = workspace_diameter_m(task_locations, START_POS)
    tau_raw = (diam_m / U_MPS) * 1000.0
    env = TurtleBot3Env(num_agents=1)
    points = list(task_locations.values()) + [START_POS]
    tau_impl = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            ax, ay = points[i]
            bx, by = points[j]
            distance_m = euclid(points[i], points[j])
            for (ox_r, oy_r, w, h) in DEFAULT_OBSTACLES:
                if env._line_intersects_rect(ax, ay, bx, by, ox_r, oy_r, w, h):
                    distance_m *= OBSTACLE_PENALTY
                    break
            travel_time_ms = (distance_m / U_MPS) * 1000.0
            travel_impl = travel_time_ms * SAFETY_MARGIN
            if travel_impl > tau_impl:
                tau_impl = travel_impl
    return tau_raw, tau_impl


def _r_bar_v(
    node_ids: Sequence[str],
    predecessors: Dict[str, List[str]],
    r_off_v: Dict[str, float],
) -> Dict[str, float]:
    """Offline worst-case readiness: 0 for sources; else max_p (r_bar_p + R_off_p)."""
    r_bar: Dict[str, float] = {}
    remaining = set(node_ids)
    guard = 0
    while remaining and guard < len(node_ids) + 2:
        guard += 1
        progressed = False
        for v in list(remaining):
            preds = list(predecessors.get(v, []))
            if any(p not in r_bar for p in preds):
                continue
            if not preds:
                r_bar[v] = 0.0
            else:
                r_bar[v] = max(r_bar[p] + r_off_v[p] for p in preds)
            remaining.remove(v)
            progressed = True
        if not progressed:
            break
    for v in remaining:
        r_bar[v] = float("inf")
    return r_bar


def _all_sources(predecessors: Dict[str, List[str]], node_ids: Sequence[str]) -> bool:
    return all(not predecessors.get(n, []) for n in node_ids)


def evaluate_thm2a(
    workload_id: str,
    alloc_map: Dict[str, str],
    D_v_ms: float,
    P_ms: float,
    impl_factors: bool = True,
    L: int = 1,
) -> Dict[str, Any]:
    declared, effective, wl, dag = _effective_and_declared_xi(workload_id)
    divergences = {
        nid: (declared[nid], effective[nid])
        for nid in wl.node_ids
        if abs(declared[nid] - effective[nid]) > 1e-9
    }
    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    Lambda_lat = 2.0 * H_MS + float(A_bar) * DELTA_AUC_MS
    tau_raw, tau_impl = _tau_bar_conventions_ms(dict(wl.task_locations_m))
    tau_bar = tau_impl if impl_factors else tau_raw
    factor_convention = "on (1.5x path-metric, 1.1x travel)" if impl_factors else "off (raw diameter)"

    jobs: List[Dict[str, Any]] = []
    c_bar: Dict[str, float] = {}
    r_off: Dict[str, float] = {}
    for nid in wl.node_ids:
        xi = effective[nid]
        c = xi + H_COMM_MS + tau_bar
        c_bar[nid] = c
        r_off[nid] = Lambda_lat + xi + H_COMM_MS + tau_bar + J_BAR_MS
        denom = float(D_v_ms) - Lambda_lat
        delta_v = float("inf") if denom <= 0 else c / denom
        jobs.append({
            "class": nid,
            "xi_declared": declared[nid],
            "xi_effective": xi,
            "tau_bar_raw": tau_raw,
            "tau_bar_impl": tau_impl,
            "tau_bar_used": tau_bar,
            "c_bar": c,
            "delta_v": delta_v,
            "agent": alloc_map.get(nid, ""),
        })

    agents = sorted({alloc_map.get(nid, "") for nid in wl.node_ids if alloc_map.get(nid)})
    if not agents:
        agents = sorted(set(_identity_alloc(wl.node_ids).values()))
    delta_sum: Dict[str, float] = {a: 0.0 for a in agents}
    for nid in wl.node_ids:
        a = alloc_map.get(nid)
        if not a:
            continue
        delta_sum.setdefault(a, 0.0)
        dv = next(j["delta_v"] for j in jobs if j["class"] == nid)
        delta_sum[a] += dv

    r_bar = _r_bar_v(list(wl.node_ids), dict(wl.predecessors), r_off)
    assumption_ok = True
    assumption_detail: Dict[str, Any] = {}
    for nid in wl.node_ids:
        lhs = r_bar[nid] + float(D_v_ms)
        ok = lhs <= float(P_ms) + 1e-9
        assumption_detail[nid] = {
            "r_bar_v": r_bar[nid],
            "r_bar_plus_Dv": lhs,
            "P": float(P_ms),
            "ok": ok,
        }
        if not ok:
            assumption_ok = False

    bottleneck = max(delta_sum.values()) if delta_sum else float("inf")
    denom_ok = float(D_v_ms) > Lambda_lat
    test27 = denom_ok and bottleneck <= 1.0 + 1e-12
    if not assumption_ok:
        verdict = "REJECT"
        reject_reason = "not a valid Theorem 2(a) instance"
    elif not denom_ok:
        verdict = "REJECT"
        reject_reason = "D_v <= Lambda_lat"
    elif not test27:
        verdict = "REJECT"
        reject_reason = "delta_sum^i > 1"
    else:
        verdict = "ADMIT"
        reject_reason = ""

    return {
        "workload_id": workload_id,
        "declared_xi": declared,
        "effective_xi": effective,
        "xi_divergences": divergences,
        "A_bar": A_bar,
        "Lambda_lat": Lambda_lat,
        "tau_bar_raw": tau_raw,
        "tau_bar_impl": tau_impl,
        "tau_bar_used": tau_bar,
        "factor_convention": factor_convention,
        "impl_factors": impl_factors,
        "h_ms": H_COMM_MS,
        "H_ms": H_MS,
        "delta_auc_ms": DELTA_AUC_MS,
        "J_bar_ms": J_BAR_MS,
        "u_mps": U_MPS,
        "D_v_ms": float(D_v_ms),
        "P_ms": float(P_ms),
        "L": int(L),
        "jobs": jobs,
        "delta_sum": delta_sum,
        "bottleneck": bottleneck,
        "assumption_ok": assumption_ok,
        "assumption_detail": assumption_detail,
        "test27": test27,
        "verdict": verdict,
        "reject_reason": reject_reason,
        "all_sources": _all_sources(dict(wl.predecessors), list(wl.node_ids)),
        "r_bar": r_bar,
        "r_off": r_off,
        "alloc_map": dict(alloc_map),
    }


def solve_ladder(
    workload_id: str,
    alloc_map: Dict[str, str],
    target_delta_sum: float,
    impl_factors: bool = True,
    L: int = 1,
    d_v_cap_ms: float = 10_000_000.0,
) -> Dict[str, Any]:
    """Smallest D_v (multiple of 1000 ms) s.t. bottleneck delta_sum^i <= X, then P."""
    declared, effective, wl, dag = _effective_and_declared_xi(workload_id)
    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    Lambda_lat = 2.0 * H_MS + float(A_bar) * DELTA_AUC_MS
    start = int(((int(Lambda_lat) // 1000) + 1) * 1000)
    chosen_dv = None
    last = None
    d = start
    while d <= d_v_cap_ms:
        # P placeholder = d so assumption is evaluated after we pick P
        probe = evaluate_thm2a(
            workload_id, alloc_map, D_v_ms=float(d), P_ms=float(d),
            impl_factors=impl_factors, L=L,
        )
        last = probe
        if probe["bottleneck"] <= target_delta_sum + 1e-12 and probe["D_v_ms"] > Lambda_lat:
            chosen_dv = float(d)
            break
        d += 1000
    if chosen_dv is None:
        out = last or evaluate_thm2a(
            workload_id, alloc_map, D_v_ms=d_v_cap_ms, P_ms=d_v_cap_ms,
            impl_factors=impl_factors, L=L,
        )
        out["solver"] = "FAILED"
        out["target_delta_sum"] = target_delta_sum
        return out

    all_src = _all_sources(dict(wl.predecessors), list(wl.node_ids))
    if all_src:
        P = chosen_dv
    else:
        # smallest 1000-ms multiple P satisfying r_bar_v + D_v <= P
        probe_dv = evaluate_thm2a(
            workload_id, alloc_map, D_v_ms=chosen_dv, P_ms=chosen_dv,
            impl_factors=impl_factors, L=L,
        )
        need = max(probe_dv["r_bar"][nid] + chosen_dv for nid in wl.node_ids)
        P = float(((int(need) + 999) // 1000) * 1000)
        if P < chosen_dv:
            P = chosen_dv
    result = evaluate_thm2a(
        workload_id, alloc_map, D_v_ms=chosen_dv, P_ms=P,
        impl_factors=impl_factors, L=L,
    )
    result["solver"] = "OK"
    result["target_delta_sum"] = target_delta_sum
    return result


def format_table(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Theorem 2(a) offline test  workload={result['workload_id']}")
    lines.append(
        f"A_bar={result['A_bar']}  Lambda_lat={result['Lambda_lat']:.1f} ms  "
        f"D_v={result['D_v_ms']:.0f} ms  P={result['P_ms']:.0f} ms  L={result['L']}"
    )
    lines.append(
        f"h={result['h_ms']:.0f} ms  H={result['H_ms']:.0f} ms  "
        f"delta_auc={result['delta_auc_ms']:.0f} ms  J_bar={result['J_bar_ms']:.0f}  "
        f"u={result['u_mps']} m/s"
    )
    lines.append(
        f"tau_bar raw={result['tau_bar_raw']:.6f} ms  "
        f"tau_bar impl={result['tau_bar_impl']:.6f} ms  "
        f"used={result['tau_bar_used']:.6f} ms  convention={result['factor_convention']}"
    )
    if result["xi_divergences"]:
        lines.append("WCET DIVERGENCE declared -> effective:")
        for nid, (d, e) in result["xi_divergences"].items():
            lines.append(f"  {nid}: {d:.1f} -> {e:.1f}")
    else:
        lines.append("WCET declared == effective (no B1-merge divergence)")
    lines.append("declared xi: " + ", ".join(f"{k}={v:.1f}" for k, v in result["declared_xi"].items()))
    lines.append("effective xi: " + ", ".join(f"{k}={v:.1f}" for k, v in result["effective_xi"].items()))
    lines.append("")
    lines.append(
        f"{'class':<8} {'xi_eff':>10} {'tau_bar':>12} {'c_bar':>12} {'delta_v':>12} {'agent':<12}"
    )
    for j in result["jobs"]:
        lines.append(
            f"{j['class']:<8} {j['xi_effective']:10.1f} {j['tau_bar_used']:12.3f} "
            f"{j['c_bar']:12.3f} {j['delta_v']:12.6f} {j['agent']:<12}"
        )
    lines.append("")
    lines.append("per-agent delta_sum^i:")
    for a, s in sorted(result["delta_sum"].items()):
        lines.append(f"  {a}: {s:.10f}")
    lines.append(f"bottleneck={result['bottleneck']:.10f}")
    if "target_delta_sum" in result:
        lines.append(f"target_delta_sum={result['target_delta_sum']}")
        lines.append(f"solver={result.get('solver', '')}")
    lines.append(
        f"assumption-check: {'PASS' if result['assumption_ok'] else 'FAIL'} "
        f"(r_bar_v + D_v <= P for every class)"
    )
    for nid, det in result["assumption_detail"].items():
        lines.append(
            f"  {nid}: r_bar={det['r_bar_v']:.3f} + D_v={result['D_v_ms']:.0f} "
            f"= {det['r_bar_plus_Dv']:.3f}  vs P={det['P']:.0f}  "
            f"{'OK' if det['ok'] else 'VIOLATION'}"
        )
    extra = f" ({result['reject_reason']})" if result["reject_reason"] else ""
    lines.append(f"verdict: {result['verdict']}{extra}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Theorem 2(a) offline admission + ladder solver")
    ap.add_argument("--workload", type=str, default="B4_DEM")
    ap.add_argument("--alloc_map", type=str, default=None)
    ap.add_argument("--relative_deadline_ms", type=float, default=None)
    ap.add_argument("--period_ms", type=float, default=None)
    ap.add_argument("--target_delta_sum", type=float, default=None)
    ap.add_argument("--impl_factors", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--L", type=int, default=1, help="L placeholder printed in the table")
    args = ap.parse_args()

    _, _, wl, _ = _effective_and_declared_xi(args.workload)
    alloc = _load_alloc_map(args.alloc_map, list(wl.node_ids))
    impl = args.impl_factors == "on"

    if args.target_delta_sum is not None:
        result = solve_ladder(
            args.workload, alloc, float(args.target_delta_sum),
            impl_factors=impl, L=int(args.L),
        )
    else:
        if args.relative_deadline_ms is None or args.period_ms is None:
            ap.error("provide --relative_deadline_ms and --period_ms, or --target_delta_sum")
        result = evaluate_thm2a(
            args.workload, alloc,
            D_v_ms=float(args.relative_deadline_ms),
            P_ms=float(args.period_ms),
            impl_factors=impl, L=int(args.L),
        )
    sys.stdout.write(format_table(result))
    if result["verdict"] != "ADMIT":
        sys.exit(2)


if __name__ == "__main__":
    main()
