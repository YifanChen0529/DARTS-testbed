#!/usr/bin/env python3
"""Offline R̄^off_G computation for §V.G T3.

For each (benchmark, stress) cell:
  R̄^off_v   = 2H + T̄_auc + (ξ_v + h + τ̄)
  R̄^off_G   = max over source-to-sink paths P of  Σ_{v∈P} (J̄_v + R̄^off_v)

τ̄  = (workspace diameter) / u_i,max  (worst-case travel between any two
     task locations OR between the agent's start position and any task
     location, per Theorem 2(a) lifting argument).

Use task_locations_m from the Workload dataclass. The diameter is computed
including (0,0) as a conservative start position; adjust via --start_pos
if your deployment uses a different parking spot.
"""
from __future__ import annotations
import argparse, sys, itertools, math
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.workloads import get_workload
from alg.stress_profiles import get_stress_profile
from alg.config_darts import DARTS_CONFIG


def euclid(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def workspace_diameter_m(task_locations: Dict[str, Tuple[float, float]],
                         start_pos: Tuple[float, float]) -> float:
    """Max pairwise Euclidean distance over (start_pos ∪ task_locations)."""
    points = list(task_locations.values()) + [start_pos]
    return max(euclid(a, b) for a, b in itertools.combinations(points, 2))


def dag_width(node_ids: List[str], predecessors: Dict[str, List[str]]) -> int:
    """Layered BFS width = max width across topological layers."""
    in_deg = {n: len(predecessors.get(n, [])) for n in node_ids}
    succ: Dict[str, List[str]] = {n: [] for n in node_ids}
    for n, preds in predecessors.items():
        for p in preds:
            succ[p].append(n)
    layer = [n for n in node_ids if in_deg[n] == 0]
    max_w = len(layer)
    while layer:
        nxt = []
        for n in layer:
            for s in succ[n]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    nxt.append(s)
        max_w = max(max_w, len(nxt))
        layer = nxt
    return max_w


def enumerate_paths(node_ids: List[str], predecessors: Dict[str, List[str]]) -> List[List[str]]:
    """All source-to-sink paths in the DAG."""
    succ: Dict[str, List[str]] = {n: [] for n in node_ids}
    for n, preds in predecessors.items():
        for p in preds:
            succ[p].append(n)
    sources = [n for n in node_ids if not predecessors.get(n)]
    sinks = [n for n in node_ids if not succ[n]]
    paths: List[List[str]] = []

    def dfs(node, path):
        path = path + [node]
        if node in sinks:
            paths.append(path)
            return
        for s in succ[node]:
            dfs(s, path)

    for src in sources:
        dfs(src, [])
    return paths


def compute_r_off_g(workload_id: str, stress_level: str,
                    delta_auc_ms: float, H_ms: float, j_bar_ms: float,
                    ui_max_mps: float, h_comm_ms: float,
                    start_pos: Tuple[float, float]):
    wl = get_workload(workload_id)
    # Stress overrides do not change ξ_v / topology / locations; they only
    # change deadlines and batteries. So per-node R̄^off_v is stress-independent.
    _ = get_stress_profile(workload_id, stress_level)
    diam_m = workspace_diameter_m(wl.task_locations_m, start_pos)
    tau_bar_ms = (diam_m / ui_max_mps) * 1000.0   # m / (m/s) = s → ms

    A_bar = dag_width(list(wl.node_ids), dict(wl.predecessors))
    T_bar_auc_ms = A_bar * delta_auc_ms

    r_off_v = {}
    for v in wl.node_ids:
        xi_v = wl.task_wcets_ms[v]
        r_off_v[v] = 2.0 * H_ms + T_bar_auc_ms + (xi_v + h_comm_ms + tau_bar_ms)

    paths = enumerate_paths(list(wl.node_ids), dict(wl.predecessors))
    best_path, best_sum = None, -1.0
    for path in paths:
        s = sum(j_bar_ms + r_off_v[v] for v in path)
        if s > best_sum:
            best_sum, best_path = s, path
    return {
        'workload_id': workload_id,
        'stress_level': stress_level,
        'A_bar': A_bar,
        'T_bar_auc_ms': T_bar_auc_ms,
        'tau_bar_ms': tau_bar_ms,
        'diameter_m': diam_m,
        'longest_path': best_path,
        'R_off_G_ms': best_sum,
        'per_node_R_off_v_ms': r_off_v,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta_auc_ms', type=float, required=True,
                    help='Calibrated subround WCET (from §V.G T1 analysis).')
    ap.add_argument('--H_ms', type=float, default=None,
                    help='Epoch length (default: from DARTS_CONFIG.epoch_length_ms or equivalent).')
    ap.add_argument('--j_bar_ms', type=float, default=0.0,
                    help='Readiness-jitter bound J̄ (default 0 = no jitter).')
    ap.add_argument('--ui_max_mps', type=float, default=None,
                    help='Min agent max-velocity in m/s (default: from DARTS_CONFIG).')
    ap.add_argument('--h_comm_ms', type=float, default=50.0,
                    help='Communication overhead h (default 50 ms).')
    ap.add_argument('--start_x', type=float, default=0.0)
    ap.add_argument('--start_y', type=float, default=0.0)
    ap.add_argument('--benchmarks', type=str, default='B1_CDR,B2_LML,B3_IQI,B4_DEM')
    ap.add_argument('--stress_levels', type=str, default='light,medium,heavy')
    args = ap.parse_args()

    cfg = DARTS_CONFIG()
    H_ms = args.H_ms if args.H_ms is not None else getattr(cfg, 'epoch_length_ms', 1000.0)
    ui_max = args.ui_max_mps if args.ui_max_mps is not None else getattr(cfg, 'ui_max_mps', None)
    if ui_max is None:
        # Fallback: try other plausible attribute names
        ui_max = getattr(cfg, 'max_velocity_mps', None) or getattr(cfg, 'agent_velocity_mps', None)
    if ui_max is None:
        from envs.turtlebot3_spatial_env import MAX_VELOCITY_MPS
        ui_max = MAX_VELOCITY_MPS
    start_pos = (args.start_x, args.start_y)

    print("# Constants")
    print(f"- δ_auc = {args.delta_auc_ms} ms (CLI)")
    print(f"- H = {H_ms} ms")
    print(f"- J̄ = {args.j_bar_ms} ms")
    print(f"- u_i,max = {ui_max} m/s")
    print(f"- h = {args.h_comm_ms} ms")
    print(f"- start_pos = ({args.start_x}, {args.start_y}) m")
    print()

    print("# §V.G T3 — Offline R̄^off_G per (benchmark, stress)")
    print()
    print("| Benchmark | Stress | Ā | τ̄ (ms) | T̄_auc (ms) | Longest path | R̄^off_G (ms) |")
    print("|---|---|---|---|---|---|---|")
    for bench in args.benchmarks.split(','):
        for stress in args.stress_levels.split(','):
            try:
                r = compute_r_off_g(bench, stress,
                                    args.delta_auc_ms, H_ms, args.j_bar_ms,
                                    ui_max, args.h_comm_ms, start_pos)
                path_str = ' → '.join(r['longest_path'])
                print(f"| {bench} | {stress} | {r['A_bar']} | {r['tau_bar_ms']:.1f} | "
                      f"{r['T_bar_auc_ms']:.1f} | {path_str} | **{r['R_off_G_ms']:.1f}** |")
            except Exception as e:
                print(f"| {bench} | {stress} | — | — | — | error: {e} | — |")


if __name__ == '__main__':
    main()
