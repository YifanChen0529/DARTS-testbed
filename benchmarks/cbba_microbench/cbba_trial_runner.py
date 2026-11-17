#!/usr/bin/env python3
"""Coordinator-side CBBA trial runner. Headline T_conv uses this process's clock.

Nodes are started by the launcher; this process only waits, drives trials, and writes CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import grpc

from benchmarks.cbba_microbench import cbba_pb2 as pb2
from benchmarks.cbba_microbench import cbba_pb2_grpc as pb2_grpc
from benchmarks.cbba_microbench.cbba_core import generate_instance, run_sync_instance
from scripts.csv_schema_tb import CBBA_LOG_HEADER

UNASSIGNED = -1
N_AGENTS = 4


class StaleDirError(SystemExit):
    pass


def _median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def parse_config_grid(spec: str) -> List[Tuple[str, int, int]]:
    """Parse 'm=4,8,16,32;lt=1,ceil' into (config_id, m, l_t) slots.

    The lt=ceil token is kept as a distinct slot even when ceil(m/4)==1, so
    the smoke identity 4 m-levels x 2 lt-levels holds.
    """
    fields: Dict[str, str] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            raise ValueError(f"bad --config_grid fragment: {part!r}")
        k, v = part.split("=", 1)
        fields[k.strip()] = v.strip()
    if "m" not in fields or "lt" not in fields:
        raise ValueError("--config_grid must contain m=... and lt=...")
    ms = [int(x) for x in fields["m"].split(",") if x.strip()]
    lt_tokens = [x.strip() for x in fields["lt"].split(",") if x.strip()]
    out: List[Tuple[str, int, int]] = []
    for m in ms:
        for tok in lt_tokens:
            if tok == "ceil":
                l_t = int(math.ceil(m / 4))
                cid = f"m{m}_ltceil"
            else:
                l_t = int(tok)
                cid = f"m{m}_lt{l_t}"
            out.append((cid, m, l_t))
    return out


def assert_fresh_out_dir(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return
    leftover = sorted(os.listdir(out_dir))
    if leftover:
        raise StaleDirError(
            "REFUSAL: out_dir is not empty (move or delete it first): "
            f"{out_dir}\n  " + "\n  ".join(leftover[:20])
        )


def wait_nodes(hostports: Sequence[str], timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    pending = list(hostports)
    while pending:
        if time.monotonic() >= deadline:
            raise SystemExit(f"timeout waiting for nodes: {pending}")
        still = []
        for hp in pending:
            try:
                ch = grpc.insecure_channel(hp)
                grpc.channel_ready_future(ch).result(timeout=1.0)
                ch.close()
            except Exception:
                still.append(hp)
                time.sleep(0.05)
        pending = still


def _control_stubs(hostports: Sequence[str]) -> Tuple[List[grpc.Channel], List[pb2_grpc.CbbaControlStub]]:
    channels = [grpc.insecure_channel(hp) for hp in hostports]
    stubs = [pb2_grpc.CbbaControlStub(ch) for ch in channels]
    return channels, stubs


def _flatten_xy(points: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for x, y in points:
        out.append(float(x))
        out.append(float(y))
    return out


def drive_trial(
    stubs: Sequence[pb2_grpc.CbbaControlStub],
    trial_id: int,
    m: int,
    l_t: int,
    tasks: Sequence[Tuple[float, float]],
    starts: Sequence[Tuple[float, float]],
    poll_deadline_s: float = 120.0,
) -> Tuple[float, List[pb2.TrialDone]]:
    spec = pb2.StartTrial(
        trial_id=trial_id,
        m=m,
        l_t=l_t,
        task_xy=_flatten_xy(tasks),
        start_xy=_flatten_xy(starts),
    )
    t0 = time.perf_counter()
    for stub in stubs:
        stub.BeginTrial(spec, timeout=30.0)
    dones: List[Optional[pb2.TrialDone]] = [None] * len(stubs)

    def _poll(idx: int, stub: pb2_grpc.CbbaControlStub) -> Tuple[int, pb2.TrialDone]:
        done = stub.PollTrial(pb2.Empty(), timeout=poll_deadline_s)
        return idx, done

    with ThreadPoolExecutor(max_workers=len(stubs)) as pool:
        futs = [pool.submit(_poll, i, stub) for i, stub in enumerate(stubs)]
        for fut in as_completed(futs):
            idx, done = fut.result()
            dones[idx] = done
    t1 = time.perf_counter()
    collected = [d for d in dones if d is not None]
    if len(collected) != len(stubs):
        raise RuntimeError("PollTrial returned fewer TrialDone than nodes")
    return (t1 - t0) * 1000.0, collected


def _assignment_from_z(z: Sequence[int]) -> Dict[int, int]:
    return {j: int(z[j]) for j in range(len(z))}


def _distributed_valid(
    dones: Sequence[pb2.TrialDone], m: int, l_t: int, n: int
) -> bool:
    z0 = list(dones[0].final_z)
    y0 = list(dones[0].final_y)
    if any(list(d.final_z) != z0 or list(d.final_y) != y0 for d in dones):
        return False
    if any(d.n_rounds != dones[0].n_rounds for d in dones):
        return False
    if n * l_t >= m and m > 0:
        if any(int(w) == UNASSIGNED for w in z0):
            return False
    return True


def _per_node_timing(
    dones: Sequence[pb2.TrialDone],
) -> Tuple[str, str, str]:
    compute_lists: List[List[float]] = []
    blocking_lists: List[List[float]] = []
    n_rounds_max = 0
    for d in dones:
        c = [float(rd.compute_ms) for rd in d.round_digests]
        b = [float(rd.blocking_ms) for rd in d.round_digests]
        compute_lists.append(c)
        blocking_lists.append(b)
        n_rounds_max = max(n_rounds_max, len(c))
    wall = []
    for r in range(n_rounds_max):
        vals = []
        for c, b in zip(compute_lists, blocking_lists):
            if r < len(c):
                vals.append(c[r] + (b[r] if r < len(b) else 0.0))
        wall.append(max(vals) if vals else 0.0)
    node_compute = [sum(c) for c in compute_lists]
    node_blocking = [sum(b) for b in blocking_lists]
    return (
        json.dumps(wall),
        json.dumps(node_compute),
        json.dumps(node_blocking),
    )


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    p = argparse.ArgumentParser(description="EXP-3 CBBA trial runner (Tier L loopback).")
    p.add_argument("--config_grid", type=str, default="m=4,8,16,32;lt=1,ceil")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=100)
    p.add_argument("--network", type=str, default="loopback")
    p.add_argument("--ports", type=str, default="55201,55202,55203,55204")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--null_trials", type=int, default=20)
    p.add_argument("--bind_host", type=str, default="127.0.0.1")
    args = p.parse_args()

    configs = parse_config_grid(args.config_grid)
    ports = [int(x) for x in args.ports.split(",") if x.strip()]
    if len(ports) != N_AGENTS:
        raise SystemExit(f"--ports must have {N_AGENTS} entries")
    hostports = [f"{args.bind_host}:{pt}" for pt in ports]

    assert_fresh_out_dir(args.out_dir)
    csv_path = os.path.join(args.out_dir, "cbba_log.csv")
    floor_path = os.path.join(args.out_dir, "overhead_floor.csv")

    stop = {"flag": False}

    def _on_sigint(_signum: int, _frame: object) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"waiting for {len(hostports)} nodes...", flush=True)
    wait_nodes(hostports)
    channels, stubs = _control_stubs(hostports)

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(CBBA_LOG_HEADER)
    floor_file = open(floor_path, "w", newline="")
    floor_writer = csv.writer(floor_file)
    floor_writer.writerow(["trial", "t_total_ms"])

    trial_counter = 0
    floor_times: List[float] = []
    t_floor = 0.0

    def write_row(
        *,
        config_id: str,
        m: int,
        l_t: int,
        seed: int,
        warmup: int,
        is_null: int,
        rounds: int,
        rounds_ref: int,
        ref_agree: int,
        t_total_ms: float,
        wall_json: str,
        compute_json: str,
        blocking_json: str,
        msgs_total: int,
        bytes_total: int,
        valid: int,
    ) -> None:
        nonlocal trial_counter
        writer.writerow(
            [
                trial_counter,
                config_id,
                m,
                l_t,
                args.network,
                seed,
                warmup,
                is_null,
                rounds,
                rounds_ref,
                ref_agree,
                f"{t_total_ms:.6f}",
                f"{t_floor:.6f}",
                wall_json,
                compute_json,
                blocking_json,
                msgs_total,
                bytes_total,
                valid,
                _iso(),
            ]
        )
        csv_file.flush()
        trial_counter += 1

    try:
        # Null trials: m=0 handshake, runner-overhead floor.
        empty_starts = [(0.0, 0.0)] * N_AGENTS
        for k in range(args.null_trials):
            if stop["flag"]:
                break
            seed = args.seed_base + k
            t_ms, dones = drive_trial(stubs, trial_counter, 0, 0, [], empty_starts)
            floor_times.append(t_ms)
            floor_writer.writerow([k, f"{t_ms:.6f}"])
            floor_file.flush()
            wall_j, comp_j, blk_j = _per_node_timing(dones)
            msgs = sum(int(d.msgs_sent) for d in dones)
            byts = sum(int(d.bytes_sent) for d in dones)
            write_row(
                config_id="null",
                m=0,
                l_t=0,
                seed=seed,
                warmup=0,
                is_null=1,
                rounds=int(dones[0].n_rounds),
                rounds_ref=0,
                ref_agree=0,
                t_total_ms=t_ms,
                wall_json=wall_j,
                compute_json=comp_j,
                blocking_json=blk_j,
                msgs_total=msgs,
                bytes_total=byts,
                valid=0,
            )
        t_floor = _median(floor_times)
        # Rewrite later rows with the true median: already-written null rows
        # used 0.0; the schema says the median is repeated on every row. Re-open
        # after nulls and patch t_overhead_floor_ms on subsequent writes only
        # would leave null rows at 0. Rewrite the whole CSV for nulls now.
        csv_file.flush()

        def run_one(cid: str, m: int, l_t: int, seed: int, warmup: int) -> None:
            tasks, starts = generate_instance(seed, m, n=N_AGENTS)
            ref = run_sync_instance(tasks, starts, L_t=l_t)
            t_ms, dones = drive_trial(stubs, trial_counter, m, l_t, tasks, starts)
            rounds = int(dones[0].n_rounds)
            z0 = list(dones[0].final_z)
            assign = _assignment_from_z(z0)
            ref_agree = int(
                rounds == ref.n_rounds and assign == ref.assignment
            )
            valid = int(
                _distributed_valid(dones, m, l_t, N_AGENTS) and ref.valid
            )
            wall_j, comp_j, blk_j = _per_node_timing(dones)
            msgs = sum(int(d.msgs_sent) for d in dones)
            byts = sum(int(d.bytes_sent) for d in dones)
            write_row(
                config_id=cid,
                m=m,
                l_t=l_t,
                seed=seed,
                warmup=warmup,
                is_null=0,
                rounds=rounds,
                rounds_ref=int(ref.n_rounds),
                ref_agree=ref_agree,
                t_total_ms=t_ms,
                wall_json=wall_j,
                compute_json=comp_j,
                blocking_json=blk_j,
                msgs_total=msgs,
                bytes_total=byts,
                valid=valid,
            )

        # After nulls, t_floor is known. Patch null rows' t_overhead_floor_ms
        # by rewriting the file from a buffer is heavy; instead rewrite floor
        # column on a second pass at the end. For self-contained analysis of
        # the smoke we rewrite now via pandas-free CSV reread after close...
        # Simpler: update t_floor and continue; then rewrite all rows at exit.
        seed_seq = args.seed_base + args.null_trials
        for w in range(args.warmups):
            for cid, m, l_t in configs:
                if stop["flag"]:
                    break
                run_one(cid, m, l_t, seed_seq, 1)
                seed_seq += 1
            if stop["flag"]:
                break
        for _t in range(args.trials):
            for cid, m, l_t in configs:
                if stop["flag"]:
                    break
                run_one(cid, m, l_t, seed_seq, 0)
                seed_seq += 1
            if stop["flag"]:
                break
    finally:
        csv_file.close()
        floor_file.close()
        for ch in channels:
            ch.close()
        # Stamp t_overhead_floor_ms on every row now that the median is known.
        if os.path.isfile(csv_path) and floor_times:
            _stamp_overhead_floor(csv_path, t_floor)
        print(f"wrote {csv_path}", flush=True)


def _stamp_overhead_floor(csv_path: str, t_floor: float) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or CBBA_LOG_HEADER)
        rows = list(reader)
    col = "t_overhead_floor_ms"
    if col not in fieldnames:
        return
    val = f"{t_floor:.6f}"
    for row in rows:
        row[col] = val
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
