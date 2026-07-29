#!/usr/bin/env python3
"""Heartbeat monitor for DARTS sweeps (sentinels + optional log-dir stall detection)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

DEFAULT_PANES = ["light", "medium", "heavy", "energy", "agent"]
CHECK_EVERY_SEC = 300
STALL_ALERT_COOLDOWN_SEC = 3600


def notify(title: str, message: str, priority: str = "default") -> None:
    try:
        subprocess.run(
            ["python3", "scripts/notify.py", "--title", title, "--message", message, "--priority", priority],
            timeout=15,
        )
    except Exception as e:
        print(f"[monitor] notify fail: {e}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep pane monitor (sentinels + stall alerts)")
    p.add_argument("--sentinel_dir", default=None, help="Directory with PANE_DONE_<name> files")
    p.add_argument("--log_dir", default=None, help="Watch *.log mtime here for stall detection")
    p.add_argument("--results_dir", default="results_a1a2b1_20260521", help="Aggregated results tree")
    p.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated pane names (sentinel basenames), e.g. noinc_b1_light,...",
    )
    p.add_argument("--stall_minutes", type=int, default=60)
    p.add_argument("positional", nargs="*", help="Legacy: sentinel_dir [results_dir]")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> tuple[str, str, list[str], str | None, int]:
    if args.positional and not args.positional[0].startswith("-"):
        sentinel_dir = args.sentinel_dir or args.positional[0]
        results_dir = args.results_dir if len(args.positional) < 2 else args.positional[1]
        expected_panes = (
            [s.strip() for s in args.sessions.split(",") if s.strip()]
            if args.sessions
            else DEFAULT_PANES
        )
        log_dir = args.log_dir
        stall_minutes = args.stall_minutes
        return sentinel_dir, results_dir, expected_panes, log_dir, stall_minutes

    sentinel_dir = args.sentinel_dir or "sweep_sentinels"
    results_dir = args.results_dir
    if not args.sessions:
        sys.stderr.write("[monitor] FATAL: --sessions required (comma-separated pane names)\n")
        sys.exit(1)
    expected_panes = [s.strip() for s in args.sessions.split(",") if s.strip()]
    return sentinel_dir, results_dir, expected_panes, args.log_dir, args.stall_minutes


def newest_age_min(paths: list[str], now: float) -> float:
    if not paths:
        return float("inf")
    newest = max(os.path.getmtime(f) for f in paths)
    return (now - newest) / 60


def main() -> None:
    args = parse_args()
    sentinel_dir, sweep_dir, expected_panes, log_dir, stall_minutes = resolve_config(args)
    os.makedirs(sentinel_dir, exist_ok=True)
    last_stall_alert = 0.0
    notify(
        "DARTS sweep: monitor started",
        f"Watching sentinels={sentinel_dir}, results={sweep_dir}, "
        f"panes={expected_panes}. Stall threshold: {stall_minutes}min.",
        priority="low",
    )
    while True:
        now = time.time()
        done_panes: dict[str, str] = {}
        for pane in expected_panes:
            f = os.path.join(sentinel_dir, f"PANE_DONE_{pane}")
            if os.path.exists(f):
                with open(f) as fh:
                    done_panes[pane] = fh.read().strip()
        agg_cells = len(glob.glob(f"{sweep_dir}/exp_*/B*/*/*_log.csv"))
        if log_dir:
            stall_paths = glob.glob(os.path.join(log_dir, "*.log"))
            newest_age_min_val = newest_age_min(stall_paths, now)
            stall_label = f"log_dir={log_dir}"
        else:
            ws_csvs = glob.glob("results/*/*_log.csv")
            stall_paths = ws_csvs
            newest_age_min_val = newest_age_min(ws_csvs, now)
            stall_label = "workspace_csv"
        print(
            f"[monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"done_panes={len(done_panes)}/{len(expected_panes)} "
            f"agg_cells={agg_cells} "
            f"newest_{stall_label}_age_min={newest_age_min_val:.1f}",
            flush=True,
        )
        if len(done_panes) == len(expected_panes):
            ok_count = sum(1 for rc in done_panes.values() if rc == "0")
            failed = {p: rc for p, rc in done_panes.items() if rc != "0"}
            title = f"DARTS sweep: ALL DONE ({ok_count}/{len(expected_panes)} ok)"
            msg_lines = [
                f"Aggregated cells: {agg_cells}",
                f"Pane statuses: {json.dumps(done_panes)}",
            ]
            if failed:
                msg_lines.append(f"FAILED panes: {json.dumps(failed)}")
            notify(title, "\n".join(msg_lines), priority="high" if failed else "default")
            print("[monitor] all done, exiting", flush=True)
            return
        if (
            newest_age_min_val > stall_minutes
            and (now - last_stall_alert) > STALL_ALERT_COOLDOWN_SEC
            and stall_paths
        ):
            notify(
                f"DARTS sweep: STALL ({newest_age_min_val:.0f}min no activity)",
                f"No {stall_label} update in {newest_age_min_val:.0f}min. "
                f"agg_cells={agg_cells}, done_panes={list(done_panes)}. SSH in to investigate.",
                priority="urgent",
            )
            last_stall_alert = now
        time.sleep(CHECK_EVERY_SEC)


if __name__ == "__main__":
    main()
