"""Static tests for the EXP-3 P2 loopback stack (no network)."""

from __future__ import annotations

import hashlib
import os
import re
from typing import List

from benchmarks.cbba_microbench import cbba_pb2
from scripts.csv_schema_tb import (
    CBBA_LOG_HEADER,
    MAIN_CSV_HEADER_TB76,
    SUBROUND_LOG_HEADER,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_launcher_console_log_is_outside_out_dir():
    path = os.path.join(_PROJECT_ROOT, "benchmarks", "cbba_microbench", "run_loopback.sh")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"^CONSOLE_LOG=(.+)$", text, re.M)
    assert m, "CONSOLE_LOG assignment missing in run_loopback.sh"
    rhs = m.group(1).strip()
    if len(rhs) >= 2 and rhs[0] == rhs[-1] and rhs[0] in "\"'":
        rhs = rhs[1:-1]
    assert not rhs.startswith("$OUT_DIR/"), rhs
    assert not rhs.startswith("${OUT_DIR}/"), rhs
    assert "${OUT_DIR%/}_console.log" in rhs or rhs.endswith("_console.log")


# SHA-256 digests of the two schema headers that predate the CBBA microbenchmark, taken
# over the column names joined by line feeds. New headers may be appended to the schema
# module; these two must stay exactly as they were when the microbenchmark was added.
P1_HEADER_SHA256 = {
    "SUBROUND_LOG_HEADER": "9efab1ddd8d7894d46fc288d72a21c3e2c02891bdb28161457335be22c977589",
    "MAIN_CSV_HEADER_TB76": "292305d0b6684dacb2ada28234c719d535e763319ee8fb7d7080f60723dcd1c3",
}


def _header_sha256(header: List[str]) -> str:
    return hashlib.sha256("\n".join(header).encode("utf-8")).hexdigest()


def test_cbba_log_header_schema_and_p1_headers_untouched():
    expected = [
        "trial", "config_id", "m", "l_t", "network", "seed", "warmup", "null",
        "rounds", "rounds_ref", "ref_agree", "t_total_ms", "t_overhead_floor_ms",
        "per_round_wall_ms_json", "per_node_compute_ms_json",
        "per_node_blocking_ms_json", "msgs_total", "bytes_total", "valid", "ts_iso",
    ]
    assert CBBA_LOG_HEADER == expected
    assert len(CBBA_LOG_HEADER) == 20
    assert _header_sha256(SUBROUND_LOG_HEADER) == P1_HEADER_SHA256["SUBROUND_LOG_HEADER"]
    assert _header_sha256(MAIN_CSV_HEADER_TB76) == P1_HEADER_SHA256["MAIN_CSV_HEADER_TB76"]


def test_proto_agent_lists_roundtrip():
    msg = cbba_pb2.AgentLists(
        sender_id=1,
        round=3,
        z=[0, -1, 2],
        y=[1.5, 0.0, 9.25],
        s=[0, 3, 3, 2],
        send_ns=123456789,
    )
    blob = msg.SerializeToString()
    back = cbba_pb2.AgentLists()
    back.ParseFromString(blob)
    assert back.sender_id == 1
    assert back.round == 3
    assert list(back.z) == [0, -1, 2]
    assert list(back.y) == [1.5, 0.0, 9.25]
    assert list(back.s) == [0, 3, 3, 2]
    assert back.send_ns == 123456789


def test_gates_file_pre_registered_strings():
    path = os.path.join(_PROJECT_ROOT, "docs", "exp3", "exp3_gates.md")
    text = open(path, encoding="utf-8").read()
    for token in ("Tier L", "ref_agree", "l_t * m", "advisor sign-off"):
        assert token in text, token
