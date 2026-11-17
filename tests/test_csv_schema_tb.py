"""Schema guards for the TB era: 76-column main log and the subround sidecar.

``scripts/csv_schema_tb.py`` deliberately keeps a *copy* of the main header rather than
importing it, so that analysis code can pin the expected columns without importing torch.
That copy is only trustworthy if something asserts the two agree -- that is this file.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from scripts.csv_schema_tb import (
    MAIN_CSV_HEADER_TB76,
    MAIN_CSV_N_COLS_AE63,
    MAIN_CSV_N_COLS_TB76,
    SUBROUND_LOG_HEADER,
    TB_APPENDED_COLUMNS,
)

AE63_TAIL = (
    "n_miss_n1_drop",
    "n_miss_n2_fallback_insufficient",
    "n_miss_n3_unreached",
    "n_miss_full_fidelity_late",
)

RETIRED_62_COL_NAMES = ("cert_attempts", "cert_rejects", "cert_reject_rate")


def test_main_header_widths():
    assert len(MAIN_CSV_HEADER_TB76) == MAIN_CSV_N_COLS_TB76 == 76
    assert len(TB_APPENDED_COLUMNS) == 13


def test_ae63_prefix_is_untouched():
    """The 63 AE columns must keep their identity and order; TB columns are appended only."""
    assert tuple(MAIN_CSV_HEADER_TB76[MAIN_CSV_N_COLS_AE63 - 4:MAIN_CSV_N_COLS_AE63]) == AE63_TAIL


def test_tb_columns_in_specified_order():
    assert TB_APPENDED_COLUMNS == [
        "n_bids_evaluated",
        "n_bids_cert_pass",
        "mean_pool_size",
        "n_ff_missed",
        "episode_span_ms",
        "exec_ms_A1", "exec_ms_A2", "exec_ms_A3", "exec_ms_A4",
        "travel_ms_A1", "travel_ms_A2", "travel_ms_A3", "travel_ms_A4",
    ]


def test_retired_62_col_names_not_reused():
    """Those names carried a different definition; reusing them would corrupt
    header-keyed cross-era analysis."""
    for name in RETIRED_62_COL_NAMES:
        assert name not in MAIN_CSV_HEADER_TB76


def test_no_duplicate_columns():
    assert len(set(MAIN_CSV_HEADER_TB76)) == len(MAIN_CSV_HEADER_TB76)
    assert len(set(SUBROUND_LOG_HEADER)) == len(SUBROUND_LOG_HEADER)


def test_subround_header_is_canonical():
    assert SUBROUND_LOG_HEADER == [
        "episode", "phase", "clock_ms", "job_id", "subround_pos_j", "A_k_size",
        "n_alive", "C_v_size", "t_subround_ms", "t_cert_ms", "t_infer_ms",
        "t_exchange_ms", "t_fallback_ms", "t_commit_ms", "queue_len_per_agent",
        "winner_idx", "outcome",
    ]


def test_registry_copy_matches_train_darts():
    pytest.importorskip("torch", reason="alg.train_darts requires torch")
    from alg.train_darts import EVAL_CSV_HEADER

    assert list(EVAL_CSV_HEADER) == MAIN_CSV_HEADER_TB76
