#!/usr/bin/env python3
"""Lenient CSV reader for DARTS sweep logs.

train_darts.py and the baselines write per-episode rows + occasional
eval-checkpoint rows to the same _log.csv. Column counts may differ
across row types, which crashes pd.read_csv with default settings.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
from collections import Counter
import sys

import pandas as pd


def safe_read_csv(path: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV tolerantly. Skip rows whose field count differs from the header.

    Returns None on unreadable / missing file. Always passes engine='python'
    and on_bad_lines='skip'. Caller-supplied kwargs override.
    """
    path = Path(path)
    if not path.exists():
        return None
    base_kwargs = dict(engine='python', on_bad_lines='skip')
    base_kwargs.update(kwargs)
    try:
        df = pd.read_csv(path, **base_kwargs)
        return df if not df.empty else None
    except Exception as e:
        sys.stderr.write(f"[safe_read_csv] {path}: {e}\n")
        return None


def diagnose_field_counts(path: Union[str, Path]) -> dict:
    """Count field-count distribution across all lines of a CSV.

    Useful for one-time inspection of writer-side schema drift.
    Returns: {n_fields: line_count, ...}
    """
    path = Path(path)
    if not path.exists():
        return {}
    counter: Counter = Counter()
    with open(path) as fh:
        for line in fh:
            stripped = line.rstrip('\n').rstrip('\r')
            if not stripped:
                continue
            n = stripped.count(',') + 1
            counter[n] += 1
    return dict(counter)
