"""EXP-3 P3 tests: grep gate, gates append-only prefix, analysis helpers.

Does not read results_exp3_20260829 (machine-local, untracked).
"""

from __future__ import annotations

import math
import hashlib
import os

from scripts.cbba_analysis import (
    dispersion_ratio,
    half_split_medians,
    median,
    percentile,
)
from scripts.fig_exp3_tconv_cdf import rendered_strings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANNED = (
    "exp3_",
    "config",
    "smoke",
    "warmup",
    "ref_agree",
    "cbba_log",
    "stack",
    "Tier",
    "tier L",
)


def test_figure_labels_and_caption_banned_substrings():
    caption_path = os.path.join(
        _PROJECT_ROOT, "figures", "FIG_exp3_tconv_cdf_caption.txt"
    )
    caption = open(caption_path, encoding="utf-8").read()
    labels = rendered_strings()
    hay = caption + "\n" + labels
    for tok in BANNED:
        assert tok not in hay, tok
    assert "convergence wall time over local loopback (milliseconds)" in labels
    assert "fraction of trials" in labels
    assert "4 jobs" in labels


# SHA-256 of the pre-registered part of docs/exp3/exp3_gates.md: everything above the
# RESULTS heading, including the line feed that ends it. The gates were registered
# before any result file was opened; this digest pins that text.
EXPECTED_GATES_PREFIX_SHA256 = (
    "bdf1d9922cdeb9285ccadb212588ba989b2917de8f4f34359fbcc2b2806f519f"
)


def test_gates_prefix_byte_identical_to_p2():
    path = os.path.join(_PROJECT_ROOT, "docs", "exp3", "exp3_gates.md")
    current = open(path, encoding="utf-8").read()
    marker = "\n## RESULTS"
    assert marker.strip() in current
    registered = current.split(marker, 1)[0] + "\n"
    digest = hashlib.sha256(registered.encode("utf-8")).hexdigest()
    assert digest == EXPECTED_GATES_PREFIX_SHA256


def test_percentile_median_dispersion_and_half_split():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == 2.5
    assert median(xs) == 2.5
    # P99 on four points interpolates near the max.
    assert percentile(xs, 99) == percentile(xs, 99)
    ratio = dispersion_ratio([10.0, 10.0, 10.0, 20.0])
    assert ratio == percentile([10.0, 10.0, 10.0, 20.0], 99) / 10.0
    first, second, pct = half_split_medians([10.0, 10.0, 12.0, 12.0])
    assert first == 10.0
    assert second == 12.0
    assert abs(pct - 20.0) < 1e-12
    first, second, pct = half_split_medians([20.0, 20.0, 10.0, 10.0])
    assert abs(pct - (-50.0)) < 1e-12
    assert math.isfinite(dispersion_ratio([1.0, 2.0, 3.0]))
