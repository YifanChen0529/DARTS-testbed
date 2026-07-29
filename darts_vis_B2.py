#!/usr/bin/env python3
"""Delegate to alg/darts_vis_B2.py (canonical implementation)."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "alg" / "darts_vis_B2.py"), run_name="__main__")
