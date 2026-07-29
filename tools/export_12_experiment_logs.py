#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "12_experiments_full_logs.txt"

BENCHMARKS = ["b1", "b2", "b3", "b4"]
METHODS = [
    ("darts", "darts_log.csv", "config.json"),
    ("cea", "cea_edf_log.csv", "cea_edf_config.json"),
    ("sadcher", "sadcher_log.csv", "sadcher_config.json"),
]


def read_text(path: Path) -> str:
    if not path.exists():
        return f"[MISSING] {path}\n"
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    sections = []
    sections.append("# Full logs for 12 medium-stress experiments (B1-B4 x DARTS/CEA/SADCHER)\n")
    for b in BENCHMARKS:
        for method, log_name, cfg_name in METHODS:
            run_dir = RESULTS / f"{b}_medium_{method}_1000"
            log_path = run_dir / log_name
            cfg_path = run_dir / cfg_name
            sections.append("\n" + "=" * 120)
            sections.append(f"\n# {run_dir}\n")
            sections.append("-" * 40 + " CONFIG " + "-" * 40 + "\n")
            sections.append(read_text(cfg_path))
            if not sections[-1].endswith("\n"):
                sections.append("\n")
            sections.append("-" * 41 + " LOG " + "-" * 42 + "\n")
            sections.append(read_text(log_path))
            if not sections[-1].endswith("\n"):
                sections.append("\n")
    OUT.write_text("".join(sections), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
