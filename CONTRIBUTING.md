# Contributing to DARTS-Simulation

## Test fixtures and dry-run data

When writing or running tests that need synthetic input data
(e.g. for visualization scripts, runner dry-runs, or unit tests),
**fixtures MUST be created outside `results/`**. Use:

  - `/tmp/<test_name>/` for ephemeral fixtures (auto-cleaned by the OS)
  - `tests/fixtures/` for fixtures that should be version-controlled

**Cleanup of test fixtures MUST use absolute paths to the fixture
location**, never wildcards or globs that could match production
sweep output.

This rule exists because `results/` may contain expensive Jetson
sweep output (typically ~25 minutes per cell × hundreds of cells).
A single misplaced `rm -rf results/exp_{1,2,3}` once collaterally
deleted layered B1 production copies that had to be reconstructed
from flat-name backups; the root cause was a fixture written
inside `results/` and a too-broad cleanup glob.

## Sweep matrix changes

The sweep matrix (methods × benchmarks × stress × seeds) is
configured in `run_all_experiments_{light,medium,heavy}.py`. Any
expansion of the matrix (new method, new benchmark, additional
seed range) should be tested with `--dry_run` before launching
the full sweep.

## Code review for runner edits

The runners orchestrate multi-day Jetson sweeps via `--resume`,
which decides whether a cell is "done" by checking the layered
log destination at `_result_log_dst(...)`. Edits that change
either the layered path formula or the resume check must be
tested end-to-end on a 2-cell synthetic matrix before merging.

## Long-running operations and AI assistants

When using AI coding assistants to execute production runs,
**never silently substitute a smaller hyperparameter** (e.g.
reducing `--num_episodes` from 20000 to 500) just to fit within
a session timeout. A cell run with a substituted hyperparameter
produces semantically different output that pollutes downstream
aggregation. Mark such steps as DEFERRED and provide manual run
instructions (typically tmux + detach) instead.
