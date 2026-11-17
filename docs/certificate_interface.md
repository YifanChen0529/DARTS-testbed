# Certificate interface

The conclusion of the paper names "the documented certificate interface of the release"
as the place where tighter cost envelopes and less conservative offline analyses are
plugged in. This file is that documentation. It covers three places in the code: the
online certificate, the cost envelopes that the certificate reads, and the offline test.
References are to the camera-ready paper and to `supplementary.pdf`. Replacing an
offline test or rescaling an envelope does not change the auction (Algorithm 1 of the
paper).

## 1. Online certificate

`DartsMARLAgent.check_feasibility` in `alg/darts_agent.py` is the joint certificate: the
Sequential Feasibility Certificate together with the energy certificate. It is the
admission predicate of mechanism M2 in `DEVELOPMENT.md`.

    check_feasibility(new_job, current_time_ms, current_battery_wh=1.0,
                      dynamic_wcet_ms=None, release_time_ms=0.0, env=None) -> bool

What it computes:

- The candidate job is added to the committed queue of the agent, and the queue is
  sorted by absolute deadline.
- The cost of every queue position is recomputed along that order: a travel envelope
  from the location of the previous job (from the position of the agent for the first
  job) plus the remaining execution time, both from `env.estimate_chain_wcet_ms`, plus
  the overhead reservation `delta_max_ms` of the configuration. A fixed travel allowance
  replaces the envelope when the environment or the location of a job is not available.
- Temporal part: `_feasibility_test` in the same file accepts if every prefix of the
  queue completes by the deadline of its last job.
- Energy part: the battery must cover the worst-case energy of the queued jobs and of
  the candidate, plus the reserve `e_safe_wh` of the configuration.

The flag `--disable_feasibility_mask` of `alg/train_darts.py` makes the method return
`True` unconditionally. It exists for the NoCert ablation only.

The runtime reaches the certificate through the class attribute
`DartsMARLAgent.check_feasibility`, so a variant with the same signature can be bound in
its place without editing the method. `scaled_cert_envelopes` (Section 2) does this.
Tests: `tests/test_certificate.py` and `tests/test_cert_duplicate_fidelity.py`.

## 2. Cost envelopes

The certificate reads three envelopes: travel time, execution time, and energy. Travel
and execution come from `estimate_chain_wcet_ms` and `estimate_chain_wcet_components_ms`
of the environment class in `envs/turtlebot3_spatial_env.py`. Energy comes from
`get_wc_energy_wh` of the agent.

`scaled_cert_envelopes(kappa, exec_scale, travel_scale, energy_scale)` in
`alg/darts_agent.py` is a context manager that multiplies these envelopes inside the
certificate and nothing else. The overhead reservation, the epoch length, the
environment dynamics, the policy features, the fallback costs, and the energy reserve
are not scaled. While any factor differs from 1, the context manager binds
`check_feasibility_scaled`, a copy of the certificate that carries the factors. At the
default, the original method runs.

The factors are set with flags of `alg/train_darts.py`: `--cert_conservatism` scales
all three envelopes (the scale zeta of the paper), and `--cert_conservatism_exec`,
`--cert_conservatism_travel`, and `--cert_conservatism_energy` scale one envelope each.
The envelope-scaling experiment (Fig. 7; Supp. Sec. VIII-A) is driven by
`scripts/sweep_tb1.py`. A factor below 1 evaluates the certificate below its calibrated
envelopes; the guarantees of the paper are stated for the calibrated envelopes.

## 3. Offline test

The offline test decides before deployment whether a mission instance is certified for
its end-to-end deadline. The runtime does not call it. The paper identifies it as
replaceable and as the main source of conservatism (Section V-E; Supp. Sec. VIII).

Code: `scripts/offline_test_harness.py`. A test is a function

    test(inst: MissionInstance, env: Envelopes, xi_source: str) -> BoundResult

- `MissionInstance` (`alg/tb3_instance.py`) is the workload and configuration that the
  runtime would see for one mission. `xi_source` selects which execution bounds of the
  instance are used; the module docstring of the harness explains why there are two.
- `Envelopes` carries the platform constants of the test: subround bound, epoch length,
  readiness jitter, speed, communication bound, parking position, and map size.
- `BoundResult.R_G_off_ms` is the bound on the graph response time. The harness
  compares it with the end-to-end deadline of the instance after the deadline scale.

Two tests are provided:

| Test | Function | Reference |
|---|---|---|
| S0, the path-sum test | `s0_path_sum` | Eq. (23) of the paper; the same computation as `scripts/r_off_g.py` |
| S1, the per-node travel refinement | `s1_eccentricity` | Lemma 3 of the supplementary; `docs/tb3/S1_lemma.tex` |

To add a test:

1. Write a function with the signature above, next to `s1_eccentricity`.
2. Register it under a name of its own in the same way as S0 and S1 (search the file
   for `s1_eccentricity`).
3. Select it by that name with the option `--tests` of the subcommands `eval` and
   `grid`. `PYTHONPATH=. python scripts/offline_test_harness.py --help` lists the
   subcommands.

The conditions under which a test may replace the provided ones are stated in Remark 2
of the paper and in Supp. Sec. VIII-C. In short: the test must be sound, it must match
the release model and the scheduling semantics of the runtime, and its graph-level
bound must be monotone nondecreasing in the per-job bounds. The online certificate and
Algorithm 1 do not depend on which offline test is used.

How the two provided tests are checked in this repository:

- `tests/test_tb3_s0_port.py` and `tests/test_tb3_s1.py` are their unit tests.
- The empirical comparison of the paper (Supp. Sec. VIII-B) replays frozen policies on
  the same seeded instances (`scripts/sweep_tb3.py`, `scripts/tb3_empirical_collect.py`)
  and compares the outcomes with the verdicts of the offline tests. Its records are in
  `docs/tb3/`. The replay needs trained policy checkpoints, which are not part of the
  repository.

For periodic and sporadic releases, the offline test is the density test of
Theorem 2(a): `evaluate_thm2a` in `scripts/thm2a_offline_test.py`.
