# EXP-3 pre-registered adjudication gates (CBBA convergence-latency microbenchmark)

Registered after the loopback smoke passed and **before** any full-session
result file was read. Do not edit the execution-tier decision or the Gates
section after this file is written. Analysis appends a RESULTS heading only.

## Execution-tier decision of record (Yifan, 2026-08-29)

Tier L, loopback-only. Evidence: the sole ICMP-reachable candidate
192.168.0.103 was proven to be this coordinator Jetson itself (interactive
ssh landed on hostname "ubuntu", hostname -I 192.168.251.46, repo HEAD
57187b7, uptime 123 days); agent hosts .100/.101/.102/.104 are unreachable
(No route). Consequences to encode in all artifacts of this prompt:
per-round compute and N_r distributions are MEASURED; the physical wireless
channel is NOT traversed, so any cross-machine network term is calibrated,
not measured; the former acceptance criterion "loopback floor well below
WLAN numbers" is downgraded to report-only (no WLAN session exists to
compare against).

TODO: wording of the network term in TR and paper sentence pends advisor sign-off, EXP-2 precedent.

## Full-session recipe (frozen)

- `trials=200` recorded per config; `warmups=5` excluded by analysis
- `seed_base=100`; instances from `cbba_core.generate_instance(seed_base + trial_index, m)`
- round-robin interleave across configs (one trial of each config in rotation)
- `null_trials=20` for the runner-overhead floor
- network tag `loopback`; four processes on `127.0.0.1:55201..55204`
- configs: `m ∈ {4,8,16,32}` × `lt ∈ {1, ceil(m/4)}` (eight slots; `m=4` has two
  slots both with `l_t=1`)

## Comparison anchors (analysis provenance; T0 R5)

- mean 9.6 ms per subround at frontier size 1: **paper supplementary source
  only, no repo artifact**
- 23.38 ms P99 of per-episode maximum auction time: `vg_analysis_output.md`
  line 8
- 30.0 ms calibrated `delta_auc` envelope: `vg_t3_combined.md` line 3

T0 R5 caveat, verbatim: 23.38 ms (`vg_analysis_output.md:8`) and 30.0 ms
(`vg_t3_combined.md:3`) are repo-internal; 9.6 ms mean-at-frontier-1 exists
only in paper sources.

## Gates

Recorded rows = non-null, `warmup=0`. Smoke rows are not in scope.

### G-V (validity, strict)

`valid == 1` on 100% of non-null recorded trials (any failure is an
implementation bug: fix, rerun the config).

### G-C (round cap, strict)

`rounds <= l_t * m` on every recorded trial.

### G-R (reference identity, strict)

`ref_agree == 1` on every non-null recorded trial (the distributed stack
must reproduce the unit-tested reference exactly).

### G-S (stability, informational, stated direction)

Per config, first-half vs second-half median `T_conv` (`t_total_ms`) within
~10%; else rerun with isolation noted. No pass/fail identity is registered.

### G-F (floors, report-only)

Overhead floor (`t_overhead_floor_ms`, median of this session's null trials)
and loopback compute floor reported alongside distributions. No threshold:
the WLAN comparison target does not exist at Tier L. The former criterion
"loopback floor well below WLAN numbers" is report-only.

## Report-only distribution statistics (no gates)

Per config, for `T_conv` (`t_total_ms`) and for `rounds`: mean, P50, P95,
P99, max, P99/median; and `rounds` vs the `l_t * m` cap. **NO** strict
identities over stochastic aggregates and **NO** cross-config orderings are
registered.

## RESULTS

G-V PASS: valid==1 on 1600/1600 recorded rows.
G-C PASS: rounds <= l_t*m on 1600/1600 recorded rows.
G-R PASS: ref_agree==1 on 1600/1600 recorded rows.
G-F report-only: null-trial overhead floor 16.2674 ms (n_null=20).

G-S informational (first-half vs second-half median t_total_ms):
- m4_lt1: 27.247 vs 26.421 ms (-3.03%).
- m4_ltceil: 25.732 vs 26.437 ms (2.74%).
- m8_lt1: 24.963 vs 24.902 ms (-0.25%).
- m8_ltceil: 31.286 vs 31.232 ms (-0.17%).
- m16_lt1: 20.767 vs 21.235 ms (2.25%).
- m16_ltceil: 38.079 vs 38.569 ms (1.29%).
- m32_lt1: 20.911 vs 20.706 ms (-0.98%).
- m32_ltceil: 68.581 vs 65.179 ms (-4.96%).

Recorded-row medians (t_total_ms / rounds P50):
- m4_lt1: t_p50=26.617 ms, rounds_p50=2.000, n=200.
- m4_ltceil: t_p50=26.074 ms, rounds_p50=2.000, n=200.
- m8_lt1: t_p50=24.963 ms, rounds_p50=2.000, n=200.
- m8_ltceil: t_p50=31.248 ms, rounds_p50=3.000, n=200.
- m16_lt1: t_p50=21.045 ms, rounds_p50=1.000, n=200.
- m16_ltceil: t_p50=38.437 ms, rounds_p50=4.000, n=200.
- m32_lt1: t_p50=20.790 ms, rounds_p50=1.000, n=200.
- m32_ltceil: t_p50=67.352 ms, rounds_p50=5.000, n=200.
