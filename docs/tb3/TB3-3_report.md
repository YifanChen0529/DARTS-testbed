# TB3-3 — S1 per-node travel eccentricity + soundness lemma

Conclusion first: S1 is live in the harness registry. On every registered
DBench-4 cell, per-node corner eccentricity is at most S0's workspace-diameter
tau, so S1 is uniformly tighter (B4 is a sanity anchor: S1 = S0 = 74810.0695).
At beta=1.0, **B2_LML:medium is rejected by S0 (255513.275 > 240000) and
accepted by S1 (236519.251 <= 240000)**. S0's `anchor` stdout is byte-identical
to the pre-P3 capture. Off the registered geometries the two tests are
incomparable because S0 fixes the start set to {(0,0)} while deployment starts
are uniform over W; a B4 relocation witness has S0 tau = 43241.6 ms and S1
tau = 61068.3 ms. Paper sources (`draft.tex` / `supplementary.tex`) are not in
this repo, so the `\hat\tau_v` symbol check is deferred to the writing phase.
TB3-4 is not started.

Base HEAD was `dd4bf71` (`tb3-stack-p2`).

## S1 table (declared xi unless noted)

R to 4 decimals; ratios vs archived darts means
109677.8 / 103381.5 / 100540.4 / 94403.1 / 121692.9 / 57059.4.

| cell | xi | R_G_off | path | beta*_e2e | beta*_rel | ratio |
|---|---|---|---|---|---|---|
| B1_CDR:light | declared | 207936.4752 | A→B→D | 1.0397 | 1.3862 | 1.90 |
| B1_CDR:medium | declared | 207936.4752 | A→B→D | 1.3862 | 1.3862 | 2.01 |
| B1_CDR:heavy | declared | 207936.4752 | A→B→D | 1.7328 | 1.3862 | 2.07 |
| B2_LML:medium | declared | 236519.2510 | Pk→Hf→Dr→Cf | 0.9855 | 1.3913 | 2.51 |
| B3_IQI:medium | declared | 234546.1207 | A→B→C→D | 1.3030 | 1.5636 | 1.93 |
| B3_IQI:medium | runtime | 269546.1207 | A→B→C→D | 1.4975 | 1.7970 | 2.21 |
| B4_DEM:medium | declared | 74810.0695 | J1 | 0.7481 | 0.7481 | 1.31 |

B1 path A→B→D is the first-max tie with A→C→D. B1 node envelopes:
A 49251.2173 + B 81750.0695 + D 76935.1885. B4 matches S0 because every
location has corner eccentricity 12.0208152802 m (same as S0's diameter).

S0 on the same cells is unchanged: B1 240250.2084, B2=B3 255513.2753,
B4 74810.0695.

## B2 headline

`eval --cell B2_LML:medium --beta 1.0 --tests S0,S1`:

```
nominal 1 S0 declared 255513.275291  240000.0  0  (reject e2e)
nominal 1 S1 declared 236519.251022  240000.0  1  (accept e2e)
```

S1 still rejects the relative deadline (236519.251 > 170000; beta*_rel = 1.3913).

## Incomparability witness

`replace(B4_DEM, all four nodes at (9.5, 0.5))`, B4/medium deadlines.

| | geometry | tau_ms | R_G_off |
|---|---|---|---|
| S0 | diam(locs ∪ {(0,0)}) = sqrt(90.5) = 9.5131 m | 43241.6 | 63411.6 |
| S1 | to corner (0,10), sqrt(180.5) = 13.4350 m | 61068.3 | 81238.3 |

S1 R > S0 R here. This is not S1 looseness: S0's published start set is the
parking corner (0,0), while episode starts are uniform over obstacle-free W, so
S0 under-covers the first-leg worst case on this placement (T0 §6). Never
assert S1 ≤ S0 globally. On the four registered geometries, `hat tau_v ≤ tau_bar`
for every node.

## Informational S0-W (not a registered test)

`--show-s0w` prints: tau = diam(W)/u = 14.1421356/0.22 s/node = 64282.4347
ms/node; B1 declared R_G_off = **269177.3 ms**. Whether this line enters the
paper is a writing-phase decision.

## Symbol check

`grep` for `\hat{\tau}` / `\hat\tau` over `draft.tex` and `supplementary.tex`:
**no paper TeX sources in this repo or `/home/yifan`**. Lemma uses
`\hat\tau_v` provisionally (`docs/tb3/S1_lemma.tex`). Deferred to the writing
phase.

## File list

| file | lines | role |
|---|---|---|
| `scripts/offline_test_harness.py` | 715 | S1, `--show-s0w`, default tests `S0,S1` |
| `scripts/csv_schema_tb.py` | +2 docstring lines | `tau_bar_ms` = max_v for per-node tests |
| `tests/test_tb3_s1.py` | 210 | s1a–s1f |
| `tests/test_tb3_s0_port.py` | t7 only | S1 stub assertion removed (S1 is live) |
| `docs/tb3/S1_lemma.tex` | 36 | paper-facing lemma + remark |
| `docs/tb3/TB3-3_report.md` | 154 | this report |

`anchor` was not edited. Frozen list vs `dd4bf71` is empty except the
csv_schema docstring addition and the harness.

## Commands run

```
# captured BEFORE edits
python3 scripts/offline_test_harness.py anchor ... \
  > results_tb3_offline/anchor_pre_p3.txt

python3 -m pytest -q tests/
# 72 passed (66 at p2 + 6 in test_tb3_s1.py)

python3 scripts/offline_test_harness.py frontier --tests S0,S1 --xi-source declared
python3 scripts/offline_test_harness.py frontier --show-s0w --tests S1
python3 scripts/offline_test_harness.py eval --cell B2_LML:medium --beta 1.0 --tests S0,S1

python3 scripts/offline_test_harness.py anchor ... \
  > results_tb3_offline/anchor_post_p3.txt
diff -u results_tb3_offline/anchor_pre_p3.txt results_tb3_offline/anchor_post_p3.txt
# empty
```

No simulation runs.

## `git status --porcelain`

### Before

```
 M README.md
 M scripts/csv_schema_tb.py
 M scripts/offline_test_harness.py
 M tests/test_tb3_s0_port.py
?? docs/tb3/S1_lemma.tex
?? tests/test_tb3_s1.py
?? results_tb3_offline/
… plus pre-existing dirty/untracked leftovers
```

Staged only this prompt's files (including the t7 update so the suite stays
green). `results_tb3_offline/` not committed.

### After

TB3-3 files leave porcelain. Pre-existing dirty tree unchanged.

## Commit and tag

- message: `TB-3 P3: S1 per-node travel eccentricity refinement + soundness lemma`
- tag: `tb3-stack-p3`
- hash: see `git rev-parse tb3-stack-p3`

## Carried forward

(i) TB3-4 grid decision now has concrete anchors: S1 B1/heavy beta*_e2e =
1.7328 sits in the 1.6–2.0 gap (S1 accepts at 2.0; S0 still rejects at 2.0,
beta*_e2e = 2.0021, so S0 still needs points > 2.0). Spec "13 points" vs the
P0 12-point literal is unresolved.

(ii) Whether the start-set-corrected S0 line (269177.3) enters the paper is a
writing-phase decision for Yifan/advisor.

(iii) TB3-5 `sweep_tb1.py` passthrough for `--instance_seed` is still pending.

Stop. Do not start TB3-4.
