# Signed community detection — numerical-range fix verification

All commands below assume `PYTHONPATH=/workspace` (or an editable
install of this checkout).  The environment used Python 3.11 with no
third-party packages; a small pytest-compatible shim lives in
`/workspace/work/pytest_shim` and the runner in
`/workspace/work/run_tests.py`.

## 1. Exact reproductions from the report

```
python3 -I /workspace/work/repro.py     # four-node graph, all four scales
python3 -I /workspace/work/repro2.py    # overflow errors + greedy regression
```

Observed after the fix (before the fix they showed 0.8,
ZeroDivisionError, four singletons, ValueError and the 0.2296 greedy
partition):

| scale | public score | Louvain(seed=19) | greedy |
|---|---|---|---|
| 1.0    | 0.5 | [{0,1},{2,3}] | [{0,1},{2,3}] |
| 1e-165 | 0.5 | [{0,1},{2,3}] | [{0,1},{2,3}] |
| 1e-310 | 0.5 | [{0,1},{2,3}] | [{0,1},{2,3}] |
| 1e155  | 0.5 | [{0,1},{2,3}] | [{0,1},{2,3}] |

The six-node greedy graph returns `[{0,2,3,5},{1},{4}]` (score
0.2585185185) at both scale 1.0 and 1e155.

## 2. Test suites

```
cat /workspace/work/modlist.txt | xargs python3 -I /workspace/work/run_tests.py
```

Result: **271 passed, 0 failed, 1 skipped** (the skip is the pre-existing
scipy-optional test in `test_divisive.py`).  The breakdown includes:

* `test_signed.py`: 65 passed (original signed suite, updated for the
  new private helper signatures)
* `test_signed_numeric_range.py`: 106 passed (new range/gain tests with
  the independent normalization-aware reference)
* `test_quality.py`, `test_louvain.py`, `test_modularity_max.py`:
  7 + 14 + 19 passed (unsigned/signed=False behaviour unchanged)
* all remaining community modules: pass

Doctests of the four touched modules also pass
(`doctest.testmod` based, to avoid the dispatch-registry conflict of
the file-based runner).

## 3. Extreme-value smoke checks

```
python3 -I /workspace/work/edge_cases.py
```

Covers: the smallest subnormal edge (5e-324 -> score -0.5), the largest
finite edge (~1.8e308), a 1e290 layer-mass gap, explicit
`ValueError` when the accumulated raw mass overflows, 1e300
multi-graphs with self-loops, resolution variants, `weight=None` /
missing attributes, and rejection of NaN/inf/bool/non-real weights.

## 4. Before/after cost experiment (100 and 200 nodes, degree ~8)

```
python3 -I /workspace/work/experiment.py /workspace/work/before_tree BEFORE
python3 -I /workspace/work/experiment.py /workspace AFTER
```

Graph construction and reference scoring are outside the measurements;
each optimizer runs three timed times, one `tracemalloc` pass, and the
`_EVAL_STATS` counters are recorded.

### n=100, m=390

| optimizer | version | 3 run times (s) | peak (MB) | gain evaluations | \|C\| | score |
|---|---|---|---|---|---|---|
| louvain | before | 0.0035, 0.0031, 0.0031 | 0.263 | 17757 | 9 | 0.34301 |
| louvain | after  | 0.0037, 0.0032, 0.0032 | 0.268 | 17757 | 9 | 0.34301 |
| greedy  | before | 0.0064, 0.0063, 0.0062 | 0.292 | 5246  | 7 | 0.34176 |
| greedy  | after  | 0.0067, 0.0068, 0.0066 | 0.292 | 5246  | 7 | 0.34176 |

### n=200, m=787

| optimizer | version | 3 run times (s) | peak (MB) | gain evaluations | \|C\| | score |
|---|---|---|---|---|---|---|
| louvain | before | 0.0087, 0.0089, 0.0089 | 0.558 | 74974 | 9  | 0.34012 |
| louvain | after  | 0.0090, 0.0089, 0.0092 | 0.570 | 74974 | 9  | 0.34012 |
| greedy  | before | 0.0180, 0.0178, 0.0180 | 0.610 | 21915 | 11 | 0.34170 |
| greedy  | after  | 0.0185, 0.0184, 0.0182 | 0.610 | 21915 | 11 | 0.34170 |

The fix adds one O(m) exponent scan per optimizer entry; wall-time and
peak memory are within run-to-run noise, and the candidate/gain
evaluation workload, the chosen partitions and the scores are
identical at normal scale.

On the same two graphs, scaling every edge by any of
1e-310 / 1e-165 / 1e155 / 1e300 leaves both optimizers' partitions
bit-for-bit identical and the score unchanged.

## 5. Design choice (recorded for review)

Two organizations were considered:

* **chosen — global exact power-of-two normalization**: one scan finds
  the maximum `frexp` exponent; every weight, mass, degree, compressed
  edge and gain is maintained in the resulting normalized space where
  the largest magnitude is in (0.5, 1].  `ldexp` never rounds.
* alternative — rewriting each formula into ratio form without
  normalizing.  Products of two subnormal weights still underflow, the
  1e-12 threshold loses its meaning across scales, and every formula
  needs separate auditing.

The normalized space keeps the existing formulas, sparse structures,
per-row heaps and single-entry-per-community non-adjacent cache
unchanged; no all-node-pair or all-community-pair gain table is kept,
and no candidate evaluation copies the graph or recomputes the full
modularity.

## 6. Not verified here

* No real pytest binary exists in this environment; the included shim
  implements the subset of pytest used by the community tests.  The new
  test file uses only standard pytest features (`mark.parametrize`,
  `raises`, `approx`, plain classes/functions) and should run unchanged
  under upstream pytest.
* No backend (numpy/scipy/graphblas) was available; only the default
  Python backend was exercised.
* The single-level `louvain_partitions` snapshot exposure on a graph
  that yields exactly one level is pre-existing, shared with the
  unsigned path, and unrelated to this fix; snapshot protection for
  multi-level runs is covered by the new tests.
