"""Benchmark for signed-mode community detection.

Runs ``louvain_communities`` and ``greedy_modularity_communities`` with
``signed=True`` on fixed-seed sparse mixed-sign graphs (600 and 1200
nodes, average undirected degree ~12) and reports, for each optimizer:

* wall time (measured from after the graph is built),
* peak extra Python memory (``tracemalloc``),
* candidate/gain evaluation counts (via the private
  ``networkx.algorithms.community._signed._EVAL_STATS`` hook),
* the signed modularity of the result (public scorer, timed separately).

Graph construction and the reference scoring are timed separately from
the optimizers.  Memory figures are the *additional* traced Python
allocations made while the optimizer runs; they should scale roughly
linearly with n + m (no quadratic candidate table is kept).

Run with:  python3 benchmark_signed_communities.py
"""

import random
import time
import tracemalloc
from collections import Counter

import networkx as nx
from networkx.algorithms.community import _signed


def build_graph(n, avg_degree=12, seed=123456789):
    """Fixed-seed sparse mixed-sign graph: 60% positive, 35% negative,
    5% zero-weight edges; weights uniform in [0.5, 3] in absolute value."""
    p = avg_degree / (n - 1)
    G = nx.fast_gnp_random_graph(n, p, seed=seed)
    rng = random.Random(seed + 1)
    for u, v in G.edges:
        r = rng.random()
        if r < 0.35:
            G[u][v]["weight"] = -rng.uniform(0.5, 3.0)
        elif r < 0.40:
            G[u][v]["weight"] = 0.0
        else:
            G[u][v]["weight"] = rng.uniform(0.5, 3.0)
    return G


def measure(func, *args, **kwargs):
    """Run ``func`` twice: once for wall time and candidate/gain counters
    (clean run), once under ``tracemalloc`` for peak extra Python memory
    (its instrumentation slows execution, so it is not timed)."""
    _signed._EVAL_STATS = Counter()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    stats = dict(_signed._EVAL_STATS)
    _signed._EVAL_STATS = None
    tracemalloc.start()
    func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak, stats


def main():
    print(
        f"{'n':>6} {'m':>7} {'optimizer':>8} {'time(s)':>9} {'peak(MB)':>9}"
        f" {'gain evals':>12} {'|C|':>5} {'score':>9} {'score_t(s)':>10}"
    )
    for n in (600, 1200):
        t0 = time.perf_counter()
        G = build_graph(n)
        build_t = time.perf_counter() - t0
        m = G.number_of_edges()
        print(f"# n={n} m={m} graph build time: {build_t:.3f}s")
        for name, func in (
            (
                "louvain",
                lambda: nx.community.louvain_communities(G, seed=42, signed=True),
            ),
            (
                "greedy",
                lambda: nx.community.greedy_modularity_communities(G, signed=True),
            ),
        ):
            part, elapsed, peak, stats = measure(func)
            t0 = time.perf_counter()
            score = nx.community.modularity(G, part, signed=True)
            score_t = time.perf_counter() - t0
            evals = sum(stats.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
            print(
                f"{n:>6} {m:>7} {name:>8} {elapsed:>9.3f} {peak / 1e6:>9.2f}"
                f" {evals:>12} {len(part):>5} {score:>9.4f} {score_t:>10.3f}"
            )
            print(f"        counters: {detail}")


if __name__ == "__main__":
    main()
