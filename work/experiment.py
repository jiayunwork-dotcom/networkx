"""Before/after cost experiment for the signed numerical-range fix.

Runs both signed optimizers three times each on fixed-seed sparse
mixed-sign graphs (n=100 and n=200, average undirected degree ~8) and
records wall time, peak extra traced Python memory and the
candidate/gain evaluation counters.  Graph construction and the
independent reference score happen outside every measurement.

Usage: python3 experiment.py <package_root> <label>
"""

import random
import sys
import time
import tracemalloc
from collections import Counter

sys.path.insert(0, sys.argv[1])
label = sys.argv[2]

import networkx as nx  # noqa: E402
from networkx.algorithms.community import _signed  # noqa: E402
from networkx.algorithms.community import (  # noqa: E402
    greedy_modularity_communities,
    louvain_communities,
    modularity,
)


def build_graph(n, seed):
    """Fixed-seed sparse mixed-sign graph, average undirected degree ~8.

    60% positive, 35% negative, 5% zero; |weight| uniform in [0.5, 3].
    """
    p = 8.0 / (n - 1)
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


def measure(func):
    """Three clean timed runs plus one tracemalloc run and the counters."""
    times = []
    for _ in range(3):
        _signed._EVAL_STATS = Counter()
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
        counters = dict(_signed._EVAL_STATS)
        _signed._EVAL_STATS = None
    tracemalloc.start()
    func()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, times, peak, counters


def main():
    print(f"===== {label} ({nx.__file__}) =====")
    print(
        f"{'n':>4} {'m':>5} {'optimizer':>8} {'runs (s)':>34} "
        f"{'peak(MB)':>9} {'gain evals':>12} {'|C|':>4} {'score':>9}"
    )
    for n in (100, 200):
        seed = 987654321 + n
        G = build_graph(n, seed)  # outside every measurement
        m = G.number_of_edges()
        assert abs(2 * m / n - 8.0) < 0.6
        for name, func in (
            ("louvain", lambda: louvain_communities(
                G, weight="weight", seed=42, signed=True)),
            ("greedy", lambda: greedy_modularity_communities(
                G, weight="weight", signed=True)),
        ):
            part, times, peak, counters = measure(func)
            # reference scoring outside the timed region
            score = modularity(G, part, weight="weight", signed=True)
            evals = sum(counters.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counters.items()))
            times_str = ", ".join(f"{t:.4f}" for t in times)
            print(
                f"{n:>4} {m:>5} {name:>8} {times_str:>34} "
                f"{peak / 1e6:>9.3f} {evals:>12} {len(part):>4} {score:>9.5f}"
            )
            print(f"         counters: {detail}")


if __name__ == "__main__":
    main()
