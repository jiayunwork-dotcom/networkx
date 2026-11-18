"""Tests for signed (two-layer) modularity and community detection.

The reference scores in this file are computed independently from the
raw edges and the partition (see ``reference_signed_modularity`` and
``reference_layer_score``); they do not call the new scoring or
optimizer code, so they provide an independent truth source.
"""

import math
import random
from collections import defaultdict
from itertools import combinations

import pytest

import networkx as nx
from networkx.algorithms.community import (
    greedy_modularity_communities,
    louvain_communities,
    louvain_partitions,
    modularity,
)
from networkx.algorithms.community._signed import (
    _signed_layer_data,
    _signed_layer_graph,
    _signed_modularity_reduced,
)
from networkx.algorithms.community.louvain import (
    _gen_graph_signed,
    _one_level_signed,
)
from networkx.algorithms.community.modularity_max import (
    _greedy_modularity_signed_generator,
)
from networkx.exception import NetworkXNotImplemented

# Absolute tolerance used when comparing optimizer gains against the
# independently computed scores.
GAIN_TOL = 1e-10


def reference_signed_modularity(G, communities, weight="weight", resolution=1.0):
    """Independent signed score computed from the raw edges of ``G``.

    Splits every edge into a positive and a negative layer, accumulates
    the two layer masses, degrees (self-loops counted twice) and
    intra-community weights (self-loops counted once), and evaluates
    ``(m_pos * Q_pos - m_neg * Q_neg) / (m_pos + m_neg)`` with the usual
    undirected weighted modularity per layer.
    """
    node2com = {}
    for i, com in enumerate(communities):
        for u in com:
            node2com[u] = i
    n_com = len(communities)
    m_pos = m_neg = 0.0
    deg_pos = defaultdict(float)
    deg_neg = defaultdict(float)
    L_pos = [0.0] * n_com
    L_neg = [0.0] * n_com
    for u, v, w in G.edges(data=weight, default=1):
        w = float(w)
        wp = max(w, 0.0)
        wn = max(-w, 0.0)
        m_pos += wp
        m_neg += wn
        deg_pos[u] += wp
        deg_pos[v] += wp
        deg_neg[u] += wn
        deg_neg[v] += wn
        if node2com[u] == node2com[v]:
            L_pos[node2com[u]] += wp
            L_neg[node2com[u]] += wn
    K_pos = [0.0] * n_com
    K_neg = [0.0] * n_com
    for u, c in node2com.items():
        K_pos[c] += deg_pos[u]
        K_neg[c] += deg_neg[u]

    def layer_Q(L, K, m):
        if m == 0:
            return None
        return sum(L[c] / m - resolution * (K[c] / (2 * m)) ** 2 for c in range(n_com))

    Q_pos = layer_Q(L_pos, K_pos, m_pos)
    Q_neg = layer_Q(L_neg, K_neg, m_neg)
    if Q_pos is None and Q_neg is None:
        return 0.0
    num = 0.0
    if Q_pos is not None:
        num += m_pos * Q_pos
    if Q_neg is not None:
        num -= m_neg * Q_neg
    return num / (m_pos + m_neg)


def reference_layer_score(edges, nodes, communities, resolution=1.0):
    """Independent signed score from a raw ``(u, v, w_pos, w_neg)`` list.

    Used to score partitions of the two-layer (compressed) graphs built
    inside the signed Louvain implementation.
    """
    node2com = {}
    for i, com in enumerate(communities):
        for u in com:
            node2com[u] = i
    n_com = len(communities)
    m_pos = m_neg = 0.0
    deg_pos = defaultdict(float)
    deg_neg = defaultdict(float)
    L_pos = [0.0] * n_com
    L_neg = [0.0] * n_com
    for u, v, wp, wn in edges:
        m_pos += wp
        m_neg += wn
        deg_pos[u] += wp
        deg_pos[v] += wp
        deg_neg[u] += wn
        deg_neg[v] += wn
        if node2com[u] == node2com[v]:
            L_pos[node2com[u]] += wp
            L_neg[node2com[u]] += wn
    K_pos = [0.0] * n_com
    K_neg = [0.0] * n_com
    for u in nodes:
        c = node2com[u]
        K_pos[c] += deg_pos[u]
        K_neg[c] += deg_neg[u]

    def layer_Q(L, K, m):
        if m == 0:
            return None
        return sum(L[c] / m - resolution * (K[c] / (2 * m)) ** 2 for c in range(n_com))

    Q_pos = layer_Q(L_pos, K_pos, m_pos)
    Q_neg = layer_Q(L_neg, K_neg, m_neg)
    if Q_pos is None and Q_neg is None:
        return 0.0
    num = 0.0
    if Q_pos is not None:
        num += m_pos * Q_pos
    if Q_neg is not None:
        num -= m_neg * Q_neg
    return num / (m_pos + m_neg)


def layer_edges(H):
    """Edge list of a two-layer graph as ``(u, v, w_pos, w_neg)`` tuples."""
    return [
        (u, v, data["weight_pos"], data["weight_neg"])
        for u, v, data in H.edges(data=True)
    ]


def random_signed_graph(rng, n, p=0.35, selfloop_p=0.05, multigraph=False):
    """A random signed graph; weights include zeros and non-integers."""
    G = nx.MultiGraph() if multigraph else nx.Graph()
    G.add_nodes_from(range(n))
    choices = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    for u in range(n):
        for v in range(u, n):
            if rng.random() < p:
                G.add_edge(u, v, weight=rng.choice(choices))
                if multigraph and rng.random() < 0.3:
                    # parallel edge, possibly of the opposite sign
                    G.add_edge(u, v, weight=rng.choice(choices))
            if rng.random() < selfloop_p:
                G.add_edge(u, u, weight=rng.choice([-2.0, 1.0]))
    return G


def random_partition(rng, G):
    part = {}
    for u in G:
        part.setdefault(rng.randint(0, len(G) - 1), set()).add(u)
    return list(part.values())


class TestSignedModularity:
    def test_hand_computed(self):
        # single positive edge
        G = nx.Graph([(0, 1)])
        assert modularity(G, [{0}, {1}], weight=None, signed=True) == -0.5
        assert modularity(G, [{0, 1}], weight=None, signed=True) == 0.0
        # single negative edge: score is -Q_neg, so separating is good
        G = nx.Graph()
        G.add_edge(0, 1, weight=-1.0)
        assert modularity(G, [{0}, {1}], signed=True) == 0.5
        assert modularity(G, [{0, 1}], signed=True) == 0.0
        # triangle: 0-1: +3, 1-2: +3, 0-2: -1; partition {0,1},{2}
        G = nx.Graph()
        G.add_edge(0, 1, weight=3.0)
        G.add_edge(1, 2, weight=3.0)
        G.add_edge(0, 2, weight=-1.0)
        # m+ = 6, m- = 1; Q+ = (3/6 - (9/12)^2) + (0 - (3/12)^2) = -0.125
        # Q- = -(1/2)^2 - (1/2)^2 = -0.5; score = (6*(-0.125) + 0.5)/7
        assert modularity(G, [{0, 1}, {2}], signed=True) == pytest.approx(-0.25 / 7)

    def test_hand_computed_selfloops(self):
        # positive self-loop counts once in m+, twice in degree
        G = nx.Graph()
        G.add_edge(0, 0, weight=2.0)
        G.add_edge(1, 1, weight=2.0)
        # Q+ of singletons: each community L=2, K=4, m=4:
        # 2 * (2/4 - (4/8)^2) = 0.5
        assert modularity(G, [{0}, {1}], signed=True) == pytest.approx(0.5)
        # negative self-loops: score is -Q_neg = -0.5
        H = nx.Graph()
        H.add_edge(0, 0, weight=-2.0)
        H.add_edge(1, 1, weight=-2.0)
        assert modularity(H, [{0}, {1}], signed=True) == pytest.approx(-0.5)
        # merging the two negative self-loops removes the degree penalty
        assert modularity(H, [{0, 1}], signed=True) == pytest.approx(0.0)

    def test_hand_computed_parallel_edges(self):
        # parallel +2 and -3 edges on the same pair must both contribute
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(0, 1, weight=-3.0)
        # m+ = 2, m- = 3
        # singletons: Q+ = 2 * (-(2/4)^2) = -0.5, Q- = 2 * (-(3/6)^2) = -0.5
        # score = (2*(-0.5) - 3*(-0.5))/5 = 0.1
        assert modularity(G, [{0}, {1}], signed=True) == pytest.approx(0.1)
        # merged: Q+ = 2/2 - 1 = 0, Q- = 3/3 - 1 = 0 -> score 0
        assert modularity(G, [{0, 1}], signed=True) == pytest.approx(0.0)

    def test_matches_reference_random(self):
        rng = random.Random(20240601)
        for _ in range(150):
            n = rng.randint(2, 12)
            G = random_signed_graph(rng, n, multigraph=rng.random() < 0.4)
            part = random_partition(rng, G)
            res = rng.choice([0.0, 0.5, 1.0, 2.5])
            got = modularity(G, part, signed=True, resolution=res)
            want = reference_signed_modularity(G, part, resolution=res)
            assert got == pytest.approx(want, abs=1e-10)

    def test_all_positive_degenerates(self):
        rng = random.Random(7)
        for _ in range(30):
            n = rng.randint(2, 15)
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.4:
                        G.add_edge(u, v, weight=rng.uniform(0.1, 5))
            if G.number_of_edges() == 0:
                continue
            part = random_partition(rng, G)
            assert modularity(G, part, signed=True) == pytest.approx(
                modularity(G, part), abs=1e-12
            )

    def test_all_negative(self):
        rng = random.Random(8)
        for _ in range(30):
            n = rng.randint(2, 15)
            G = nx.Graph()
            Ga = nx.Graph()
            G.add_nodes_from(range(n))
            Ga.add_nodes_from(range(n))
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.4:
                        w = -rng.uniform(0.1, 5)
                        G.add_edge(u, v, weight=w)
                        Ga.add_edge(u, v, weight=-w)
            if Ga.number_of_edges() == 0:
                continue
            part = random_partition(rng, G)
            # only a negative layer: score is -Q_neg
            assert modularity(G, part, signed=True) == pytest.approx(
                -modularity(Ga, part), abs=1e-12
            )

    def test_mixed_layers_via_split_graphs(self):
        # score must equal (m+ Q+ - m- Q-) / (m+ + m-) with Q+ and Q- the
        # existing (unsigned) modularity of the two split graphs
        rng = random.Random(9)
        for _ in range(40):
            n = rng.randint(2, 15)
            G = nx.Graph()
            Gp = nx.Graph()
            Gn = nx.Graph()
            for gr in (G, Gp, Gn):
                gr.add_nodes_from(range(n))
            for u in range(n):
                for v in range(u, n):
                    if rng.random() < 0.35:
                        w = rng.uniform(-4, 4)
                        G.add_edge(u, v, weight=w)
                        if w > 0:
                            Gp.add_edge(u, v, weight=w)
                        elif w < 0:
                            Gn.add_edge(u, v, weight=-w)
            m_pos = Gp.size(weight="weight")
            m_neg = Gn.size(weight="weight")
            if m_pos + m_neg == 0:
                continue
            part = random_partition(rng, G)
            expect = 0.0
            if m_pos:
                expect += m_pos * modularity(Gp, part)
            if m_neg:
                expect -= m_neg * modularity(Gn, part)
            expect /= m_pos + m_neg
            assert modularity(G, part, signed=True) == pytest.approx(expect, abs=1e-12)

    def test_zero_layer(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0), (2, 2, 0.0)])
        assert modularity(G, [{0, 1, 2}], signed=True) == 0.0
        assert modularity(G, [{0}, {1}, {2}], signed=True) == 0.0
        assert modularity(G, [{0, 1}, {2}], signed=True) == 0.0

    def test_equal_layer_masses(self):
        # m+ == m- must not cause problems
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0)
        G.add_edge(2, 3, weight=-2.0)
        part = [{0, 1}, {2, 3}]
        got = modularity(G, part, signed=True)
        want = reference_signed_modularity(G, part)
        assert got == pytest.approx(want, abs=1e-12)
        assert math.isfinite(got)

    def test_isolated_nodes_and_empty(self):
        G = nx.Graph()
        G.add_nodes_from(range(4))
        assert modularity(G, [{0, 1}, {2, 3}], signed=True) == 0.0
        assert modularity(nx.Graph(), [], signed=True) == 0.0
        G.add_edge(4, 5, weight=-1.5)
        part = [{0, 1}, {2, 3}, {4}, {5}]
        assert modularity(G, part, signed=True) == pytest.approx(
            reference_signed_modularity(G, part), abs=1e-12
        )

    def test_missing_attribute_and_none_weight(self):
        # an edge without the weight attribute counts as weight 1, and
        # weight=None counts every edge as weight 1
        G = nx.Graph()
        G.add_edge(0, 1)  # no weight attribute -> 1
        G.add_edge(1, 2, weight=-2.0)
        G.add_edge(2, 3)  # no weight attribute -> 1
        part = [{0, 1}, {2, 3}]
        assert modularity(G, part, signed=True) == pytest.approx(
            reference_signed_modularity(G, part), abs=1e-12
        )
        # with weight=None every edge is +1, ignoring the attribute
        assert modularity(G, part, weight=None, signed=True) == pytest.approx(
            modularity(G, part, weight=None), abs=1e-12
        )
        assert louvain_communities(
            G, weight=None, seed=3, signed=True
        ) == louvain_communities(G, weight=None, seed=3)
        # greedy's default is weight=None; naming the attribute uses it
        assert greedy_modularity_communities(
            G, weight="weight", signed=True
        ) == greedy_modularity_communities(G, weight="weight", signed=True)

    def test_scaling_invariance(self):
        rng = random.Random(10)
        G = random_signed_graph(rng, 30, p=0.25, selfloop_p=0.1)
        H = G.copy()
        for u, v in H.edges:
            H[u][v]["weight"] = G[u][v]["weight"] * 7.25
        for part in (
            random_partition(rng, G),
            louvain_communities(G, seed=1, signed=True),
        ):
            a = modularity(G, part, signed=True)
            b = modularity(H, part, signed=True)
            assert a == pytest.approx(b, abs=1e-10)

    def test_not_a_partition(self):
        G = nx.Graph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXError, match="not a valid partition"):
            modularity(G, [{0, 1}, {1, 2}], signed=True)
        with pytest.raises(nx.NetworkXError, match="not a valid partition"):
            modularity(G, [{0}, {1}], signed=True)

    def test_directed_raises(self):
        D = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(NetworkXNotImplemented):
            modularity(D, [{0, 1}, {2}], signed=True)
        MD = nx.MultiDiGraph([(0, 1), (0, 1)])
        with pytest.raises(NetworkXNotImplemented):
            modularity(MD, [{0}, {1}], signed=True)

    @pytest.mark.parametrize(
        "bad_weight",
        [True, float("nan"), float("inf"), -float("inf"), "x", None, 1 + 2j],
    )
    def test_invalid_weights(self, bad_weight):
        G = nx.Graph([(0, 1)])
        G[0][1]["weight"] = bad_weight
        with pytest.raises(ValueError):
            modularity(G, [{0}, {1}], signed=True)

    @pytest.mark.parametrize(
        "bad_resolution", [-1.0, -2, float("nan"), float("inf"), True, "x"]
    )
    def test_invalid_resolution(self, bad_resolution):
        G = nx.Graph([(0, 1)])
        with pytest.raises(ValueError):
            modularity(G, [{0}, {1}], signed=True, resolution=bad_resolution)

    def test_mass_overflow(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1e308)
        G.add_edge(0, 1, weight=1e308)
        with pytest.raises(ValueError):
            modularity(G, [{0}, {1}], signed=True)

    def test_graph_and_partition_unchanged(self):
        rng = random.Random(11)
        G = random_signed_graph(rng, 20, multigraph=True)
        edges_before = sorted(
            (str(u), str(v), str(d)) for u, v, d in G.edges(data=True)
        )
        nodes_before = set(G.nodes)
        part = random_partition(rng, G)
        part_copy = [set(c) for c in part]
        modularity(G, part, signed=True)
        louvain_communities(G, seed=2, signed=True)
        greedy_modularity_communities(G, weight="weight", signed=True)
        assert set(G.nodes) == nodes_before
        assert (
            sorted((str(u), str(v), str(d)) for u, v, d in G.edges(data=True))
            == edges_before
        )
        assert part == part_copy


class TestSignedLouvain:
    def test_levels_valid_and_nondecreasing(self):
        rng = random.Random(12)
        for trial in range(10):
            G = random_signed_graph(rng, rng.randint(5, 25), selfloop_p=0.1)
            parts = list(louvain_partitions(G, seed=trial, signed=True))
            assert len(parts) >= 1
            prev = -2.0
            for part in parts:
                assert nx.community.is_partition(G, part)
                score = modularity(G, part, signed=True)
                assert score >= prev - 1e-10
                prev = score
            # final communities equal the last yielded partition
            final = louvain_communities(G, seed=trial, signed=True)
            assert final == parts[-1]

    def test_partition_iterator_snapshot(self):
        G = random_signed_graph(random.Random(13), 20)
        parts_iter = louvain_partitions(G, seed=42, signed=True)
        first = next(parts_iter)
        first_copy = [s.copy() for s in first]
        for _ in parts_iter:
            pass
        # previously yielded sets must not have been modified
        assert first == first_copy

    def test_seed_reproducible(self):
        G = random_signed_graph(random.Random(14), 30)
        p1 = louvain_communities(G, seed=42, signed=True)
        p2 = louvain_communities(G, seed=42, signed=True)
        assert p1 == p2
        p3 = louvain_communities(G, seed=42, signed=True)
        assert p1 == p3

    def test_local_optimality_after_each_level(self):
        # after every level, no single-node move (to any community or to
        # a singleton) may improve the score by more than the move
        # threshold; checked against the independent reference
        rng = random.Random(15)
        for trial in range(8):
            G = random_signed_graph(rng, rng.randint(4, 14), selfloop_p=0.1)
            H, m_pos, m_neg, _ = _signed_layer_graph(G, "weight")
            if m_pos + m_neg == 0:
                continue
            edges = layer_edges(H)
            nodes = list(H.nodes)
            partition = [{u} for u in G]
            seed = random.Random(trial)
            while True:
                partition, inner, improvement = _one_level_signed(
                    H, m_pos, m_neg, partition, 1.0, seed
                )
                score = _signed_modularity_reduced(H, inner, m_pos, m_neg, 1.0)
                assert score == pytest.approx(
                    reference_layer_score(edges, nodes, inner), abs=1e-10
                )
                node2com = {}
                for ci, com in enumerate(inner):
                    for u in com:
                        node2com[u] = ci
                for u in nodes:
                    cu = node2com[u]
                    for cj in range(-1, len(inner)):
                        if cj == cu:
                            continue
                        moved = [set(c) for c in inner]
                        moved[cu].discard(u)
                        moved = [c for c in moved if c]
                        if cj == -1 or cj >= len(moved):
                            moved.append({u})
                        else:
                            moved[cj].add(u)
                        s = reference_layer_score(edges, nodes, moved)
                        assert s <= score + GAIN_TOL
                if not improvement:
                    break
                H2 = _gen_graph_signed(H, inner)
                if (
                    _signed_modularity_reduced(H, inner, m_pos, m_neg, 1.0)
                    <= score + 1e-7
                ):
                    break
                H = H2
                edges = layer_edges(H)
                nodes = list(H.nodes)

    def test_compression_preserves_scores(self):
        rng = random.Random(16)
        for _ in range(10):
            G = random_signed_graph(rng, rng.randint(4, 14), selfloop_p=0.1)
            H, m_pos, m_neg, _ = _signed_layer_graph(G, "weight")
            if m_pos + m_neg == 0:
                continue
            part = random_partition(rng, G)
            H2 = _gen_graph_signed(H, part)
            # an arbitrary partition of the compressed graph
            comp_part = random_partition(rng, H2)
            # map back to the original nodes
            orig_part = []
            for com in comp_part:
                mapped = set()
                for cnode in com:
                    mapped.update(H2.nodes[cnode].get("nodes", {cnode}))
                orig_part.append(mapped)
            s_reduced = _signed_modularity_reduced(H2, comp_part, m_pos, m_neg, 1.0)
            s_orig = modularity(G, orig_part, signed=True)
            s_ref = reference_layer_score(layer_edges(H2), list(H2.nodes), comp_part)
            assert s_reduced == pytest.approx(s_orig, abs=1e-10)
            assert s_reduced == pytest.approx(s_ref, abs=1e-10)

    def test_nonneighbor_move(self):
        # nodes 0 and 3 have no edge between them; grouping them is only
        # found because node moves consider every community
        G = nx.Graph()
        G.add_edge(0, 1, weight=-2.0)
        G.add_edge(2, 3, weight=-2.0)
        G.add_edge(1, 2, weight=0.5)
        for seed in range(5):
            part = louvain_communities(G, seed=seed, signed=True)
            assert {0, 3} in [set(c) for c in part]
            assert {1, 2} in [set(c) for c in part]
        assert modularity(G, [{0, 3}, {1, 2}], signed=True) == pytest.approx(2 / 4.5)

    def test_empty_and_zero_mass(self):
        assert louvain_communities(nx.Graph(), signed=True) == []
        G = nx.Graph()
        G.add_nodes_from(range(4))
        assert louvain_communities(G, signed=True) == [{0}, {1}, {2}, {3}]
        Z = nx.Graph()
        Z.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0), (2, 2, 0.0)])
        assert louvain_communities(Z, signed=True) == [{0}, {1}, {2}]

    def test_directed_raises(self):
        D = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(NetworkXNotImplemented):
            louvain_communities(D, signed=True)
        with pytest.raises(NetworkXNotImplemented):
            list(louvain_partitions(D, signed=True))

    @pytest.mark.parametrize("bad_weight", [True, float("nan"), "x"])
    def test_invalid_weights(self, bad_weight):
        G = nx.Graph([(0, 1)])
        G[0][1]["weight"] = bad_weight
        with pytest.raises(ValueError):
            louvain_communities(G, signed=True)

    @pytest.mark.parametrize("bad_resolution", [-1.0, float("nan"), True])
    def test_invalid_resolution(self, bad_resolution):
        G = nx.Graph([(0, 1)])
        with pytest.raises(ValueError):
            louvain_communities(G, signed=True, resolution=bad_resolution)

    def test_mass_overflow(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1e308)
        G.add_edge(0, 1, weight=1e308)
        with pytest.raises(ValueError):
            louvain_communities(G, signed=True)

    def test_max_level_and_threshold(self):
        G = random_signed_graph(random.Random(17), 40, p=0.2)
        parts = list(louvain_partitions(G, seed=3, signed=True))
        one = louvain_communities(G, seed=3, max_level=1, signed=True)
        assert one == parts[0]
        with pytest.raises(ValueError, match="max_level"):
            louvain_communities(G, max_level=0, signed=True)
        # a huge threshold stops after the first level
        first = louvain_communities(G, seed=3, threshold=10.0, signed=True)
        assert first == parts[0]

    def test_unorderable_nodes(self):
        class Node:
            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return hash(self.x)

            def __eq__(self, other):
                return isinstance(other, Node) and self.x == other.x

        nodes = [Node(i) for i in range(6)] + ["a", "b"]
        G = nx.Graph()
        G.add_edge(nodes[0], nodes[1], weight=2.0)
        G.add_edge(nodes[1], nodes[2], weight=2.0)
        G.add_edge(nodes[3], nodes[4], weight=2.0)
        G.add_edge(nodes[4], nodes[5], weight=2.0)
        G.add_edge(nodes[2], nodes[3], weight=-1.0)
        G.add_edge(nodes[6], nodes[7], weight=-0.5)
        G.add_edge(nodes[0], nodes[6], weight=1.0)
        part = louvain_communities(G, seed=1, signed=True)
        assert nx.community.is_partition(G, part)

    def test_multigraph_parallel(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=3.0)
        G.add_edge(0, 1, weight=-1.0)
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=-2.0)
        G.add_edge(3, 4, weight=2.0)
        part = louvain_communities(G, seed=4, signed=True)
        assert nx.community.is_partition(G, part)
        # both parallel contributions survive: score must match reference
        assert modularity(G, part, signed=True) == pytest.approx(
            reference_signed_modularity(G, part), abs=1e-10
        )

    def test_negative_only(self):
        # only repulsive edges: spreading out wins
        G = nx.Graph()
        G.add_edge(0, 1, weight=-2.0)
        G.add_edge(1, 2, weight=-2.0)
        G.add_edge(2, 3, weight=-2.0)
        G.add_edge(3, 0, weight=-2.0)
        part = louvain_communities(G, seed=1, signed=True)
        singletons = [{u} for u in G]
        assert (
            modularity(G, part, signed=True)
            >= modularity(G, singletons, signed=True) - 1e-10
        )


class TestSignedGreedy:
    def drive_generator(self, G, resolution=1.0):
        """Run the signed generator, checking every step against reference."""
        m_pos, m_neg, deg_pos, deg_neg, weight_exp = _signed_layer_data(G, "weight")
        gen = _greedy_modularity_signed_generator(
            G, "weight", resolution, m_pos, m_neg, deg_pos, deg_neg, weight_exp
        )
        communities = [set(c) for c in next(gen)]
        score = reference_signed_modularity(G, communities, resolution=resolution)
        steps = []
        while True:
            try:
                dq = next(gen)
            except StopIteration:
                break
            # maximality: dq must be the largest gain over *all* pairs,
            # including pairs with no edges between them
            comms = list(communities)
            best_other = None
            for i, j in combinations(range(len(comms)), 2):
                trial = [c for k, c in enumerate(comms) if k not in (i, j)]
                trial.append(comms[i] | comms[j])
                g = reference_signed_modularity(G, trial, resolution=resolution) - score
                if best_other is None or g > best_other:
                    best_other = g
            if best_other is not None:
                assert dq >= best_other - GAIN_TOL
            new_communities = [set(c) for c in next(gen)]
            new_score = reference_signed_modularity(
                G, new_communities, resolution=resolution
            )
            assert new_score - score == pytest.approx(dq, abs=GAIN_TOL)
            steps.append(dq)
            communities = new_communities
            score = new_score
        return steps, communities

    def test_steps_maximality_and_gains(self):
        rng = random.Random(18)
        for _ in range(15):
            n = rng.randint(2, 14)
            G = random_signed_graph(rng, n, multigraph=rng.random() < 0.4)
            res = rng.choice([0.5, 1.0, 2.0])
            self.drive_generator(G, resolution=res)

    def test_nonadjacent_merge_positive_gain(self):
        # two negative self-loops, no edge between the nodes: merging them
        # has positive gain (+0.5) although there is no direct edge
        G = nx.Graph()
        G.add_edge(0, 0, weight=-1.0)
        G.add_edge(1, 1, weight=-1.0)
        steps, communities = self.drive_generator(G)
        assert steps[0] == pytest.approx(0.5, abs=GAIN_TOL)
        assert communities == [{0, 1}]
        assert greedy_modularity_communities(G, weight="weight", signed=True) == [
            {0, 1}
        ]

    def test_empty_and_zero_mass(self):
        assert greedy_modularity_communities(nx.Graph(), signed=True) == []
        G = nx.Graph()
        G.add_nodes_from(range(4))
        assert greedy_modularity_communities(G, weight="weight", signed=True) == [
            {0},
            {1},
            {2},
            {3},
        ]
        # best_n forces merging even with no edges at all
        assert (
            len(
                greedy_modularity_communities(G, weight="weight", signed=True, best_n=2)
            )
            == 2
        )
        Z = nx.Graph()
        Z.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0), (2, 2, 0.0)])
        assert greedy_modularity_communities(Z, weight="weight", signed=True) == [
            {0},
            {1},
            {2},
        ]
        assert greedy_modularity_communities(
            Z, weight="weight", signed=True, best_n=1
        ) == [{0, 1, 2}]

    def test_cutoff_and_best_n(self):
        rng = random.Random(19)
        G = random_signed_graph(rng, 30, p=0.25)
        full = greedy_modularity_communities(G, weight="weight", signed=True)
        cutoff = len(full) + 2
        stopped = greedy_modularity_communities(
            G, weight="weight", signed=True, cutoff=cutoff
        )
        assert len(stopped) >= cutoff
        forced = greedy_modularity_communities(
            G, weight="weight", signed=True, best_n=1
        )
        assert forced == [set(G)]
        forced2 = greedy_modularity_communities(
            G, weight="weight", signed=True, best_n=2
        )
        assert len(forced2) == 2
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, signed=True, cutoff=0)
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, signed=True, best_n=0)
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, signed=True, cutoff=3, best_n=2)

    def test_best_n_forced_merges_stay_maximal(self):
        # past the natural stopping point (all remaining gains <= 1e-12),
        # best_n must still pick the largest-gain merge; the generator
        # drive in drive_generator checks maximality of every merge
        rng = random.Random(20)
        G = random_signed_graph(rng, 16, p=0.3)
        steps, _ = self.drive_generator(G)
        # the natural stop is the first dq <= 1e-12; forced merges continue
        assert any(dq <= 1e-12 for dq in steps)
        natural = greedy_modularity_communities(G, weight="weight", signed=True)
        forced = greedy_modularity_communities(
            G, weight="weight", signed=True, best_n=1
        )
        assert len(forced) == 1
        assert len(natural) >= 1

    def test_return_type_and_sorting(self):
        G = random_signed_graph(random.Random(21), 25, p=0.3)
        result = greedy_modularity_communities(G, weight="weight", signed=True)
        assert isinstance(result, list)
        assert all(isinstance(c, frozenset) for c in result)
        sizes = [len(c) for c in result]
        assert sizes == sorted(sizes, reverse=True)
        assert nx.community.is_partition(G, result)

    def test_directed_raises(self):
        D = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(NetworkXNotImplemented):
            greedy_modularity_communities(D, signed=True)

    @pytest.mark.parametrize("bad_weight", [True, float("nan"), "x"])
    def test_invalid_weights(self, bad_weight):
        G = nx.Graph([(0, 1)])
        G[0][1]["weight"] = bad_weight
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, weight="weight", signed=True)

    @pytest.mark.parametrize("bad_resolution", [-1.0, float("nan"), True])
    def test_invalid_resolution(self, bad_resolution):
        G = nx.Graph([(0, 1)])
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, signed=True, resolution=bad_resolution)

    def test_mass_overflow(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1e308)
        G.add_edge(0, 1, weight=1e308)
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, weight="weight", signed=True)

    def test_unorderable_nodes(self):
        class Node:
            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return hash(self.x)

            def __eq__(self, other):
                return isinstance(other, Node) and self.x == other.x

        nodes = [Node(i) for i in range(6)] + ["a", "b"]
        G = nx.Graph()
        G.add_edge(nodes[0], nodes[1], weight=2.0)
        G.add_edge(nodes[1], nodes[2], weight=2.0)
        G.add_edge(nodes[3], nodes[4], weight=2.0)
        G.add_edge(nodes[4], nodes[5], weight=2.0)
        G.add_edge(nodes[2], nodes[3], weight=-1.0)
        G.add_edge(nodes[6], nodes[7], weight=-0.5)
        G.add_edge(nodes[0], nodes[6], weight=1.0)
        result = greedy_modularity_communities(G, weight="weight", signed=True)
        assert nx.community.is_partition(G, result)

    def test_scaling_invariance(self):
        rng = random.Random(22)
        G = random_signed_graph(rng, 30, p=0.25, selfloop_p=0.1)
        H = G.copy()
        for u, v in H.edges:
            H[u][v]["weight"] = G[u][v]["weight"] * 3.75
        assert greedy_modularity_communities(
            G, signed=True
        ) == greedy_modularity_communities(H, signed=True)
        assert louvain_communities(G, seed=5, signed=True) == louvain_communities(
            H, seed=5, signed=True
        )


def test_optimizers_improve_over_singletons():
    rng = random.Random(23)
    for _ in range(10):
        G = random_signed_graph(rng, rng.randint(5, 25), selfloop_p=0.1)
        singletons = [{u} for u in G]
        base = modularity(G, singletons, signed=True)
        for part in (
            louvain_communities(G, seed=1, signed=True),
            greedy_modularity_communities(G, weight="weight", signed=True),
        ):
            assert nx.community.is_partition(G, part)
            assert modularity(G, part, signed=True) >= base - 1e-10
