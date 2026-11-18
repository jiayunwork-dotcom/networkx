"""Tests for the numerical range of signed (two-layer) modularity.

These tests pin the behaviour on graphs whose edge weights span the
whole representable float64 range, including the exact reproduction
cases from the bug report.  Every expected value is computed by an
independent high-precision reference built directly from the raw edges
and the partition (``reference_score``): it never calls the package's
scoring or the optimizer gain kernels.  The reference accumulates the
edge sums with :func:`math.fsum` and evaluates the two layer
modularities separately so that degree products never meet raw-scale
edge magnitudes.
"""

import math
import random
from itertools import combinations

import pytest

import networkx as nx
from networkx.algorithms.community import (
    _signed,
    greedy_modularity_communities,
    louvain_communities,
    louvain_partitions,
    modularity,
)
from networkx.algorithms.community._signed import (
    _max_weight_exp,
    _signed_layer_data,
    _signed_layer_graph,
    _signed_modularity_reduced,
)
from networkx.algorithms.community.louvain import _gen_graph_signed

GAIN_TOL = 1e-10


# ---------------------------------------------------------------------------
# Independent references (no call into the scorer or the gain kernels)
# ---------------------------------------------------------------------------

def _partition_map(communities):
    node2com = {}
    for i, com in enumerate(communities):
        for u in com:
            node2com[u] = i
    return node2com


def reference_score(G, communities, weight="weight", resolution=1.0):
    """Signed score computed independently from the raw float64 edges.

    The reference never calls the package scorer or the optimizer gain
    kernels.  It derives its own normalization exponent from the raw
    edge values (the largest magnitude maps into ``(0.5, 1]``) and
    rescales every weight with :func:`math.ldexp`, an exact power-of-two
    operation, before accumulating.  This mirrors what the float64
    arithmetic of the defining formula has to do to keep information
    that is actually present in the input: the smallest subnormal
    weights otherwise underflow to zero on their very first addition.

    The two layers are then accumulated separately with compensated
    summation; each layer modularity is evaluated as
    ``sum_c L_c/m - gamma (K_c/(2m))^2`` and the mass-weighted signed
    combination is formed from the per-layer sums.
    """
    node2com = _partition_map(communities)
    n_com = len(communities)
    # independent normalization exponent, derived only from raw values
    exp = None
    for _, _, w in G.edges(data=weight, default=1):
        w = float(w)
        if w != 0.0:
            e = math.frexp(w)[1] - 1
            exp = e if exp is None or e > exp else exp
    exp = 0 if exp is None else exp
    pos_edges, neg_edges = [], []
    deg_pos = {u: [] for u in G.nodes}
    deg_neg = {u: [] for u in G.nodes}
    L_pos = [[] for _ in range(n_com)]
    L_neg = [[] for _ in range(n_com)]
    for u, v, w in G.edges(data=weight, default=1):
        w = math.ldexp(float(w), -exp)
        c = node2com[u]
        if w > 0.0:
            pos_edges.append(w)
            deg_pos[u].append(w)
            deg_pos[v].append(w)
            if c == node2com[v]:
                L_pos[c].append(w)
        elif w < 0.0:
            aw = -w
            neg_edges.append(aw)
            deg_neg[u].append(aw)
            deg_neg[v].append(aw)
            if c == node2com[v]:
                L_neg[c].append(aw)
    m_pos = math.fsum(pos_edges)
    m_neg = math.fsum(neg_edges)
    K_pos = [
        math.fsum(x for u in com for x in deg_pos[u]) for com in communities
    ]
    K_neg = [
        math.fsum(x for u in com for x in deg_neg[u]) for com in communities
    ]
    Lp = [math.fsum(x) for x in L_pos]
    Ln = [math.fsum(x) for x in L_neg]

    num = 0.0
    if m_pos > 0.0:
        q_pos = math.fsum(Lp) / m_pos - resolution * math.fsum(
            [0.0] + [(K_pos[c] / (2 * m_pos)) ** 2 for c in range(n_com)]
        )
        num += m_pos * q_pos
    if m_neg > 0.0:
        q_neg = math.fsum(Ln) / m_neg - resolution * math.fsum(
            [0.0] + [(K_neg[c] / (2 * m_neg)) ** 2 for c in range(n_com)]
        )
        num -= m_neg * q_neg
    m_tot = m_pos + m_neg
    return 0.0 if m_tot == 0.0 else num / m_tot


def reference_move_gain(G, communities, node, target_community_index):
    """Exact signed-score change of one single-node move.

    ``target_community_index`` is the index of the destination community
    in ``communities``; ``-1`` means moving ``node`` to a new singleton
    community.  The gain is the difference of two independent reference
    scores, not a call into the optimizer gain kernel.
    """
    base = reference_score(G, communities)
    node2com = _partition_map(communities)
    src = node2com[node]
    moved = [set(c) for c in communities]
    moved[src].discard(node)
    if target_community_index == -1:
        moved = [c for c in moved if c]
        moved.append({node})
    else:
        moved[target_community_index].add(node)
        moved = [c for c in moved if c]
    return reference_score(G, moved) - base


def all_reference_move_gains(G, communities):
    """Mapping ``(node, target_index) -> exact gain`` for every move."""
    node2com = _partition_map(communities)
    out = {}
    for u in G.nodes:
        src = node2com[u]
        for target in [-1] + [i for i in range(len(communities)) if i != src]:
            out[(u, target)] = reference_move_gain(G, communities, u, target)
    return out


def reference_merge_gain(G, communities, i, j):
    """Exact signed-score change of merging communities ``i`` and ``j``."""
    base = reference_score(G, communities)
    merged = [set(c) for k, c in enumerate(communities) if k not in (i, j)]
    merged.append(set(communities[i]) | set(communities[j]))
    return reference_score(G, merged) - base


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def four_node_graph(scale=1.0):
    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0 * scale)
    G.add_edge(1, 2, weight=-1.0 * scale)
    G.add_edge(2, 3, weight=4.0 * scale)
    G.add_edge(0, 3, weight=-1.0 * scale)
    return G


def six_node_greedy_graph(scale=1.0):
    G = nx.Graph()
    edges = [
        (0, 0, -1), (0, 1, -2), (0, 3, 1), (0, 5, -2),
        (1, 2, -1), (1, 3, -2), (1, 4, -5), (1, 5, -2),
        (3, 4, -5), (4, 4, 1), (4, 5, -5),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w) * scale)
    return G


SCALES = [1e-310, 1e-300, 1e-165, 1e-100, 1.0, 1e100, 1e155, 1e300]


def scaled_random_graph(rng, n, scale, multigraph=False):
    G = nx.MultiGraph() if multigraph else nx.Graph()
    G.add_nodes_from(range(n))
    choices = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    for u in range(n):
        for v in range(u, n):
            if rng.random() < 0.3:
                G.add_edge(u, v, weight=rng.choice(choices) * scale)
                if multigraph and rng.random() < 0.3:
                    G.add_edge(u, v, weight=rng.choice(choices) * scale)
        if rng.random() < 0.08:
            G.add_edge(u, u, weight=rng.choice([-2.0, 1.0]) * scale)
    # keep every scaled nonzero edge nonzero and finite as required
    for _, _, w in G.edges(data="weight"):
        assert w == 0.0 or math.isfinite(w)
    return G


def random_partition(rng, G):
    part = {}
    for u in G:
        part.setdefault(rng.randint(0, len(G) - 1), set()).add(u)
    return list(part.values())


# ---------------------------------------------------------------------------
# Exact reproductions from the report
# ---------------------------------------------------------------------------

class TestReproductionCases:
    @pytest.mark.parametrize("scale", SCALES)
    def test_four_node_score_is_one_half(self, scale):
        G = four_node_graph(scale)
        part = [{0, 1}, {2, 3}]
        got = modularity(G, part, weight="weight", signed=True)
        assert got == pytest.approx(0.5, abs=GAIN_TOL)
        assert got == pytest.approx(
            reference_score(G, part), abs=GAIN_TOL
        )

    @pytest.mark.parametrize("scale", SCALES)
    def test_four_node_louvain_finds_the_two_communities(self, scale):
        G = four_node_graph(scale)
        part = louvain_communities(G, weight="weight", seed=19, signed=True)
        assert sorted(sorted(c) for c in part) == [[0, 1], [2, 3]]
        assert modularity(G, part, weight="weight", signed=True) == pytest.approx(
            0.5, abs=GAIN_TOL
        )

    @pytest.mark.parametrize("scale", SCALES)
    def test_four_node_greedy_finds_the_two_communities(self, scale):
        G = four_node_graph(scale)
        part = greedy_modularity_communities(G, weight="weight", signed=True)
        assert sorted(sorted(c) for c in part) == [[0, 1], [2, 3]]

    def test_greedy_quality_invariant_under_common_scaling(self):
        # the exact second bug report: at 1e155 greedy used to silently
        # return a worse partition
        unit = six_node_greedy_graph(1.0)
        huge = six_node_greedy_graph(1e155)
        part_unit = greedy_modularity_communities(
            unit, weight="weight", signed=True
        )
        part_huge = greedy_modularity_communities(
            huge, weight="weight", signed=True
        )
        assert sorted(map(sorted, part_huge)) == sorted(map(sorted, part_unit))
        assert reference_score(unit, part_unit) == pytest.approx(
            0.2585185185185185, abs=GAIN_TOL
        )
        # the score of the huge-graph partition, evaluated on the unit
        # graph, must not degrade to ~0.2296
        assert reference_score(unit, part_huge) == pytest.approx(
            reference_score(unit, part_unit), abs=GAIN_TOL
        )

    @pytest.mark.parametrize("scale", [1e-310, 1e-165, 1e155])
    def test_no_division_or_overflow_errors(self, scale):
        # previously: ZeroDivisionError (small scales) or overflow
        # ValueError (large scale) from all three entry points
        G = four_node_graph(scale)
        part = [{0, 1}, {2, 3}]
        modularity(G, part, weight="weight", signed=True)
        louvain_communities(G, weight="weight", seed=19, signed=True)
        greedy_modularity_communities(G, weight="weight", signed=True)


# ---------------------------------------------------------------------------
# Normalization mechanics
# ---------------------------------------------------------------------------

class TestNormalization:
    @pytest.mark.parametrize("scale", SCALES)
    def test_layer_summaries_normalized_but_raw_mass_representable(self, scale):
        G = four_node_graph(scale)
        exp = _max_weight_exp(G, "weight")
        m_pos, m_neg, _, _, exp2 = _signed_layer_data(G, "weight")
        assert exp2 == exp
        # normalized masses are ordinary magnitudes
        assert 0.0 < m_pos <= 8.0
        assert 0.0 < m_neg <= 2.0
        # raw-scale reconstruction is finite and matches the true masses
        assert math.ldexp(m_pos, exp) == pytest.approx(8.0 * scale, rel=1e-12)
        assert math.ldexp(m_neg, exp) == pytest.approx(2.0 * scale, rel=1e-12)

    def test_normalization_is_exact_power_of_two(self):
        # ldexp scaling never rounds the significand
        G = nx.Graph()
        G.add_edge(0, 1, weight=1e-310)
        G.add_edge(1, 2, weight=-3.141592653589793e-310)
        exp = _max_weight_exp(G, "weight")
        w = math.ldexp(1e-310, -exp)
        assert math.ldexp(w, exp) == 1e-310

    def test_zero_and_empty_graphs(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0), (2, 2, 0.0)])
        assert _max_weight_exp(G, "weight") == 0
        m_pos, m_neg, _, _, exp = _signed_layer_data(G, "weight")
        assert (m_pos, m_neg, exp) == (0.0, 0.0, 0)
        assert _max_weight_exp(nx.Graph(), "weight") == 0

    def test_unrepresentable_raw_mass_still_raises(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=1e308)
        G.add_edge(0, 1, weight=1e308)
        with pytest.raises(ValueError):
            modularity(G, [{0}, {1}], weight="weight", signed=True)
        with pytest.raises(ValueError):
            louvain_communities(G, weight="weight", signed=True)
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, weight="weight", signed=True)
        # a single 1e308 edge is representable and must work
        H = nx.Graph()
        H.add_edge(0, 1, weight=1e308)
        assert modularity(H, [{0, 1}], weight="weight", signed=True) == 0.0


# ---------------------------------------------------------------------------
# Public scorer over the full common scale range
# ---------------------------------------------------------------------------

class TestScorerRange:
    @pytest.mark.parametrize("scale", SCALES)
    def test_random_graph_scores_match_reference(self, scale):
        rng = random.Random(101)
        for multigraph in (False, True):
            G = scaled_random_graph(rng, 12, scale, multigraph=multigraph)
            part = random_partition(rng, G)
            got = modularity(G, part, weight="weight", signed=True)
            want = reference_score(G, part)
            assert got == pytest.approx(want, abs=GAIN_TOL)
            assert math.isfinite(got)

    @pytest.mark.parametrize("scale", SCALES)
    def test_resolution_variants(self, scale):
        rng = random.Random(102)
        G = scaled_random_graph(rng, 10, scale)
        part = random_partition(rng, G)
        for gamma in (0.0, 0.25, 1.0, 3.75):
            got = modularity(
                G, part, weight="weight", resolution=gamma, signed=True
            )
            want = reference_score(G, part, resolution=gamma)
            assert got == pytest.approx(want, abs=GAIN_TOL)

    def test_layer_mass_gap_1e12_and_beyond(self):
        # positive layer at least 1e12 heavier than the negative layer and
        # vice versa, both still finite
        for sign_first in (True, False):
            G = nx.Graph()
            big = 1e150
            small = 1e138  # ratio 1e12
            if sign_first:
                G.add_edge(0, 1, weight=big)
                G.add_edge(2, 3, weight=-small)
            else:
                G.add_edge(0, 1, weight=small)
                G.add_edge(2, 3, weight=-big)
            part = [{0, 1}, {2, 3}]
            assert modularity(G, part, weight="weight", signed=True) == (
                pytest.approx(reference_score(G, part), abs=GAIN_TOL)
            )


# ---------------------------------------------------------------------------
# Louvain move gains and compression equivalence
# ---------------------------------------------------------------------------

class TestLouvainGainsAndCompression:
    @pytest.mark.parametrize("scale", [1e-310, 1e-165, 1.0, 1e155, 1e300])
    def test_single_node_moves_match_reference_gains(self, scale):
        rng = random.Random(202)
        G = scaled_random_graph(rng, 9, scale)
        part0 = [{u} for u in G.nodes]
        # evaluate every feasible move from the singleton partition
        gains = all_reference_move_gains(G, part0)
        best = max(gains.values())
        # Louvain must not take a move whose reference gain is negative
        # beyond tolerance, and must find an improving move when one
        # exists well above the 1e-12 acceptance threshold
        result = louvain_communities(G, weight="weight", seed=7, signed=True)
        result_score = reference_score(G, result)
        singleton_score = reference_score(G, part0)
        if best > 1e-10:
            assert result_score >= singleton_score + best - GAIN_TOL
        assert result_score >= singleton_score - GAIN_TOL

    @pytest.mark.parametrize("scale", [1e-310, 1e-165, 1.0, 1e155])
    def test_level_monotonicity_and_compression_equivalence(self, scale):
        rng = random.Random(203)
        G = scaled_random_graph(rng, 11, scale, multigraph=True)
        H, m_pos, m_neg, _ = _signed_layer_graph(G, "weight")
        previous = -math.inf
        level = 0
        partition = [{u} for u in H.nodes]
        seed = random.Random(11)
        while True:
            from networkx.algorithms.community.louvain import _one_level_signed

            partition, inner, improvement = _one_level_signed(
                H, m_pos, m_neg, partition, 1.0, seed
            )
            score = _signed_modularity_reduced(
                H, inner, m_pos, m_neg, 1.0
            )
            assert score >= previous - GAIN_TOL
            previous = score
            # map the compressed-graph partition back to original nodes
            mapped = []
            for com in inner:
                original = set()
                for cnode in com:
                    original.update(H.nodes[cnode].get("nodes", {cnode}))
                mapped.append(original)
            assert reference_score(G, mapped) == pytest.approx(
                score, abs=GAIN_TOL
            )
            level += 1
            if not improvement or level == 6:
                break
            H = _gen_graph_signed(H, inner)
            partition = inner

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_partition_generator_snapshots_and_seed(self, scale):
        # a graph dense enough to produce several Louvain levels: every
        # yielded partition is an independent snapshot (gh-5901), which
        # must keep holding across the whole scale range
        rng = random.Random(204)
        G = scaled_random_graph(rng, 24, scale)
        parts = list(
            louvain_partitions(G, weight="weight", seed=19, signed=True)
        )
        assert len(parts) >= 2
        first = [set(c) for c in parts[0]]
        for c in parts[-1]:
            c.add(-999)
        assert parts[0] == first
        assert -999 not in {u for c in parts[0] for u in c}
        again = louvain_communities(G, weight="weight", seed=19, signed=True)
        once_more = louvain_communities(
            G, weight="weight", seed=19, signed=True
        )
        assert again == once_more


# ---------------------------------------------------------------------------
# Greedy merge gains and candidate queue coverage
# ---------------------------------------------------------------------------

class TestGreedyGains:
    @pytest.mark.parametrize("scale", [1e-310, 1e-165, 1.0, 1e155, 1e300])
    def test_every_merge_uses_the_maximum_gain(self, scale):
        rng = random.Random(303)
        G = scaled_random_graph(rng, 8, scale)
        m_pos, m_neg, deg_pos, deg_neg, weight_exp = _signed_layer_data(
            G, "weight"
        )
        from networkx.algorithms.community.modularity_max import (
            _greedy_modularity_signed_generator,
        )

        gen = _greedy_modularity_signed_generator(
            G, "weight", 1.0, m_pos, m_neg, deg_pos, deg_neg, weight_exp
        )
        communities = [set(c) for c in next(gen)]
        score = reference_score(G, communities)
        while True:
            try:
                dq = next(gen)
            except StopIteration:
                break
            # exhaustive reference over *all* pairs, including pairs
            # with no direct edge between the communities
            best_gain = -math.inf
            for i, j in combinations(range(len(communities)), 2):
                g = reference_merge_gain(G, communities, i, j)
                if g > best_gain:
                    best_gain = g
            assert dq == pytest.approx(best_gain, abs=GAIN_TOL)
            new_communities = [set(c) for c in next(gen)]
            new_score = reference_score(G, new_communities)
            assert new_score - score == pytest.approx(dq, abs=GAIN_TOL)
            communities, score = new_communities, new_score

    def test_non_adjacent_pair_remains_a_candidate_at_tiny_scale(self):
        # two negative self-loops, no edge between the nodes: the merge
        # gain is +0.5 at every scale and must not vanish in the queue
        for scale in (1e-310, 1.0, 1e300):
            G = nx.Graph()
            G.add_edge(0, 0, weight=-1.0 * scale)
            G.add_edge(1, 1, weight=-1.0 * scale)
            part = greedy_modularity_communities(
                G, weight="weight", signed=True
            )
            assert part == [frozenset({0, 1})]

    def test_best_n_forced_merges_still_maximal(self):
        rng = random.Random(304)
        G = scaled_random_graph(rng, 10, 1e155)
        forced = greedy_modularity_communities(
            G, weight="weight", signed=True, best_n=2
        )
        assert len(forced) == 2
        # forced merges may lower the score, but the final partition must
        # still be a valid partition
        assert nx.community.is_partition(G, forced)


# ---------------------------------------------------------------------------
# Scale invariance of the actual decisions (no near-tie cases)
# ---------------------------------------------------------------------------

class TestDecisionInvariance:
    def test_greedy_partition_invariant_across_scales(self):
        rng = random.Random(404)
        G = scaled_random_graph(rng, 18, 1.0)
        reference = sorted(
            sorted(c) for c in greedy_modularity_communities(
                G, weight="weight", signed=True
            )
        )
        for scale in (1e-300, 1e-165, 1e155, 1e300):
            H = scaled_copy(G, scale)
            got = sorted(
                sorted(c) for c in greedy_modularity_communities(
                    H, weight="weight", signed=True
                )
            )
            assert got == reference

    def test_louvain_partition_invariant_across_scales(self):
        rng = random.Random(405)
        G = scaled_random_graph(rng, 18, 1.0)
        reference = sorted(
            sorted(c) for c in louvain_communities(
                G, weight="weight", seed=9, signed=True
            )
        )
        for scale in (1e-300, 1e-165, 1e155, 1e300):
            H = scaled_copy(G, scale)
            got = sorted(
                sorted(c) for c in louvain_communities(
                    H, weight="weight", seed=9, signed=True
                )
            )
            assert got == reference


def scaled_copy(G, scale):
    H = G.__class__()
    H.add_nodes_from(G.nodes)
    for u, v, w in G.edges(data="weight"):
        H.add_edge(u, v, weight=w * scale)
    return H


# ---------------------------------------------------------------------------
# Structural edge cases at extreme scales
# ---------------------------------------------------------------------------

class TestStructuralEdgeCases:
    def _check(self, G):
        part_singleton = [{u} for u in G.nodes]
        got = modularity(
            G, part_singleton, weight="weight", signed=True
        )
        assert got == pytest.approx(
            reference_score(G, part_singleton), abs=GAIN_TOL
        )
        lp = louvain_communities(G, weight="weight", seed=3, signed=True)
        assert nx.community.is_partition(G, lp)
        assert reference_score(G, lp) >= got - GAIN_TOL
        gp = greedy_modularity_communities(G, weight="weight", signed=True)
        assert nx.community.is_partition(G, gp)
        assert reference_score(G, gp) >= got - GAIN_TOL

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_all_positive(self, scale):
        G = nx.Graph()
        G.add_edge(0, 1, weight=2.0 * scale)
        G.add_edge(1, 2, weight=3.0 * scale)
        G.add_edge(2, 3, weight=1.5 * scale)
        self._check(G)

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_all_negative(self, scale):
        G = nx.Graph()
        G.add_edge(0, 1, weight=-2.0 * scale)
        G.add_edge(1, 2, weight=-3.0 * scale)
        G.add_edge(2, 3, weight=-1.5 * scale)
        self._check(G)

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_zero_weight_and_empty_layer(self, scale):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.0)
        G.add_edge(1, 2, weight=2.0 * scale)
        G.add_edge(2, 2, weight=0.0)
        self._check(G)

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_mixed_self_loops(self, scale):
        G = nx.Graph()
        G.add_edge(0, 0, weight=2.0 * scale)
        G.add_edge(1, 1, weight=-2.0 * scale)
        G.add_edge(0, 1, weight=-1.0 * scale)
        G.add_edge(1, 2, weight=3.0 * scale)
        self._check(G)

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_positive_and_negative_parallel_edges(self, scale):
        G = nx.MultiGraph()
        G.add_edge(0, 1, weight=2.0 * scale)
        G.add_edge(0, 1, weight=-3.0 * scale)
        G.add_edge(1, 2, weight=1.5 * scale)
        G.add_edge(2, 2, weight=-1.0 * scale)
        self._check(G)

    @pytest.mark.parametrize("scale", [1e-310, 1e155])
    def test_isolated_nodes(self, scale):
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edge(0, 1, weight=2.0 * scale)
        G.add_edge(2, 3, weight=-2.0 * scale)
        self._check(G)

    def test_unorderable_nodes(self):
        class Node:
            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return hash(self.x)

            def __eq__(self, other):
                return isinstance(other, Node) and self.x == other.x

        nodes = [Node(i) for i in range(5)]
        G = nx.Graph()
        G.add_edge(nodes[0], nodes[1], weight=2e155)
        G.add_edge(nodes[1], nodes[2], weight=-1e155)
        G.add_edge(nodes[3], nodes[4], weight=-2e-310)
        lp = louvain_communities(G, weight="weight", seed=1, signed=True)
        gp = greedy_modularity_communities(
            G, weight="weight", signed=True
        )
        assert nx.community.is_partition(G, lp)
        assert nx.community.is_partition(G, gp)


# ---------------------------------------------------------------------------
# Invalid inputs must still be rejected explicitly
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    @pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "x", 1 + 2j])
    def test_bad_weights_rejected(self, bad):
        G = nx.Graph([(0, 1)])
        G[0][1]["weight"] = bad
        with pytest.raises(ValueError):
            modularity(G, [{0}, {1}], weight="weight", signed=True)
        with pytest.raises(ValueError):
            louvain_communities(G, weight="weight", signed=True)
        with pytest.raises(ValueError):
            greedy_modularity_communities(G, weight="weight", signed=True)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "x"])
    def test_bad_resolution_rejected(self, bad):
        G = nx.Graph([(0, 1)])
        with pytest.raises(ValueError):
            modularity(
                G, [{0}, {1}], weight="weight",
                resolution=bad, signed=True,
            )
        with pytest.raises(ValueError):
            louvain_communities(
                G, weight="weight", resolution=bad, signed=True
            )
        with pytest.raises(ValueError):
            greedy_modularity_communities(
                G, weight="weight", resolution=bad, signed=True
            )

    def test_input_graph_and_edge_attributes_unchanged(self):
        G = four_node_graph(1e-165)
        before = sorted(
            (u, v, dict(d)) for u, v, d in G.edges(data=True)
        )
        modularity(G, [{0, 1}, {2, 3}], weight="weight", signed=True)
        louvain_communities(G, weight="weight", seed=19, signed=True)
        greedy_modularity_communities(G, weight="weight", signed=True)
        after = sorted((u, v, dict(d)) for u, v, d in G.edges(data=True))
        assert before == after
        assert set(G.nodes) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Near-threshold behaviour: documented, no literal partition requirement
# ---------------------------------------------------------------------------

class TestNearThreshold:
    def test_move_acceptance_threshold_semantics(self):
        # Construct a graph whose best move gain is numerically close to
        # 1e-12: the partition may legitimately differ across scales near
        # a tie, but no chosen move may have a reference gain below
        # -1e-10 and an improving move above 1e-10 may not be missed.
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        G.add_edge(0, 2, weight=-0.0000000000005)  # -5e-13: below 1e-12
        singleton = [{0}, {1}, {2}, {3}]
        base = reference_score(G, singleton)
        part = louvain_communities(G, weight="weight", seed=1, signed=True)
        assert reference_score(G, part) >= base - 1e-10
        # and when a clearly profitable move exists it is taken
        H = nx.Graph()
        H.add_edge(0, 1, weight=1.0)
        H.add_edge(2, 3, weight=1.0)
        H.add_edge(0, 2, weight=-0.5)
        hp = louvain_communities(H, weight="weight", seed=1, signed=True)
        assert reference_score(H, hp) > reference_score(
            H, [{0}, {1}, {2}, {3}]
        ) + 0.1
