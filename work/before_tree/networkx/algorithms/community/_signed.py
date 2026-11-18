"""Shared helpers for signed-network community detection (private module).

This module implements the two-layer (positive/negative) machinery behind
the ``signed=True`` keyword of

- :func:`~networkx.algorithms.community.quality.modularity`
- :func:`~networkx.algorithms.community.louvain.louvain_partitions`
- :func:`~networkx.algorithms.community.louvain.louvain_communities`
- :func:`~networkx.algorithms.community.modularity_max.greedy_modularity_communities`

Scoring convention
------------------
Every edge weight is split into a *positive layer* contribution
``w_pos = max(w, 0)`` and a *negative layer* contribution ``w_neg =
max(-w, 0)``; zero-weight edges contribute to neither layer.  Let
``m_pos``/``m_neg`` be the total edge mass of the two layers (undirected
self-loops counted once) and ``Q_pos``/``Q_neg`` the usual undirected
weighted modularity of a partition computed on each layer separately
(with the same ``resolution``; self-loops count twice in the degrees).
The signed score of a partition is then

.. math::

    Q_{signed} = \\frac{m_{pos} \\, Q_{pos} - m_{neg} \\, Q_{neg}}{m_{pos} + m_{neg}}

If one layer has no mass its term is omitted; if neither layer has any
mass the score is 0.  The score rewards placing positive (support)
edges inside communities while keeping negative (rejection) edges
between them.  Graphs with only positive weights degenerate to the
usual modularity.

The two layers are kept separate everywhere: per-pair positive and
negative weights are preserved through Louvain's graph compression, so
the score of any partition of a compressed graph equals the score of
the corresponding partition of the original graph.

Benchmarking hook
-----------------
The module-level variable ``_EVAL_STATS`` may be set to a mutable
mapping (e.g. ``collections.Counter``); the signed optimizers then
count their candidate/gain evaluations under string keys.  It is
``None`` (disabled) by default and intended only for benchmarks.
"""

import math
from collections import defaultdict
from numbers import Real

import networkx as nx
from networkx.exception import NetworkXNotImplemented

__all__ = []  # private module

#: Optional mutable mapping used to count candidate/gain evaluations.
_EVAL_STATS = None


def _check_signed_graph(G):
    """Raise if ``G`` is not supported in signed mode (only undirected)."""
    if G.is_directed():
        raise NetworkXNotImplemented(
            "signed modularity is not implemented for directed graphs"
        )


def _validate_signed_weight(w):
    """Return ``w`` as a float after checking the signed-mode weight rules.

    Signed mode requires finite, non-bool real weights; negative, positive
    and zero values are all accepted.
    """
    if isinstance(w, bool) or not isinstance(w, Real):
        raise ValueError(
            "signed modularity requires finite real (non-bool) edge weights,"
            f" got {w!r}"
        )
    try:
        w = float(w)
    except OverflowError as err:
        raise ValueError(
            f"signed modularity edge weight is too large to represent: {w!r}"
        ) from err
    if not math.isfinite(w):
        raise ValueError(f"signed modularity requires finite edge weights, got {w!r}")
    return w


def _validate_signed_resolution(resolution):
    """Return ``resolution`` as a float after checking it is finite and >= 0."""
    if isinstance(resolution, bool) or not isinstance(resolution, Real):
        raise ValueError(
            "signed modularity requires a finite non-negative real"
            f" resolution, got {resolution!r}"
        )
    try:
        resolution = float(resolution)
    except OverflowError as err:
        raise ValueError(
            f"resolution is too large to represent: {resolution!r}"
        ) from err
    if not math.isfinite(resolution) or resolution < 0:
        raise ValueError(
            "signed modularity requires a finite non-negative real"
            f" resolution, got {resolution!r}"
        )
    return resolution


def _signed_layer_data(G, weight):
    """Split ``G`` into positive/negative degree and mass summaries.

    Returns ``(m_pos, m_neg, deg_pos, deg_neg)`` where ``m_pos``/``m_neg``
    are the total layer masses (self-loops counted once) and
    ``deg_pos``/``deg_neg`` are dicts mapping each node to its
    positive/negative weighted degree (self-loops counted twice, as usual
    for undirected graphs).  Every edge weight is validated; a
    :exc:`ValueError` is raised for invalid weights or if an accumulated
    mass overflows the floating-point range.
    """
    deg_pos = defaultdict(float)
    deg_neg = defaultdict(float)
    m_pos = 0.0
    m_neg = 0.0
    for u, v, w in G.edges(data=weight, default=1):
        w = _validate_signed_weight(w)
        if w > 0:
            m_pos += w
            deg_pos[u] += w
            deg_pos[v] += w
        elif w < 0:
            m_neg -= w
            deg_neg[u] -= w
            deg_neg[v] -= w
        # zero weights contribute no mass to either layer
    if not (math.isfinite(m_pos) and math.isfinite(m_neg)):
        raise ValueError(
            "signed modularity total edge mass overflowed the floating-point range"
        )
    return m_pos, m_neg, deg_pos, deg_neg


def _signed_layer_graph(G, weight):
    """Build a two-layer simple graph from ``G``.

    Returns ``(H, m_pos, m_neg)`` where ``H`` is a :class:`nx.Graph` on the
    same nodes whose edges carry the attributes ``"weight_pos"`` and
    ``"weight_neg"``: the summed positive and (absolute) negative weights
    of all parallel edges between the pair.  Both contributions are kept;
    they are never netted against each other.  Pairs whose total
    contribution is zero in both layers carry no information and are
    dropped.  ``m_pos``/``m_neg`` are the total layer masses (self-loops
    counted once), checked to be finite.
    """
    H = nx.Graph()
    H.add_nodes_from(G)
    m_pos = 0.0
    m_neg = 0.0
    for u, v, w in G.edges(data=weight, default=1):
        w = _validate_signed_weight(w)
        if w > 0:
            m_pos += w
            if H.has_edge(u, v):
                H[u][v]["weight_pos"] += w
            else:
                H.add_edge(u, v, weight_pos=w, weight_neg=0.0)
        elif w < 0:
            m_neg -= w
            if H.has_edge(u, v):
                H[u][v]["weight_neg"] -= w
            else:
                H.add_edge(u, v, weight_pos=0.0, weight_neg=-w)
        # zero weights contribute no mass to either layer
    if not (math.isfinite(m_pos) and math.isfinite(m_neg)):
        raise ValueError(
            "signed modularity total edge mass overflowed the floating-point range"
        )
    return H, m_pos, m_neg


def _signed_modularity_value(m_pos, m_neg, resolution, stats):
    """Evaluate the signed score from per-community layer aggregates.

    ``stats`` is an iterable of ``(L_pos, L_neg, K_pos, K_neg)`` tuples,
    one per community: ``L_pos``/``L_neg`` are the intra-community layer
    weights (each edge counted once) and ``K_pos``/``K_neg`` are the
    community sums of the layer degrees (self-loops counted twice).
    ``m_pos``/``m_neg`` are the total layer masses.  This computes
    ``(m_pos * Q_pos - m_neg * Q_neg) / (m_pos + m_neg)`` in the
    algebraically equivalent "mass-weighted" form

        sum_c [L_pos_c - L_neg_c
               - gamma * K_pos_c^2 / (4 m_pos)
               + gamma * K_neg_c^2 / (4 m_neg)] / (m_pos + m_neg)

    where a layer term is omitted when the corresponding mass is zero,
    and the score is 0 when both masses are zero.
    """
    m_tot = m_pos + m_neg
    if m_tot == 0:
        return 0.0
    num = 0.0
    for L_pos, L_neg, K_pos, K_neg in stats:
        term = L_pos - L_neg
        if m_pos:
            term -= resolution * K_pos * K_pos / (4 * m_pos)
        if m_neg:
            term += resolution * K_neg * K_neg / (4 * m_neg)
        num += term
    result = num / m_tot
    if not math.isfinite(result):
        raise ValueError("signed modularity overflowed the floating-point range")
    return result


def _signed_modularity(G, communities, weight, resolution):
    """Signed modularity of a valid partition of an undirected graph.

    This is a single pass over the edges of ``G``; it does not copy the
    graph.  ``communities`` must be a valid partition (checked by the
    caller).  See the module docstring for the scoring convention.
    """
    _check_signed_graph(G)
    resolution = _validate_signed_resolution(resolution)
    node2com = {}
    for i, community in enumerate(communities):
        for u in community:
            node2com[u] = i
    n_com = len(communities)
    L_pos = [0.0] * n_com
    L_neg = [0.0] * n_com
    deg_pos = defaultdict(float)
    deg_neg = defaultdict(float)
    m_pos = 0.0
    m_neg = 0.0
    for u, v, w in G.edges(data=weight, default=1):
        w = _validate_signed_weight(w)
        if w > 0:
            m_pos += w
            deg_pos[u] += w
            deg_pos[v] += w
            if node2com[u] == node2com[v]:
                L_pos[node2com[u]] += w
        elif w < 0:
            m_neg -= w
            deg_neg[u] -= w
            deg_neg[v] -= w
            if node2com[u] == node2com[v]:
                L_neg[node2com[u]] -= w
        # zero weights contribute no mass to either layer
    if not (math.isfinite(m_pos) and math.isfinite(m_neg)):
        raise ValueError(
            "signed modularity total edge mass overflowed the floating-point range"
        )
    K_pos = [0.0] * n_com
    K_neg = [0.0] * n_com
    for u, c in node2com.items():
        K_pos[c] += deg_pos[u]
        K_neg[c] += deg_neg[u]
    return _signed_modularity_value(
        m_pos, m_neg, resolution, zip(L_pos, L_neg, K_pos, K_neg)
    )


def _signed_modularity_reduced(G, communities, m_pos, m_neg, resolution):
    """Signed modularity of a partition of a two-layer graph.

    ``G`` is a graph built by :func:`_signed_layer_graph` (or compressed
    by ``_gen_graph_signed``) whose edges carry ``"weight_pos"`` and
    ``"weight_neg"`` attributes; ``m_pos``/``m_neg`` are the (already
    validated) total layer masses.  Because compression preserves both
    layers and the masses, the result equals the signed modularity of
    the corresponding partition of the original graph.
    """
    node2com = {}
    for i, community in enumerate(communities):
        for u in community:
            node2com[u] = i
    n_com = len(communities)
    L_pos = [0.0] * n_com
    L_neg = [0.0] * n_com
    for u, v, data in G.edges(data=True):
        c = node2com[u]
        if c == node2com[v]:
            L_pos[c] += data["weight_pos"]
            L_neg[c] += data["weight_neg"]
    deg_pos = dict(G.degree(weight="weight_pos"))
    deg_neg = dict(G.degree(weight="weight_neg"))
    K_pos = [0.0] * n_com
    K_neg = [0.0] * n_com
    for u, c in node2com.items():
        K_pos[c] += deg_pos[u]
        K_neg[c] += deg_neg[u]
    return _signed_modularity_value(
        m_pos, m_neg, resolution, zip(L_pos, L_neg, K_pos, K_neg)
    )
