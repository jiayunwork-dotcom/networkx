"""Functions for detecting communities based on modularity."""

import math
from collections import defaultdict

import networkx as nx
from networkx.algorithms.community import _signed
from networkx.algorithms.community._signed import (
    _check_signed_graph,
    _signed_layer_data,
    _validate_signed_resolution,
)
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue

__all__ = [
    "greedy_modularity_communities",
    "naive_greedy_modularity_communities",
]


def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    r"""Yield community partitions of G and the modularity change at each step.

    This function performs Clauset-Newman-Moore greedy modularity maximization [2]_
    At each step of the process it yields the change in modularity that will occur in
    the next step followed by yielding the new community partition after that step.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until one community contains all nodes (the partition has one set).

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Yields
    ------
    Alternating yield statements produce the following two objects:

    communities: dict_values
        A dict_values of frozensets of nodes, one for each community.
        This represents a partition of the nodes of the graph into communities.
        The first yield is the partition with each node in its own community.

    dq: float
        The change in modularity when merging the next two communities
        that leads to the largest modularity.

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    directed = G.is_directed()
    N = G.number_of_nodes()

    # Count edges (or the sum of edge-weights for weighted graphs)
    m = G.size(weight)
    q0 = 1 / m

    # Calculate degrees (notation from the papers)
    # a : the fraction of (weighted) out-degree for each node
    # b : the fraction of (weighted) in-degree for each node
    if directed:
        a = {node: deg_out * q0 for node, deg_out in G.out_degree(weight=weight)}
        b = {node: deg_in * q0 for node, deg_in in G.in_degree(weight=weight)}
    else:
        a = b = {node: deg * q0 * 0.5 for node, deg in G.degree(weight=weight)}

    # this preliminary step collects the edge weights for each node pair
    # It handles multigraph and digraph and works fine for graph.
    dq_dict = defaultdict(lambda: defaultdict(float))
    for u, v, wt in G.edges(data=weight, default=1):
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt

    # now scale and subtract the expected edge-weights term
    for u, nbrdict in dq_dict.items():
        for v, wt in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])

    # Use -dq to get a max_heap instead of a min_heap
    # dq_heap holds a heap for each node's neighbors
    dq_heap = {u: MappedQueue({(u, v): -dq for v, dq in dq_dict[u].items()}) for u in G}
    # H -> all_dq_heap holds a heap with the best items for each node
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])

    # Initialize single-node communities
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()

    # Merge the two communities that lead to the largest modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        # Remove best merge from row u heap
        dq_heap[u].pop()
        # Push new row max onto H
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        # If this element was also at the root of row v, we need to remove the
        # duplicate entry from H
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            # Remove best merge from row v heap
            dq_heap[v].remove((v, u))
            # Push new row max onto H
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            # Duplicate wasn't in H, just remove from row v heap
            dq_heap[v].remove((v, u))

        # Perform merge
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]

        # Get neighbor communities connected to the merged communities
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        # Update dq for merge of u into v
        for w in all_nbrs:
            # Calculate new dq value
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:  # w in u_nbrs
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            # Update rows v and w
            for row, col in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                # Update dict for v,w only (u is removed below)
                dq_dict[row][col] = dq_vw
                # Save old max of per-row heap
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (row, col)
                d_negdq = -dq_vw
                # Save old value for finding heap index
                if w in v_nbrs:
                    # Update existing element in per-row heap
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    # We're creating a new nonzero element, add to heap
                    dq_heap_row.push(d, priority=d_negdq)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d, priority=d_negdq)
                else:
                    # We've updated an entry in this row, has the max changed?
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)

        # Remove row/col u from dq_dict matrix
        for w in dq_dict[u]:
            # Remove from dict
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            # Remove from heaps if we haven't already
            if w != v:
                # Remove both row and column
                for row, col in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    # Check if replaced dq is row max
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        # Only update per-row heap
                        dq_heap_row.remove(d_old)

        del dq_dict[u]
        # Mark row u as deleted, but keep placeholder
        dq_heap[u] = MappedQueue()
        # Merge u into v and update a
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0

        yield communities.values()


@nx._dispatchable(edge_attrs="weight")
def greedy_modularity_communities(
    G,
    weight=None,
    resolution=1,
    cutoff=1,
    best_n=None,
    *,
    signed=False,
):
    r"""Find communities in G using greedy modularity maximization.

    This function uses Clauset-Newman-Moore greedy modularity maximization [2]_
    to find the community partition with the largest modularity.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until no further increase in modularity is possible (a maximum).
    Two keyword arguments adjust the stopping condition. `cutoff` is a lower
    limit on the number of communities so you can stop the process before
    reaching a maximum (used to save computation time). `best_n` is an upper
    limit on the number of communities so you can make the process continue
    until at most n communities remain even if the maximum modularity occurs
    for more. To obtain exactly n communities, set both `cutoff` and `best_n` to n.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float, optional (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    cutoff : int, optional (default=1)
        A minimum number of communities below which the merging process stops.
        The process stops at this number of communities even if modularity
        is not maximized. The goal is to let the user stop the process early.
        The process stops before the cutoff if it finds a maximum of modularity.

    best_n : int or None, optional (default=None)
        A maximum number of communities above which the merging process will
        not stop. This forces community merging to continue after modularity
        starts to decrease until `best_n` communities remain.
        If ``None``, don't force it to continue beyond a maximum.

    signed : bool, optional (default=False)
        If True, maximize the signed two-layer modularity (see
        :func:`~networkx.algorithms.community.quality.modularity`) where
        negative edge weights repel communities. Only implemented for
        undirected graphs. In signed mode every pair of communities is a
        merge candidate -- including pairs with no edges between them,
        which can still have positive gain -- and merging stops when no
        merge gain exceeds 1e-12 (unless `best_n` forces it to continue,
        in which case the largest-gain merge is still chosen).

    Raises
    ------
    ValueError : If the `cutoff` or `best_n`  value is not in the range
        ``[1, G.number_of_nodes()]``, or if `best_n` < `cutoff`.

    NetworkXNotImplemented
        If `signed` is True and `G` is directed.

    ValueError
        If `signed` is True and a weight or `resolution` is not a finite
        real number (weights must also not be boolean), if `resolution`
        is negative, or if an accumulated layer mass overflows the
        floating-point range.

    Returns
    -------
    communities: list
        A list of frozensets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    if signed:
        return _greedy_modularity_signed(G, weight, resolution, cutoff, best_n)

    if not G.size():
        return [{n} for n in G]

    if (cutoff < 1) or (cutoff > G.number_of_nodes()):
        raise ValueError(f"cutoff must be between 1 and {len(G)}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > G.number_of_nodes()):
            raise ValueError(f"best_n must be between 1 and {len(G)}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()

    # retrieve generator object to construct output
    community_gen = _greedy_modularity_communities_generator(
        G, weight=weight, resolution=resolution
    )

    # construct the first best community
    communities = next(community_gen)

    # continue merging communities until one of the breaking criteria is satisfied
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        # StopIteration occurs when communities are the connected components
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            # if best_n requires more merging, merge big sets for highest modularity
            while len(communities) > best_n:
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities

        # keep going unless max_mod is reached or best_n says to merge more
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)

    return sorted(communities, key=len, reverse=True)


def _greedy_modularity_signed(G, weight, resolution, cutoff, best_n):
    """Find communities in ``G`` using greedy signed-modularity maximization.

    This is the ``signed=True`` path of
    :func:`greedy_modularity_communities`; see its docstring for the
    scoring convention and parameter semantics.
    """
    _check_signed_graph(G)
    resolution = _validate_signed_resolution(resolution)
    N = G.number_of_nodes()
    if N == 0:
        return []

    if (cutoff < 1) or (cutoff > N):
        raise ValueError(f"cutoff must be between 1 and {N}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > N):
            raise ValueError(f"best_n must be between 1 and {N}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
    else:
        best_n = N

    # validate all weights and accumulate the two layer masses/degrees
    m_pos, m_neg, deg_pos, deg_neg = _signed_layer_data(G, weight)

    if best_n == 1:
        return [set(G)]

    # retrieve generator object to construct output
    community_gen = _greedy_modularity_signed_generator(
        G, weight, resolution, m_pos, m_neg, deg_pos, deg_neg
    )

    # construct the first best community
    communities = next(community_gen)

    # continue merging communities until one of the breaking criteria is
    # satisfied
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        # StopIteration occurs when a single community remains
        except StopIteration:
            break

        # keep going unless the gain threshold is reached or best_n says
        # to merge more
        if dq <= 1e-12 and len(communities) <= best_n:
            break
        communities = next(community_gen)

    return sorted(communities, key=len, reverse=True)


def _greedy_modularity_signed_generator(
    G, weight, resolution, m_pos, m_neg, deg_pos, deg_neg
):
    r"""Yield partitions and signed-modularity changes at each merge step.

    Clauset-Newman-Moore greedy maximization of the signed two-layer
    modularity (see :func:`greedy_modularity_communities` with
    ``signed=True``).  ``m_pos``/``m_neg`` and ``deg_pos``/``deg_neg`` are
    the validated layer masses and per-node layer degrees from
    :func:`~networkx.algorithms.community._signed._signed_layer_data`.

    The gain of merging communities ``u`` and ``v`` is

        dq(u, v) = w(u, v) / m_tot
                   - a_pos * K_pos_u * K_pos_v + a_neg * K_neg_u * K_neg_v

    where ``w(u, v)`` is the net (positive minus negative) weight between
    the communities, ``K_pos``/``K_neg`` are the community layer-degree
    sums, ``m_tot = m_pos + m_neg`` and ``a_pos = gamma / (2 m_pos
    m_tot)``, ``a_neg = gamma / (2 m_neg m_tot)`` (a layer term is dropped
    when its mass is zero; all gains are 0 when both are zero).

    Candidate coverage: adjacent community pairs are tracked with the
    usual CNM per-row heaps (``dq_dict``/``dq_heap``/``H``, memory
    proportional to the number of inter-community edges).  Pairs with no
    edges between them can still have positive gain in signed networks
    (their gain depends only on the two communities' degree sums), so
    each community additionally caches its best non-adjacent partner in
    ``na_best`` (one entry per community, no all-pairs gain table) with
    the global maxima in ``na_H``.  Entries of ``na_best`` are either
    exact or lazy upper bounds; upper bounds are recomputed exactly (an
    O(#communities) scan) only when they reach the root of ``na_H`` and
    can still beat the best adjacent-pair gain, which keeps the number
    of exact evaluations small without ever missing the best pair.

    Yields
    ------
    Alternating yield statements produce the following two objects:

    communities: dict_values
        A dict_values of frozensets of nodes, one for each community.
        The first yield is the partition with each node in its own
        community.

    dq: float
        The change in signed modularity when merging the next two
        communities that leads to the largest signed modularity.
    """
    stats = _signed._EVAL_STATS
    nodes = list(G)
    N = len(nodes)
    node_idx = {u: i for i, u in enumerate(nodes)}
    m_tot = m_pos + m_neg
    inv_mtot = 1.0 / m_tot if m_tot else 0.0
    a_pos = resolution / (2 * m_pos * m_tot) if m_pos else 0.0
    a_neg = resolution / (2 * m_neg * m_tot) if m_neg else 0.0

    # community id == index into these lists; communities are keyed by
    # integers so that no ordering of the (possibly unorderable) nodes is
    # ever needed
    k_pos = [0.0] * N
    k_neg = [0.0] * N
    for u, i in node_idx.items():
        k_pos[i] = deg_pos[u]
        k_neg[i] = deg_neg[u]

    # net weight per adjacent community pair (both directions); the two
    # layers enter the gain through this net weight and, separately,
    # through the layer degrees and masses above
    dq_dict = defaultdict(lambda: defaultdict(float))
    for u, v, w in G.edges(data=weight, default=1):
        if u == v:
            continue
        w = float(w)  # already validated by _signed_layer_data
        if w == 0:
            continue
        i, j = node_idx[u], node_idx[v]
        dq_dict[i][j] += w
        dq_dict[j][i] += w
    # convert net weights into gains
    for i, nbrdict in dq_dict.items():
        kpi, kni = k_pos[i], k_neg[i]
        for j, w in nbrdict.items():
            nbrdict[j] = w * inv_mtot - a_pos * kpi * k_pos[j] + a_neg * kni * k_neg[j]

    # per-row heaps of adjacent-pair gains, plus a heap of row maxima
    dq_heap = {
        i: MappedQueue({(i, j): -dq for j, dq in dq_dict[i].items()})
        if i in dq_dict
        else MappedQueue()
        for i in range(N)
    }
    H = MappedQueue()
    for i in range(N):
        if len(dq_heap[i]) > 0:
            H.push(i, priority=dq_heap[i].heap[0].priority)

    communities = {i: frozenset([nodes[i]]) for i in range(N)}

    # Non-adjacent candidates.  na_best[c] = [value, partner, exact]:
    # ``value`` is either the exact best gain of community ``c`` over its
    # non-adjacent partners (exact=True) or an upper bound on it
    # (exact=False).  na_H is keyed by -value, so its root is the largest
    # exact gain or upper bound.  Stale (upper-bound) entries are
    # recomputed exactly only when they reach the root and can still beat
    # the best adjacent-pair gain; recomputation is an O(#communities)
    # scan.  This keeps memory at one entry per community (no all-pairs
    # gain table) and avoids most exact evaluations, at the price of
    # occasionally recomputing a best partner from scratch.
    na_best = {}
    na_H = MappedQueue()

    def heap_set(q, elt, priority):
        """Push or update ``elt`` in a MappedQueue."""
        try:
            q.update(elt, elt, priority=priority)
        except KeyError:
            q.push(elt, priority=priority)

    def heap_discard(q, elt):
        """Remove ``elt`` from a MappedQueue if present."""
        try:
            q.remove(elt)
        except KeyError:
            pass

    def row_sync(i):
        """Reflect row ``i`` of the adjacent-pair heaps in the max heap H."""
        if len(dq_heap[i]) > 0:
            heap_set(H, i, dq_heap[i].heap[0].priority)
        else:
            heap_discard(H, i)

    def na_set(c, value, partner, exact):
        na_best[c] = [value, partner, exact]
        heap_set(na_H, c, -value)

    def na_drop(c):
        na_best.pop(c, None)
        heap_discard(na_H, c)

    def gain_na(c, d):
        return a_neg * k_neg[c] * k_neg[d] - a_pos * k_pos[c] * k_pos[d]

    def na_rescan(c):
        """Recompute the best non-adjacent partner of community ``c``."""
        adj = dq_dict.get(c)
        kpc = k_pos[c]
        knc = k_neg[c]
        best_g = None
        best_d = -1
        n_evals = 0
        for d in communities:
            if d == c or (adj is not None and d in adj):
                continue
            n_evals += 1
            g = a_neg * knc * k_neg[d] - a_pos * kpc * k_pos[d]
            if best_g is None or g > best_g:
                best_g = g
                best_d = d
        if stats is not None:
            stats["greedy_na_gain_evals"] += n_evals
        if best_g is None:
            na_drop(c)
        else:
            na_set(c, best_g, best_d, True)

    def merge(u, v):
        """Merge community ``u`` into community ``v``."""
        nonlocal kn_max
        # --- update the adjacent-pair gains (CNM-style) ---
        u_nbrs = dq_dict.get(u, {})
        v_nbrs = dq_dict[v]  # auto-creates the row if v had no neighbors
        if v in u_nbrs:
            # remove the merged pair itself
            del u_nbrs[v]
            del v_nbrs[u]
            dq_heap[u].remove((u, v))
            dq_heap[v].remove((v, u))
        kpu, knu = k_pos[u], k_neg[u]
        kpv, knv = k_pos[v], k_neg[v]
        n_evals = 0
        for w in set(u_nbrs) | set(v_nbrs):
            in_u = w in u_nbrs
            in_v = w in v_nbrs
            if in_u and in_v:
                dq_vw = v_nbrs[w] + u_nbrs[w]
            elif in_u:
                # dq(u, w) - phi(v, w) with pre-merge degrees of v
                dq_vw = u_nbrs[w] - (a_pos * kpv * k_pos[w] - a_neg * knv * k_neg[w])
            else:
                # dq(v, w) - phi(u, w)
                dq_vw = v_nbrs[w] - (a_pos * kpu * k_pos[w] - a_neg * knu * k_neg[w])
            n_evals += 1
            # write the new gain into rows v and w
            v_nbrs[w] = dq_vw
            dq_dict[w][v] = dq_vw
            heap_set(dq_heap[v], (v, w), -dq_vw)
            heap_set(dq_heap[w], (w, v), -dq_vw)
            if in_u:
                del u_nbrs[w]
                del dq_dict[w][u]
                heap_discard(dq_heap[w], (w, u))
            row_sync(w)
        if stats is not None:
            stats["greedy_merge_gain_evals"] += n_evals
        # remove row u entirely
        dq_dict.pop(u, None)
        dq_heap[u] = MappedQueue()
        heap_discard(H, u)
        row_sync(v)
        # --- layer degrees of the merged community ---
        k_pos[v] = kpv + kpu
        k_neg[v] = knv + knu
        k_pos[u] = 0.0
        k_neg[u] = 0.0
        if k_neg[v] > kn_max:
            kn_max = k_neg[v]
        # --- the merged community set ---
        communities[v] = communities[u] | communities[v]
        del communities[u]
        # --- refresh the non-adjacent candidates (lazily) ---
        na_drop(u)
        # v's own best partner is recomputed lazily from an upper bound
        na_set(v, a_neg * k_neg[v] * kn_max, None, False)
        for c in communities:
            if c == v:
                continue
            entry = na_best.get(c)
            if v in dq_dict.get(c, ()):
                # (c, v) is adjacent: not a non-adjacent candidate.  An
                # exact entry whose partner was merged degrades to a bound.
                if entry is not None and entry[2] and entry[1] in (u, v):
                    na_set(c, entry[0], None, False)
                continue
            g = gain_na(c, v)
            if stats is not None:
                stats["greedy_na_gain_evals"] += 1
            if entry is None:
                # c had no non-adjacent candidate; v is its only one
                na_set(c, g, v, True)
            elif not entry[2]:
                # stale entry: keep the larger upper bound
                if g > entry[0]:
                    na_set(c, g, None, False)
            elif entry[1] in (u, v):
                # c's best partner was just merged
                if g >= entry[0]:
                    na_set(c, g, v, True)
                else:
                    na_set(c, entry[0], None, False)
            elif g > entry[0]:
                # v becomes c's best non-adjacent partner
                na_set(c, g, v, True)

    # initial upper bounds: gain_na(c, d) <= a_neg * k_neg[c] * max_d k_neg[d]
    kn_max = max(k_neg) if N else 0.0
    for c in range(N):
        na_set(c, a_neg * k_neg[c] * kn_max, None, False)

    # Initialize single-node communities
    yield communities.values()

    # Merge the two communities that lead to the largest signed modularity
    while len(communities) > 1:
        # best adjacent pair (from the heap of row maxima)
        best_dq = None
        u = v = -1
        if len(H) > 0:
            row = H.heap[0].element
            top = dq_heap[row].heap[0]
            best_dq = -top.priority
            u, v = row, top.element[1]
        # best non-adjacent pair: stale upper bounds are resolved exactly
        # only while they can still beat the adjacent best
        while len(na_H) > 0:
            top = na_H.heap[0]
            c = top.element
            entry = na_best[c]
            if entry[2]:
                # exact root: every other entry is bounded by it
                g = entry[0]
                if best_dq is None or g > best_dq:
                    best_dq = g
                    u, v = c, entry[1]
                break
            if best_dq is not None and -top.priority <= best_dq:
                break  # no non-adjacent pair can beat the adjacent best
            na_rescan(c)
        if best_dq is None:
            break
        if not math.isfinite(best_dq):
            raise ValueError(
                "signed modularity gain overflowed the floating-point range"
            )
        yield best_dq
        merge(u, v)
        yield communities.values()


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(edge_attrs="weight")
def naive_greedy_modularity_communities(G, resolution=1, weight=None):
    r"""Find communities in G using greedy modularity maximization.

    This implementation is O(n^4), much slower than alternatives, but it is
    provided as an easy-to-understand reference implementation.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists.

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph
        Graph must be simple and undirected.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    list
        A list of sets of nodes, one for each community.
        Sorted by length with largest communities first.

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> c = nx.community.naive_greedy_modularity_communities(G)
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    See Also
    --------
    greedy_modularity_communities
    modularity
    """
    # First create one community for each node
    communities = [frozenset([u]) for u in G.nodes()]
    # Track merges
    merges = []
    # Greedily merge communities until no improvement is possible
    old_modularity = None
    new_modularity = modularity(G, communities, resolution=resolution, weight=weight)
    while old_modularity is None or new_modularity > old_modularity:
        # Save modularity for comparison
        old_modularity = new_modularity
        # Find best pair to merge
        trial_communities = list(communities)
        to_merge = None
        for i, u in enumerate(communities):
            for j, v in enumerate(communities):
                # Skip i==j and empty communities
                if j <= i or len(u) == 0 or len(v) == 0:
                    continue
                # Merge communities u and v
                trial_communities[j] = u | v
                trial_communities[i] = frozenset([])
                trial_modularity = modularity(
                    G, trial_communities, resolution=resolution, weight=weight
                )
                if trial_modularity >= new_modularity:
                    # Check if strictly better or tie
                    if trial_modularity > new_modularity:
                        # Found new best, save modularity and group indexes
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    elif to_merge and min(i, j) < min(to_merge[0], to_merge[1]):
                        # Break ties by choosing pair with lowest min id
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                # Un-merge
                trial_communities[i] = u
                trial_communities[j] = v
        if to_merge is not None:
            # If the best merge improves modularity, use it
            merges.append(to_merge)
            i, j, dq = to_merge
            u, v = communities[i], communities[j]
            communities[j] = u | v
            communities[i] = frozenset([])
    # Remove empty communities and sort
    return sorted((c for c in communities if len(c) > 0), key=len, reverse=True)
