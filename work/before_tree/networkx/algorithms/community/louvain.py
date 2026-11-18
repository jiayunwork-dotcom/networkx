"""Function for detecting communities based on Louvain Community Detection
Algorithm"""

import itertools
from collections import defaultdict, deque

import networkx as nx
from networkx.algorithms.community import _signed, modularity
from networkx.algorithms.community._signed import (
    _check_signed_graph,
    _signed_layer_graph,
    _signed_modularity_reduced,
    _validate_signed_resolution,
)
from networkx.utils import py_random_state

__all__ = ["louvain_communities", "louvain_partitions"]


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def louvain_communities(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
    *,
    signed=False,
):
    r"""Find the best partition of a graph using the Louvain Community Detection
    Algorithm.

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The algorithm works in 2 steps. On the first step it assigns every node to be
    in its own community and then for each node it tries to find the maximum positive
    modularity gain by moving each node to all of its neighbor communities. If no positive
    gain is achieved the node remains in its original community.

    The modularity gain obtained by moving an isolated node $i$ into a community $C$ can
    easily be calculated by the following formula (combining [1]_ [2]_ and some algebra):

    .. math::
        \Delta Q = \frac{k_{i,in}}{2m} - \gamma\frac{ \Sigma_{tot} \cdot k_i}{2m^2}

    where $m$ is the size of the graph, $k_{i,in}$ is the sum of the weights of the links
    from $i$ to nodes in $C$, $k_i$ is the sum of the weights of the links incident to node $i$,
    $\Sigma_{tot}$ is the sum of the weights of the links incident to nodes in $C$ and $\gamma$
    is the resolution parameter.

    For the directed case the modularity gain can be computed using this formula according to [3]_

    .. math::
        \Delta Q = \frac{k_{i,in}}{m}
        - \gamma\frac{k_i^{out} \cdot\Sigma_{tot}^{in} + k_i^{in} \cdot \Sigma_{tot}^{out}}{m^2}

    where $k_i^{out}$, $k_i^{in}$ are the outer and inner weighted degrees of node $i$ and
    $\Sigma_{tot}^{in}$, $\Sigma_{tot}^{out}$ are the sum of in-going and out-going links incident
    to nodes in $C$.

    The first phase continues until no individual move can improve the modularity.

    The second phase consists in building a new network whose nodes are now the communities
    found in the first phase. To do so, the weights of the links between the new nodes are given by
    the sum of the weight of the links between nodes in the corresponding two communities. Once this
    phase is complete it is possible to reapply the first phase creating bigger communities with
    increased modularity.

    The above two phases are executed until no modularity gain is achieved (or is less than
    the `threshold`, or until `max_levels` is reached).

    Be careful with self-loops in the input graph. These are treated as
    previously reduced communities -- as if the process had been started
    in the middle of the algorithm. Large self-loop edge weights thus
    represent strong communities and in practice may be hard to add
    other nodes to.  If your input graph edge weights for self-loops
    do not represent already reduced communities you may want to remove
    the self-loops before inputting that graph.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
        The name of an edge attribute that holds the numerical value
        used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities
    threshold : float, optional (default=0.0000001)
        Modularity gain threshold for each level. If the gain of modularity
        between 2 levels of the algorithm is less than the given threshold
        then the algorithm stops and returns the resulting communities.
    max_level : int or None, optional (default=None)
        The maximum number of levels (steps of the algorithm) to compute.
        Must be a positive integer or None. If None, then there is no max
        level and the threshold parameter determines the stopping condition.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    signed : bool, optional (default=False)
        If True, use the signed two-layer modularity (see
        :func:`~networkx.algorithms.community.quality.modularity`) where
        negative edge weights repel communities. Only implemented for
        undirected graphs. Node moves then consider every community (not
        only neighboring ones) as well as the possibility of the node
        forming a singleton community, and a move is only performed when
        its gain exceeds 1e-12.

    Returns
    -------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.petersen_graph()
    >>> nx.community.louvain_communities(G, seed=123)
    [{0, 4, 5, 7, 9}, {1, 2, 3, 6, 8}]

    Notes
    -----
    The order in which the nodes are considered can affect the final output. In the algorithm
    the ordering happens using a random shuffle.

    References
    ----------
    .. [1] Blondel, V.D. et al. Fast unfolding of communities in
       large networks. J. Stat. Mech 10008, 1-12(2008). https://doi.org/10.1088/1742-5468/2008/10/P10008
    .. [2] Traag, V.A., Waltman, L. & van Eck, N.J. From Louvain to Leiden: guaranteeing
       well-connected communities. Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z
    .. [3] Nicolas Dugué, Anthony Perez. Directed Louvain : maximizing modularity in directed networks.
        [Research Report] Université d’Orléans. 2015. hal-01231784. https://hal.archives-ouvertes.fr/hal-01231784

    See Also
    --------
    louvain_partitions
    """

    partitions = louvain_partitions(
        G, weight, resolution, threshold, seed, signed=signed
    )
    if max_level is not None:
        if max_level <= 0:
            raise ValueError("max_level argument must be a positive integer or None")
        partitions = itertools.islice(partitions, max_level)
    final_partition = deque(partitions, maxlen=1)
    return final_partition.pop()


@py_random_state("seed")
@nx._dispatchable(edge_attrs="weight")
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None, *, signed=False
):
    """Yields partitions for each level of the Louvain Community Detection Algorithm

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The partitions at each level (step of the algorithm) form a dendrogram of communities.
    A dendrogram is a diagram representing a tree and each level represents
    a partition of the G graph. The top level contains the smallest communities
    and as you traverse to the bottom of the tree the communities get bigger
    and the overall modularity increases making the partition better.

    Each level is generated by executing the two phases of the Louvain Community
    Detection Algorithm.

    Be careful with self-loops in the input graph. These are treated as
    previously reduced communities -- as if the process had been started
    in the middle of the algorithm. Large self-loop edge weights thus
    represent strong communities and in practice may be hard to add
    other nodes to.  If your input graph edge weights for self-loops
    do not represent already reduced communities you may want to remove
    the self-loops before inputting that graph.

    Parameters
    ----------
    G : NetworkX graph
    weight : string or None, optional (default="weight")
     The name of an edge attribute that holds the numerical value
     used as a weight. If None then each edge has weight 1.
    resolution : float, optional (default=1)
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities
    threshold : float, optional (default=0.0000001)
     Modularity gain threshold for each level. If the gain of modularity
     between 2 levels of the algorithm is less than the given threshold
     then the algorithm stops and returns the resulting communities.
    seed : integer, random_state, or None (default)
     Indicator of random number generation state.
     See :ref:`Randomness<randomness>`.
    signed : bool, optional (default=False)
     If True, use the signed two-layer modularity (see
     :func:`~networkx.algorithms.community.quality.modularity`) where
     negative edge weights repel communities. Only implemented for
     undirected graphs.

    Yields
    ------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.

    References
    ----------
    .. [1] Blondel, V.D. et al. Fast unfolding of communities in
       large networks. J. Stat. Mech 10008, 1-12(2008)

    See Also
    --------
    louvain_communities
    """
    if signed:
        yield from _louvain_partitions_signed(G, weight, resolution, threshold, seed)
        return

    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed
        )


def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None):
    """Calculate one level of the Louvain partitions tree

    Parameters
    ----------
    G : NetworkX Graph/DiGraph
        The graph from which to detect communities
    m : number
        The size of the graph `G`.
    partition : list of sets of nodes
        A valid partition of the graph `G`
    resolution : positive number
        The resolution parameter for computing the modularity of a partition
    is_directed : bool
        True if `G` is a directed graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    """
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        in_degrees = dict(G.in_degree(weight="weight"))
        out_degrees = dict(G.out_degree(weight="weight"))
        Stot_in = list(in_degrees.values())
        Stot_out = list(out_degrees.values())
        # Calculate weights for both in and out neighbors without considering self-loops
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data="weight"):
                if u != n:
                    nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight="weight"))
        Stot = list(degrees.values())
        nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = (
                    -weights2com[best_com] / m
                    + resolution
                    * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com])
                    / m**2
                )
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (
                    Stot[best_com] * degree
                ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution
                        * (
                            out_degree * Stot_in[nbr_com]
                            + in_degree * Stot_out[nbr_com]
                        )
                        / m**2
                    )
                else:
                    gain = (
                        remove_cost
                        + wt / m
                        - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                    )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _neighbor_weights(nbrs, node2com):
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbors as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, weight=wt + temp)
    return H


def _convert_multigraph(G, weight, is_directed):
    """Convert a Multigraph to normal Graph"""
    if is_directed:
        H = nx.DiGraph()
    else:
        H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, wt in G.edges(data=weight, default=1):
        if H.has_edge(u, v):
            H[u][v]["weight"] += wt
        else:
            H.add_edge(u, v, weight=wt)
    return H


def _louvain_partitions_signed(G, weight, resolution, threshold, seed):
    """Yields partitions for each level of signed Louvain (two-layer score).

    See :func:`louvain_partitions` with ``signed=True``.  The graph is
    kept as a two-layer simple graph (see
    :func:`~networkx.algorithms.community._signed._signed_layer_graph`)
    whose positive and negative edge contributions are preserved through
    every compression step, so the score of a partition of a compressed
    graph always equals the signed modularity of the corresponding
    partition of the original graph.
    """
    _check_signed_graph(G)
    resolution = _validate_signed_resolution(resolution)
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    graph, m_pos, m_neg = _signed_layer_graph(G, weight)
    if m_pos + m_neg == 0:
        # No mass in either layer: every partition scores 0, no move can
        # improve it, so the singleton partition is already optimal.
        yield partition
        return
    mod = _signed_modularity_reduced(graph, partition, m_pos, m_neg, resolution)
    partition, inner_partition, improvement = _one_level_signed(
        graph, m_pos, m_neg, partition, resolution, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = _signed_modularity_reduced(
            graph, inner_partition, m_pos, m_neg, resolution
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph_signed(graph, inner_partition)
        partition, inner_partition, improvement = _one_level_signed(
            graph, m_pos, m_neg, partition, resolution, seed
        )


def _one_level_signed(G, m_pos, m_neg, partition, resolution, seed):
    """Calculate one level of the signed Louvain partitions tree.

    ``G`` is a two-layer graph with ``"weight_pos"``/``"weight_neg"`` edge
    attributes and total layer masses ``m_pos``/``m_neg``.  Each node is
    moved to the community (among *all* current communities, plus the
    possibility of forming a singleton community) that maximizes the gain
    of the signed score, iterating until no gain exceeds 1e-12.

    The gain of moving node ``i`` (layer degrees ``k_pos``, ``k_neg``)
    from its community ``A`` into community ``C`` is computed exactly as
    the difference of the signed score, in the form

        remove_cost + [w(i, C) - gamma * (S_pos_C * k_pos / (2 m_pos)
                                        - S_neg_C * k_neg / (2 m_neg))] / m_tot

    where ``w(i, C)`` is the net (positive minus negative) weight from
    ``i`` to ``C`` excluding self-loops, ``S_pos_C``/``S_neg_C`` are the
    community layer-degree sums, ``m_tot = m_pos + m_neg`` and

        remove_cost = [-w(i, A) + gamma * (S_pos_A' * k_pos / (2 m_pos)
                                           - S_neg_A' * k_neg / (2 m_neg))] / m_tot

    with ``S'`` the community sums after removing ``i``.  Only the net
    per-pair weight enters the move gain; the two layers are preserved
    separately in the degrees, the community sums, the masses and the
    compressed graphs.
    """
    stats = _signed._EVAL_STATS
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    m_tot = m_pos + m_neg
    deg_pos = dict(G.degree(weight="weight_pos"))
    deg_neg = dict(G.degree(weight="weight_neg"))
    Stot_pos = list(deg_pos.values())
    Stot_neg = list(deg_neg.values())
    # per-node neighbor layer weights, self-loops excluded (they move with
    # the node and cancel out of every gain)
    nbrs = {}
    for u in G:
        nbrs[u] = {
            v: (data["weight_pos"], data["weight_neg"])
            for v, data in G[u].items()
            if v != u
        }
    inv_pos = resolution / (2 * m_pos) if m_pos else 0.0
    inv_neg = resolution / (2 * m_neg) if m_neg else 0.0
    # Community slots holding negative-layer degree mass.  A community
    # without negative mass never beats the singleton option for any node
    # (its insert gain is <= 0), and no community beats it for a node
    # without negative degree either, so scanning exactly this set (when
    # it can matter) covers every community that could win a move.
    neg_comms = {c for c, s in enumerate(Stot_neg) if s > 0}
    empty_comms = []  # stack of empty community slots, reused for singletons
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            com_u = node2com[u]
            k_pos = deg_pos[u]
            k_neg = deg_neg[u]
            # layer weights from u to each neighboring community
            weights2com_pos = defaultdict(float)
            weights2com_neg = defaultdict(float)
            for v, (w_pos, w_neg) in nbrs[u].items():
                com = node2com[v]
                weights2com_pos[com] += w_pos
                weights2com_neg[com] += w_neg
            # remove u from its community
            sp_u = Stot_pos[com_u] - k_pos
            sn_u = Stot_neg[com_u] - k_neg
            Stot_pos[com_u] = sp_u
            Stot_neg[com_u] = sn_u
            if sn_u == 0.0:
                neg_comms.discard(com_u)
            remove_cost = (
                -(weights2com_pos[com_u] - weights2com_neg[com_u])
                + inv_pos * sp_u * k_pos
                - inv_neg * sn_u * k_neg
            ) / m_tot
            best_mod = 1e-12
            best_com = com_u
            # neighboring communities (exact gain, including the edge term)
            for com, w_pos in weights2com_pos.items():
                gain = (
                    remove_cost
                    + (
                        w_pos
                        - weights2com_neg[com]
                        - inv_pos * Stot_pos[com] * k_pos
                        + inv_neg * Stot_neg[com] * k_neg
                    )
                    / m_tot
                )
                if gain > best_mod:
                    best_mod = gain
                    best_com = com
            if stats is not None:
                stats["louvain_move_gain_evals"] += len(weights2com_pos)
            # non-neighboring communities with negative-layer mass: their
            # insert gain has no edge term and can still be positive
            if k_neg > 0 and inv_neg > 0:
                n_evals = 0
                for com in neg_comms:
                    if com in weights2com_pos:
                        continue
                    n_evals += 1
                    gain = (
                        remove_cost
                        + (
                            -inv_pos * Stot_pos[com] * k_pos
                            + inv_neg * Stot_neg[com] * k_neg
                        )
                        / m_tot
                    )
                    if gain > best_mod:
                        best_mod = gain
                        best_com = com
                if stats is not None:
                    stats["louvain_move_gain_evals"] += n_evals
            # the possibility of u forming a singleton community
            if remove_cost > best_mod:
                best_mod = remove_cost
                best_com = None
            if best_com is None:
                if empty_comms:
                    best_com = empty_comms.pop()
                else:
                    best_com = len(Stot_pos)
                    Stot_pos.append(0.0)
                    Stot_neg.append(0.0)
                    inner_partition.append(set())
                    partition.append(set())
            # insert u into best_com (or back into its own community)
            Stot_pos[best_com] += k_pos
            sn_new = Stot_neg[best_com] + k_neg
            Stot_neg[best_com] = sn_new
            if sn_new > 0:
                neg_comms.add(best_com)
            if best_com != com_u:
                com = G.nodes[u].get("nodes", {u})
                partition[com_u].difference_update(com)
                inner_partition[com_u].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                if not inner_partition[com_u]:
                    empty_comms.append(com_u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _gen_graph_signed(G, partition):
    """Generate a new two-layer graph based on the partitions of ``G``.

    Like :func:`_gen_graph`, but the positive and negative contributions
    are summed separately onto the ``"weight_pos"``/``"weight_neg"``
    attributes of the compressed graph; they are never netted against
    each other, so the total layer masses and all layer degrees are
    preserved exactly.
    """
    H = nx.Graph()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, data in G.edges(data=True):
        com1 = node2com[node1]
        com2 = node2com[node2]
        if H.has_edge(com1, com2):
            d = H[com1][com2]
            d["weight_pos"] += data["weight_pos"]
            d["weight_neg"] += data["weight_neg"]
        else:
            H.add_edge(
                com1,
                com2,
                weight_pos=data["weight_pos"],
                weight_neg=data["weight_neg"],
            )
    return H
