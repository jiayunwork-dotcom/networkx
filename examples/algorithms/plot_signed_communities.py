"""
================================================
Community detection in a signed relation network
================================================

This example shows how to detect groups in a network whose edges carry
*both* supportive and hostile relations, using the signed two-layer
modularity (``signed=True``) of :func:`networkx.community.modularity`
and the Louvain and greedy-modularity optimizers.

We build a small workplace network: each edge is an observed relation
between two colleagues, weighted by its strength.  Positive weights are
cooperative/supportive relations, negative weights are
conflicting/avoiding ones.  Summing the weights (or dropping the
negative ones) would throw away half of the evidence: two groups that
openly reject each other should *not* be merged, and colleagues who
share a common opponent are drawn together even without a direct edge.
"""

import matplotlib.pyplot as plt

import networkx as nx

# A workplace relation network: (u, v, weight); positive = support,
# negative = rejection.
relations = [
    # research group
    ("Ada", "Bea", 3.0),
    ("Ada", "Cid", 2.0),
    ("Bea", "Cid", 2.5),
    ("Ada", "Dee", 1.5),
    ("Dee", "Bea", 1.0),
    # sales group
    ("Eli", "Fay", 3.0),
    ("Eli", "Gus", 2.0),
    ("Fay", "Gus", 2.5),
    ("Eli", "Hal", 1.5),
    # management
    ("Ida", "Jon", 2.5),
    ("Ida", "Kim", 2.0),
    ("Jon", "Kim", 1.5),
    # rivalries between the groups
    ("Ada", "Eli", -2.0),
    ("Cid", "Fay", -1.5),
    ("Dee", "Gus", -1.0),
    ("Bea", "Hal", -1.5),
    ("Ada", "Ida", -0.5),
    ("Eli", "Jon", -0.5),
    # Hal and Kim have no direct edge to most of their group mates: they
    # are placed correctly because supporting one member and rejecting
    # the same opponents is itself evidence for grouping.
    ("Hal", "Ada", -2.0),
    ("Kim", "Ada", -2.0),
    ("Hal", "Bea", -1.0),
    ("Kim", "Bea", -1.0),
]

G = nx.Graph()
G.add_weighted_edges_from(relations)

# Communities found with the signed two-layer modularity.
signed_louvain = nx.community.louvain_communities(G, seed=12, signed=True)
signed_greedy = nx.community.greedy_modularity_communities(
    G, weight="weight", signed=True
)
signed_score = nx.community.modularity(G, signed_louvain, signed=True)

# What the usual (unsigned) modularity sees when every relation is
# treated as supportive evidence only.
unsigned_louvain = nx.community.louvain_communities(G, seed=12, weight=None)
unsigned_score = nx.community.modularity(G, unsigned_louvain, signed=True)

print(f"Signed Louvain communities (score {signed_score:.3f}):")
for com in signed_louvain:
    print("   ", sorted(com))
print("Signed greedy communities:")
for com in signed_greedy:
    print("   ", sorted(com))
print(
    f"Treating hostile relations as edges puts {len(G)} nodes in groups "
    f"that score only {unsigned_score:.3f} signed modularity:"
)
for com in unsigned_louvain:
    print("   ", sorted(com))

# --- drawing -------------------------------------------------------------
pos = nx.spring_layout(G, seed=7)
edge_colors = [
    "tab:green" if d["weight"] > 0 else "tab:red" for _, _, d in G.edges(data=True)
]
widths = [2.5 * abs(d["weight"]) for _, _, d in G.edges(data=True)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, communities, title in (
    (axes[0], unsigned_louvain, "unsigned (all relations positive)"),
    (axes[1], signed_louvain, "signed two-layer modularity"),
):
    palette = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"]
    node_colors = []
    for node in G:
        for i, com in enumerate(communities):
            if node in com:
                node_colors.append(palette[i % len(palette)])
                break
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths, ax=ax)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
