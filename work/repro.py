import sys
import sys; sys.path.insert(0, "/workspace")
import traceback
import networkx as nx
from networkx.algorithms.community import modularity, louvain_communities, greedy_modularity_communities

def base_graph(scale=1.0):
    G = nx.Graph()
    G.add_edge(0, 1, weight=4 * scale)
    G.add_edge(1, 2, weight=-1 * scale)
    G.add_edge(2, 3, weight=4 * scale)
    G.add_edge(0, 3, weight=-1 * scale)
    return G

part = [{0, 1}, {2, 3}]

for scale in (1.0, 1e-165, 1e-310, 1e155):
    G = base_graph(scale)
    print(f"=== scale {scale:g} ===")
    try:
        q = modularity(G, part, weight="weight", signed=True)
        print("modularity:", q)
    except Exception:
        print("modularity ERROR:")
        traceback.print_exc(file=sys.stdout)
    try:
        c = louvain_communities(G, weight="weight", seed=19, signed=True)
        print("louvain:", [sorted(x) for x in c])
    except Exception:
        print("louvain ERROR:")
        traceback.print_exc(file=sys.stdout)
    try:
        c = greedy_modularity_communities(G, weight="weight", signed=True)
        print("greedy:", [sorted(x) for x in c])
    except Exception:
        print("greedy ERROR:")
        traceback.print_exc(file=sys.stdout)
