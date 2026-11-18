import sys, traceback
sys.path.insert(0, "/workspace")
import networkx as nx
from networkx.algorithms.community import modularity, louvain_communities, greedy_modularity_communities

def base_graph(scale=1.0):
    G = nx.Graph()
    G.add_edge(0, 1, weight=4 * scale)
    G.add_edge(1, 2, weight=-1 * scale)
    G.add_edge(2, 3, weight=4 * scale)
    G.add_edge(0, 3, weight=-1 * scale)
    return G

G = base_graph(1e155)
for name, fn in [
    ("modularity", lambda: modularity(G, [{0,1},{2,3}], weight="weight", signed=True)),
    ("louvain", lambda: louvain_communities(G, weight="weight", seed=19, signed=True)),
]:
    try:
        print(name, fn())
    except Exception as e:
        print(name, "->", type(e).__name__, e)

# greedy second scenario
def greedy_graph(scale=1.0):
    G = nx.Graph()
    edges = [(0,0,-1),(0,1,-2),(0,3,1),(0,5,-2),(1,2,-1),(1,3,-2),(1,4,-5),
             (1,5,-2),(3,4,-5),(4,4,1),(4,5,-5)]
    for u,v,w in edges:
        G.add_edge(u, v, weight=w*scale)
    return G

for scale in (1.0, 1e155):
    G = greedy_graph(scale)
    c = greedy_modularity_communities(G, weight="weight", signed=True)
    print(f"greedy scale={scale:g}:", [sorted(x) for x in c])
    G0 = greedy_graph(1.0)
    print("  score at unit scale:", modularity(G0, c, weight="weight", signed=True))
