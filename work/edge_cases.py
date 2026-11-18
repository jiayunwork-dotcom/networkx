import sys
sys.path.insert(0, "/workspace")
import math
import random
import networkx as nx
from networkx.algorithms.community import modularity, louvain_communities, greedy_modularity_communities

def ref(G, communities, resolution=1.0):
    node2com = {u: i for i, c in enumerate(communities) for u in c}
    n = len(communities)
    exp = None
    for _, _, w in G.edges(data="weight", default=1):
        w = float(w)
        if w != 0.0:
            e = math.frexp(w)[1] - 1
            exp = e if exp is None or e > exp else exp
    exp = 0 if exp is None else exp
    pe, ne = [], []
    dp = {u: [] for u in G}; dn = {u: [] for u in G}
    Lp = [[] for _ in range(n)]; Ln = [[] for _ in range(n)]
    for u, v, w in G.edges(data="weight", default=1):
        w = math.ldexp(float(w), -exp); c = node2com[u]
        if w > 0:
            pe.append(w); dp[u].append(w); dp[v].append(w)
            if c == node2com[v]: Lp[c].append(w)
        elif w < 0:
            aw=-w; ne.append(aw); dn[u].append(aw); dn[v].append(aw)
            if c == node2com[v]: Ln[c].append(aw)
    mp, mn = math.fsum(pe), math.fsum(ne)
    Kp = [math.fsum(x for u in c for x in dp[u]) for c in communities]
    Kn = [math.fsum(x for u in c for x in dn[u]) for c in communities]
    lp = [math.fsum(x) for x in Lp]; ln = [math.fsum(x) for x in Ln]
    num = 0.0
    if mp:
        qp = math.fsum(lp)/mp - resolution*math.fsum([0.0]+[(Kp[c]/(2*mp))**2 for c in range(n)])
        num += mp*qp
    if mn:
        qn = math.fsum(ln)/mn - resolution*math.fsum([0.0]+[(Kn[c]/(2*mn))**2 for c in range(n)])
        num -= mn*qn
    return 0.0 if mp+mn == 0 else num/(mp+mn)

# 1) single subnormal edge (5e-324): normalized arithmetic must give -0.5
G = nx.Graph(); G.add_edge(0,1,weight=5e-324)
p = [{0},{1}]
got = modularity(G,p,weight="weight",signed=True)
print("subnormal edge score:", got, "ref:", ref(G,p))
assert abs(got - ref(G,p)) < 1e-10 and abs(got - (-0.5)) < 1e-10

# 2) max finite edge
G = nx.Graph()
G.add_edge(0,1,weight=1.7976931348623157e308)
G.add_edge(1,2,weight=-1.7976931348623157e308)
p=[{0,1},{2}]
q = modularity(G,p,weight="weight",signed=True)
print("max-finite edge score:", q, "ref:", ref(G,p))
assert math.isfinite(q) and abs(q-ref(G,p))<1e-10
print("louvain:", louvain_communities(G,weight="weight",seed=1,signed=True))
print("greedy:", greedy_modularity_communities(G,weight="weight",signed=True))

# 3) layer gap 1e290
G = nx.MultiGraph()
G.add_edge(0,1,weight=1e300)
G.add_edge(2,3,weight=-1e10)
p=[{0,1},{2,3}]
q = modularity(G,p,weight="weight",signed=True)
print("gap 1e290 score:", q, "ref:", ref(G,p))
assert math.isfinite(q) and abs(q-ref(G,p))<1e-10

# 4) unrepresentable raw mass -> explicit ValueError
G = nx.MultiGraph()
G.add_edge(0,1,weight=1.5e308); G.add_edge(0,1,weight=1.5e308)
for fn in (lambda: modularity(G,[{0},{1}],weight="weight",signed=True),
           lambda: louvain_communities(G,weight="weight",signed=True),
           lambda: greedy_modularity_communities(G,weight="weight",signed=True)):
    try:
        fn(); print("ERROR: no raise")
    except ValueError as e:
        print("overflow correctly raised:", e)

# 5) 1e300 multigraph with self loops
G = nx.MultiGraph()
edges=[(0,0,2.0),(0,1,3.0),(0,1,-1.5),(1,2,-2.0),(2,2,1.0),(2,3,4.0),(0,3,-3.0)]
s=1e300
for u,v,w in edges: G.add_edge(u,v,weight=w*s)
part=[{0,1},{2,3}]
q = modularity(G,part,weight="weight",signed=True)
print("1e300 multigraph score:", q, "ref:", ref(G,part))
assert math.isfinite(q) and abs(q-ref(G,part))<1e-10
lp = louvain_communities(G,weight="weight",seed=5,signed=True)
gp = greedy_modularity_communities(G,weight="weight",signed=True)
print("louvain part:", sorted(map(sorted,lp)), "score", ref(G,lp))
print("greedy part:", sorted(map(sorted,gp)), "score", ref(G,gp))
single=[{u} for u in G]
assert ref(G,lp) >= ref(G,single) - 1e-10
assert ref(G,gp) >= ref(G,single) - 1e-10

# 6) resolution semantics
G = nx.Graph(); G.add_edge(0,1,weight=2.0); G.add_edge(2,3,weight=-2.0); G.add_edge(1,2,weight=0.5)
part=[{0,3},{1,2}]
for gamma in (0.0, 0.5, 1.0, 10.0):
    assert abs(modularity(G,part,weight="weight",resolution=gamma,signed=True)-ref(G,part,gamma))<1e-12
print("resolution variants OK")

# 7) weight=None / missing attribute
G = nx.Graph(); G.add_edge(0,1); G.add_edge(1,2,weight=-2.0); G.add_edge(2,3)
part=[{0,1},{2,3}]
a = modularity(G,part,weight=None,signed=True)
b = modularity(G,part,weight=None)
print("weight=None signed vs unsigned:", a, b)
assert abs(a-b) < 1e-12

# 8) NaN/inf must not be swallowed
for bad in (float("nan"), float("inf"), -float("inf")):
    H = nx.Graph(); H.add_edge(0,1,weight=bad)
    for fn in (lambda: modularity(H,[{0},{1}],weight="weight",signed=True),
               lambda: louvain_communities(H,weight="weight",signed=True),
               lambda: greedy_modularity_communities(H,weight="weight",signed=True)):
        try:
            fn(); raise SystemExit("FAIL: bad weight accepted")
        except ValueError:
            pass
print("invalid weights rejected OK")
print("ALL EDGE CASES OK")
