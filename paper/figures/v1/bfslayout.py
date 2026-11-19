"""Layouts of the 1,024-node test graph (drawing choices, not data): a radial tree (source at the centre, ring r = the
node's distance from the source, each subtree a wedge whose angle is proportional to its number of leaves, children in
the order of discovery) and a spring layout (networkx, seed 0)."""
import json
import numpy as np
from kit import DATA

D = json.load(open(DATA / "bfs_map.json"))


def radial(g, gap=0.0):
    n, s = g["n"], g["source"]
    par = np.array(g["label"]); place = np.array(g["place"]); depth = np.array(g["depth"])
    kids = {v: [] for v in range(n)}
    for v in np.argsort(place):
        if v != s:
            kids[int(par[v])].append(int(v))
    leaves = np.zeros(n)
    for v in np.argsort(-depth):
        leaves[v] = 1 if not kids[v] else sum(leaves[c] for c in kids[v])
    ang = np.zeros(n)
    span = {s: (0.0, 2 * np.pi)}
    for v in np.argsort(place):
        a0, a1 = span[int(v)]
        ang[v] = (a0 + a1) / 2
        t = a0
        for c in kids[int(v)]:
            w = (a1 - a0) * leaves[c] / leaves[v]
            span[c] = (t, t + w)
            t += w
    r = depth.astype(float)
    return np.c_[r * np.cos(ang), r * np.sin(ang)], depth, ang


def spring(g, seed=0):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(g["n"])); G.add_edges_from(map(tuple, g["edges"]))
    p = nx.spring_layout(G, seed=seed, iterations=200)
    return np.array([p[i] for i in range(g["n"])])
