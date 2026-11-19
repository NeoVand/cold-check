"""Fig. 1, "The check is the algorithm": breadth-first search from one finished answer to a cold run on 1,024 nodes.
183 x H mm. Cobalt: our fitted check run cold with the level condition kept; rose: the same check as fitted (without
the kept condition); navy: graphs and answers; grey: the graph's other edges.

a  One of the 30 training graphs (training graph 0, 12 nodes) and its answer, each node's parent (navy arrows, parent
   to child; the source filled). Rows: distance from the source; within a row, the order of discovery (drawing choices).
b  The completion, computed from the answer alone: each node's place in the order of discovery (the number in the node,
   which is also its position on the line). Arcs above: parent to child, always left to right. Arcs below: the graph's
   other edges.
c  The fitted check for "u is v's parent", drawn as the cold layer's pair and witness: the pair (u, v) with an edge
   u - v; a witness k joined to u by "k placed before u" and to v by an edge, and the witness test is "no such k" (the
   largest weights of the fit, -3.08 on both witness features). As fitted (rose): those conditions only. Kept (cobalt):
   the same and "u placed before v", the level condition, kept because it held at every training parent (the fit's
   gate, asserted equal to [edge, before]).
d, e  One 1,024-node test graph (test graph 0 of the 1,024-node set), run from "not yet". Radial layout: the source at
   the centre, ring r = distance from the source, each subtree a wedge (drawing choices). Pale grey: the graph's other
   edges. d, kept: the parent each node is given, all equal to the answer, and the ring is the loop in which the node
   was settled (asserted: settle loop = distance for every node). e, as fitted: parents that equal the answer in
   cobalt, wrong parents as rose chords from the parent given to the node, open circles for nodes left unsettled.
Numbers printed are asserted against data/bfs_map.json (itself asserted against data/bfs_run.json and the record).
    ./run.sh fig1.py
"""
import json
import sys
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, tone, text, head
from draw import seg, arrow, curve_arrow, node, ring, arc
from bfslayout import D, radial

g = D["graph1024"]; t0 = D["train0"]
LAB = np.array(g["label"]); S = g["source"]; N = g["n"]
KEPT = g["kept"]; AF = g["as_fitted"]
assert D["gate"] == [0, 1]
assert np.array_equal(np.array(KEPT["settle_loop"]), np.array(g["depth"]))        # settled ring by ring
assert KEPT["parent"] == g["label"]
af_par = np.array(AF["parent"])
n_right_af = int(((af_par == LAB) & (np.arange(N) != S)).sum())
n_wrong_af = int(((af_par >= 0) & (af_par != LAB)).sum())
n_open_af = int((af_par < 0).sum())
assert n_right_af == AF["right"][-1] == 312 and n_right_af + n_wrong_af + n_open_af == N - 1
assert AF["settled"][0] == 776 and AF["settled"][0] - AF["right"][0] == 554
LOOPS_KEPT = max(KEPT["settle_loop"]); assert LOOPS_KEPT == 9
LOOPS_AF = len(AF["settled"]); assert LOOPS_AF == 52                                # the loop at which nothing changed

H = 147.0
fig = kit.figure(H)
Y1 = 5.0                                    # first row of panel heads
Y2 = 51.0                                   # second row


def canvas(x, y, w, h):
    return kit.canvas(fig, x, y, w, h)


# ------------------------------------------------------------------------------------------- a: graph and its answer
def panel_a(x0, w):
    head(fig, x0, Y1, "a", "A training graph and its answer")
    cv = canvas(x0, Y1 + 3, w, 40)
    depth = np.array(t0["depth"]); place = np.array(t0["place"]); par = np.array(t0["label"]); s = t0["source"]
    P = {}
    dy = 7.6
    for d in range(depth.max() + 1):
        row = sorted(np.nonzero(depth == d)[0], key=lambda v: place[v])
        gap = 8.2
        for i, v in enumerate(row):
            P[int(v)] = np.array([x0 + w / 2 + (i - (len(row) - 1) / 2) * gap, Y1 + 9.0 + d * dy])
    tree = {(min(int(par[v]), v), max(int(par[v]), v)) for v in range(len(par)) if v != s}
    for a, b in t0["edges"]:
        if (min(a, b), max(a, b)) not in tree:
            seg(cv, P[a], P[b], tone(GREY, .55), HAIR, z=2)
    for v in range(len(par)):
        if v != s:
            arrow(cv, P[int(par[v])], P[v], INK, 0.7, r0=1.6, r1=1.6)
    for v in range(len(par)):
        node(cv, P[v], 1.6, fc=INK if v == s else "white")
    return P


# ------------------------------------------------------------------------------------------- b: the completion
def panel_b(x0, w):
    head(fig, x0, Y1, "b", "Completed from itself")
    text(fig, x0 + 4.6, Y1 + 4.2, "place in the order of discovery", color=SOFT)
    cv = canvas(x0, Y1 + 3, w, 40)
    place = np.array(t0["place"]); par = np.array(t0["label"]); s = t0["source"]; n = len(place)
    gap = (w - 4) / (n - 1)
    yl = Y1 + 24.0
    P = {v: np.array([x0 + 2 + place[v] * gap, yl]) for v in range(n)}
    tree = {(min(int(par[v]), v), max(int(par[v]), v)) for v in range(n) if v != s}
    for a, b in t0["edges"]:
        if (min(a, b), max(a, b)) not in tree:
            h = 0.9 * abs(P[a][0] - P[b][0]) ** 0.75
            cv.plot(*arc(P[a] + [0, 1.9], P[b] + [0, 1.9], h).T, color=tone(GREY, .55), lw=HAIR, zorder=2, gid="barc")
    kids = {u: sorted([v for v in range(n) if v != s and par[v] == u], key=lambda v: place[v]) for u in range(n)}
    for u, ks in kids.items():
        for i, v in enumerate(ks):
            a = np.deg2rad(62 - 52 * i / (len(ks) - 1)) if len(ks) > 1 else np.deg2rad(30)   # nearest child leaves lowest
            p0 = P[u] + 1.9 * np.array([np.sin(a), -np.cos(a)])
            p1 = P[v] + 1.9 * np.array([-np.sin(np.deg2rad(18)), -np.cos(np.deg2rad(18))])
            h = -1.0 * abs(P[u][0] - P[v][0]) ** 0.78
            curve_arrow(cv, arc(p0, p1, h), INK, 0.6, gid="barc")
    for v in range(n):
        node(cv, P[v], 1.9, fc=INK if v == s else "white", gid=f"b{v}")
        kit.names(fig, cv.text(*P[v], str(place[v]), ha="center", va="center_baseline", fontsize=LABEL,
                               color="white" if v == s else INK, zorder=8), f"b{v}", "barc")
    return P


# ------------------------------------------------------------------------------------------- c: the fitted check
def panel_c(x0, w):
    """The check for 'u is v's parent' in the pair-and-witness grammar: one glyph; the level condition in cobalt."""
    head(fig, x0, Y1, "c", "The fitted check, in 4 ms")
    text(fig, x0 + 4.6, Y1 + 4.2, "$u$ is the parent of $v$", color=SOFT)
    cv = canvas(x0, Y1 + 3, w, 42)
    cx, cy = x0 + w / 2 + 1.0, Y1 + 20.0
    u, v, k = np.array([cx - 11.0, cy]), np.array([cx + 11.0, cy]), np.array([cx, cy + 12.5])
    arrow(cv, u, v, INK, 0.7, r0=2.1, r1=2.1, gid="uv")                    # the pointer judged: u -> v, an edge
    for p, q, lab, ha, dx in ((k, u, "$k$ before $u$", "right", -1.6), (k, v, "edge to $v$", "left", 1.6)):
        e = (q - p) / np.hypot(*(q - p))
        seg(cv, p + e * 2.1, q - e * 2.1, INK, 0.5, ls=(0, (2.0, 1.2)), cap="butt", gid="leg")
        kit.names(fig, cv.text(*(0.5 * (p + q) + [dx, 1.0]), lab, ha=ha, va="center", fontsize=LABEL, color=INK), "leg")
    for p, s_ in ((u, "u"), (v, "v")):
        node(cv, p, 2.1, gid=s_)
        kit.names(fig, cv.text(*p, f"${s_}$", ha="center", va="center", fontsize=LABEL + 1.5, zorder=9), s_, "uv", "leg")
    ring(cv, k, 2.1, col=INK, lw=0.5, ls=(0, (1.6, 0.9)), gid="k")       # dashed: the witness that must not exist
    kit.names(fig, cv.text(*k, "$k$", ha="center", va="center", fontsize=LABEL + 1.5, zorder=9), "k", "leg")
    cv.text(k[0], k[1] + 5.6, "no such $k$", ha="center", va="baseline", fontsize=LABEL, color=INK)
    cv.text(cx, cy - 1.6, "edge", ha="center", va="baseline", fontsize=LABEL, color=INK)
    # the level condition: a cobalt bracket over the pair
    yb = cy - 6.2
    for xx in (u[0], v[0]):
        seg(cv, (xx, yb), (xx, yb + 1.1), COB, 0.6, cap="butt")
    seg(cv, (u[0], yb), (v[0], yb), COB, 0.6, cap="butt")
    cv.text(cx, yb - 1.5, "$u$ before $v$: the level condition", ha="center", va="baseline", fontsize=LABEL, color=COB)


# ------------------------------------------------------------------------------------------- d, e: the cold run
R, DEPTH, ANG = radial(g)
EDGES = np.array(g["edges"])
TREE = {(min(int(LAB[v]), v), max(int(LAB[v]), v)) for v in range(N) if v != S}
OTHER = [e for e in map(tuple, EDGES) if (min(e), max(e)) not in TREE]


def mapax(x, y, d):
    ax = kit.axes_mm(fig, x, y, d, d)
    ax.set_axis_off(); ax.set_aspect("equal")
    m = DEPTH.max() + 0.25
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    for r in range(1, DEPTH.max() + 1):
        ax.add_patch(Circle((0, 0), r, fc="none", ec=GUIDE, lw=0.35, zorder=1))
    ax.add_collection(LineCollection([R[list(e)] for e in OTHER], colors=tone(GREY, .22), linewidths=0.12, zorder=1.5))
    return ax


def panel_d(x0, d):
    head(fig, x0, Y2, "d", "Run cold from nothing, 1,024 nodes")
    ax = mapax(x0 + 2, Y2 + 5.5, d)
    ramp = {r: tone(COB, 1.0 - 0.42 * (r - 1) / (DEPTH.max() - 1)) for r in range(1, DEPTH.max() + 1)}
    vs = [v for v in range(N) if v != S]
    ax.add_collection(LineCollection([R[[LAB[v], v]] for v in vs], colors=[ramp[DEPTH[v]] for v in vs],
                                     linewidths=0.35, zorder=3))
    ax.scatter(*R[vs].T, s=0.3, c=[ramp[DEPTH[v]] for v in vs], lw=0, zorder=4)
    ax.scatter(*R[S], s=7, c=INK, lw=0, zorder=5)
    text(fig, x0 + 4.6, Y2 + 4.2, f"level condition kept: {N - 1:,} of {N - 1:,} parents right, the last at loop {LOOPS_KEPT}", color=COB)
    return ax


def panel_e(x0, d):
    head(fig, x0, Y2, "e", "The same check, as fitted")
    ax = mapax(x0 + 2, Y2 + 5.5, d)
    ok = [v for v in range(N) if v != S and af_par[v] == LAB[v]]
    bad = [v for v in range(N) if af_par[v] >= 0 and af_par[v] != LAB[v]]
    opn = [v for v in range(N) if af_par[v] < 0]
    ax.add_collection(LineCollection([R[[af_par[v], v]] for v in bad], colors=tone(ROSE, .45), linewidths=0.14, zorder=2.5))
    ax.add_collection(LineCollection([R[[LAB[v], v]] for v in ok], colors=COB, linewidths=0.35, zorder=3))
    ax.scatter(*R[bad].T, s=0.35, c=ROSE, lw=0, zorder=4)
    ax.scatter(*R[ok].T, s=0.35, c=COB, lw=0, zorder=4)
    ax.scatter(*R[opn].T, s=1.2, facecolors="white", edgecolors=ROSE, lw=0.25, zorder=4)
    ax.scatter(*R[S], s=7, c=INK, lw=0, zorder=5)
    text(fig, x0 + 4.6, Y2 + 4.2, f"without the level condition: {n_right_af} of {N - 1:,} right when it stops, at loop {LOOPS_AF}", color=ROSE)
    return ax


def ruler(x0, xc, d):
    """A radial scale under a map: ring r is the loop in which the cold run settles that ring's nodes."""
    unit = d / (2 * (DEPTH.max() + 0.25))
    y = Y2 + 5.5 + d + 2.2
    cv = canvas(x0, y - 1.5, 90, 6)
    seg(cv, (xc, y), (xc + DEPTH.max() * unit, y), INK, HAIR)
    for r in range(0, DEPTH.max() + 1):
        seg(cv, (xc + r * unit, y), (xc + r * unit, y + 0.9), INK, HAIR)
        if r in (1, 3, 5, 7, 9):
            text(fig, xc + r * unit, y + 3.6, str(r), ha="center")
    text(fig, xc + DEPTH.max() * unit + 2.2, y + 3.6, "loop = distance from the source", ha="left", color=SOFT)


def build():
    panel_a(0.0, 48.0)
    panel_b(52.0, 62.0)
    panel_c(118.0, 65.0)
    d = 82.0
    panel_d(0.0, d)
    panel_e(96.0, d)
    ruler(0, 2 + d / 2, d)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig1")
