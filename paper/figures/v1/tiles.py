"""The thirty CLRS-30 task glyphs (one small input of the benchmark's own generator, its answer drawn over it), in the
grammar of the earlier paper's CLRS figure, whose drawing code this is. Inputs: data/glyphs/<task>.npz, the library's
sampler at the test split's seed with ten items (articulation points, bridges and Kruskal at 24 nodes, the smallest size
whose first input has a cycle), labels only. Every glyph draws the input's true answer: a reader of pred_<key> gets the
label <key> (Truth below), so the answer marks are the label's.

Drawing choices (schematic, not data): node order around each ring (fewest crossings of the answer, then of the input,
then the shortest chords, over all 9! orders); the 24-node layout; directed input edges drawn without heads; the sort
tiles' arcs; the split tables' tints (where the chosen split falls in its range); string letters as glyph outlines.
"""
import itertools
import math
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Polygon, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

import kit
from kit import INK, SOFT, COB, ROSE, LABEL, tone

PT = 72 / 25.4                                         # points per mm
QPX = 25.4 / 300                                       # one pixel at 300 dpi, mm (two at 600 dpi)


def snap(v):
    """v (mm from the page's top-left corner) on the 300-dpi pixel grid (so also on the 600-dpi grid)."""
    return round(v / QPX) * QPX

FAINT = tone(INK, 0.24)                                # input hairlines
FAINT_LW = 0.35
NODE_LW = 0.4
ANS_LW = 0.75                                          # answer strokes
HEAD, HEADW = 1.2, 0.8                                 # arrowheads, mm
BAND = tone(INK, 0.055)                                # panel b family bands

ANS = INK                                              # the answer: navy ink (a drawn object, not a comparison)
FNC = ROSE                                             # FloydNet, the soft baseline


# ---------------------------------------------------------------------------------------------------- tile data
GLYPHS = pathlib.Path(__file__).resolve().parent / "data" / "glyphs"
SPARSE_T = ("articulation_points", "bridges", "mst_kruskal")


class Truth:
    """A glyph input whose pred_<key> reads the label <key>."""
    def __init__(self, z):
        self.z, self.files = z, z.files

    def __getitem__(self, k):
        return self.z[k[5:]] if k.startswith("pred_") and k[5:] in self.z.files else self.z[k]


Z = {p.stem.replace("_24", ""): Truth(np.load(p)) for p in sorted(GLYPHS.glob("*.npz"))
     if not (p.stem in SPARSE_T)}
assert len(Z) == 30, sorted(Z)
IDX = 0
WC, HC = 20.2, 13.6                                    # a tile's drawing, mm
RING_R = 6.15                                          # graph tiles: ring radius
NODE_R = 0.55
NODE_R24 = 0.38



# ---------------------------------------------------------------------------------------------------- primitives
def seg(ax, p, q, col, lw, z=3, gid=None, cap="round", **kw):
    return ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, zorder=z, solid_capstyle=cap, gid=gid, **kw)[0]


def arrow(ax, p, q, col=ANS, lw=ANS_LW, r0=0.0, r1=0.0, z=5, gid=None, head=HEAD, hw=HEADW):
    """Straight arrow p -> q trimmed by r0, r1, a solid head `head` mm long at q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = float(np.hypot(*d))
    u = d / L
    a, b = p + u * r0, q - u * r1
    nrm = np.array([-u[1], u[0]])
    base = b - u * head
    ax.add_patch(Polygon([b, base + nrm * hw / 2, base - nrm * hw / 2], closed=True, fc=col, ec=col, lw=0.2, zorder=z + 0.1,
                         joinstyle="miter", gid=gid))
    seg(ax, a, base + u * 0.15, col, lw, z, gid=gid, cap="butt")


def node(ax, p, r=NODE_R, fc="white", ec=INK, lw=NODE_LW, z=6, gid=None):
    ax.add_patch(Circle(p, r, fc=fc, ec=ec, lw=lw, zorder=z, gid=gid))


def ring(ax, p, r, col=INK, lw=0.5, z=6, gid=None, ls="-"):
    ax.add_patch(Circle(p, r, fc="none", ec=col, lw=lw, zorder=z, gid=gid, ls=ls))


def rect(ax, x, y, w, h, fc="white", ec=INK, lw=NODE_LW, z=3, gid=None, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="miter", gid=gid, **kw))


def cross_cell(ax, x, y, s, gid):
    """A wrong answer's cell: white, navy outline, a navy cross corner to corner (no colour: rose means warm)."""
    lw = 2 * QPX * PT                                     # 0.48 pt: 2 px at 300 dpi
    h = QPX                                               # the outline inside the cell (half its width)
    rect(ax, x + h, y + h, s - 2 * h, s - 2 * h, fc="white", ec=ANS, lw=lw, z=3.5, gid=gid)
    for a, b in (((x + h, y + h), (x + s - h, y + s - h)), ((x + h, y + s - h), (x + s - h, y + h))):
        seg(ax, a, b, ANS, lw, z=3.6, gid=gid, cap="butt")


def col(ok):
    """The answer's ink. Right and wrong answers are drawn alike; a wrong one is marked apart (a crossed cell or a dotted ring)."""
    return ANS


# ---------------------------------------------------------------------------------------------------- rings
def ring_points(n, cx, cy, R=RING_R):
    th = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return np.stack([cx + R * np.cos(th), cy - R * np.sin(th)], 1)


def _cross_count(slot, A, B):
    """Number of crossing pairs between chord sets A and B (lists of node pairs) for each row of slot [P, n]."""
    c = np.zeros(len(slot), np.int32)
    for (a, b) in A:
        la, ha = np.minimum(slot[:, a], slot[:, b]), np.maximum(slot[:, a], slot[:, b])
        for (e, f) in B:
            if len({a, b, e, f}) < 4:
                continue
            lb, hb = np.minimum(slot[:, e], slot[:, f]), np.maximum(slot[:, e], slot[:, f])
            c += ((la < lb) & (lb < ha) & (ha < hb)) | ((lb < la) & (la < hb) & (hb < ha))
    return c


def order10(inp, ans):
    """Node -> slot on a ring of ten: every order with node 0 at the top, scored lexicographically by crossings among
    the answer's edges, crossings of the answer with the input, crossings among the input (when there are at most 24
    input edges), then total chord length in slots; ties to the first found."""
    perms = np.array(list(itertools.permutations(range(1, 10))), np.int8)
    slot = np.zeros((len(perms), 10), np.int8)
    slot[np.arange(len(perms))[:, None], perms] = np.arange(1, 10, dtype=np.int8)[None]
    ans = sorted({tuple(sorted(e)) for e in ans})
    inp = sorted({tuple(sorted(e)) for e in inp} - set(ans))
    c1 = _cross_count(slot, ans, ans) // 2
    c2 = _cross_count(slot, ans, inp)
    c3 = _cross_count(slot, inp, inp) // 2 if len(inp) <= 24 else np.zeros(len(slot), np.int32)
    E = np.array(ans + inp)
    d = np.abs(slot[:, E[:, 0]].astype(int) - slot[:, E[:, 1]])
    length = np.minimum(d, 10 - d).sum(1)
    best = np.lexsort((np.arange(len(perms)), length, c3, c2, c1))[0]
    return slot[best].astype(int), (int(c1[best]), int(c2[best]), int(c3[best]), int(length[best]))


def sparse_layout(n, edges, edge_mm=2.1, gap=1.5):
    """Positions (tile-centred, mm) for the 24-node sparse graph: the nodes without an edge evenly on the ring, in index
    order from the top; each component of two or more nodes inside it, laid out by Kamada-Kawai (networkx), turned so
    its long axis is horizontal, scaled to a mean edge of `edge_mm`, and set side by side, centred. Drawing choice."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    iso = [v for v in range(n) if G.degree(v) == 0]
    xy = np.zeros((n, 2))
    xy[iso] = ring_points(len(iso), 0, 0)
    comps = sorted((sorted(c) for c in nx.connected_components(G) if len(c) > 1), key=min)
    parts = []
    for c in comps:
        pos = nx.kamada_kawai_layout(G.subgraph(c))
        P = np.array([pos[v] for v in c])
        P = P - P.mean(0)
        u = np.linalg.svd(P, full_matrices=False)[2][0]               # long axis -> horizontal
        P = P @ np.array([[u[0], -u[1]], [u[1], u[0]]])
        el = np.mean([np.hypot(*(P[c.index(a)] - P[c.index(b)])) for a, b in G.subgraph(c).edges])
        parts.append((c, P * edge_mm / el))
    total = sum(np.ptp(P[:, 0]) for _, P in parts) + gap * (len(parts) - 1)
    x = -total / 2
    for c, P in parts:
        P = P - [P[:, 0].min() - x, 0]
        xy[c] = P
        x += np.ptp(P[:, 0]) + gap
    return xy, dict(isolated=len(iso), components=[len(c) for c, _ in parts],
                    inner_extent=float(np.abs(xy[[v for c, _ in parts for v in c]]).max()))


def und_edges(adj):
    a = (adj > 0) | (adj.T > 0)
    np.fill_diagonal(a, False)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(a)))]


# ---------------------------------------------------------------------------------------------------- graph tiles
class Ring:
    """Node positions of one graph tile, centred in the tile: a ring order (int slots), or tile-centred positions."""

    def __init__(self, x0, y0, slot, r=NODE_R):
        if np.asarray(slot).ndim == 2:
            self.xy = np.asarray(slot, float) + [x0 + WC / 2, y0 + HC / 2]
        else:
            self.xy = ring_points(len(slot), x0 + WC / 2, y0 + HC / 2)[slot]
        self.r = r
        self.n = len(self.xy)


def web(ax, R, adj, gid):
    for i, j in und_edges(adj):
        seg(ax, R.xy[i], R.xy[j], FAINT, FAINT_LW, z=2, gid=gid, cap="butt")


def nodes(ax, R, fill=None, gid=None, lw=None):
    for k in range(R.n):
        f = (fill or {}).get(k)
        node(ax, R.xy[k], R.r, fc=f or "white", ec=f or INK, lw=lw or NODE_LW, gid=gid)


def tree(ax, R, pred, true, gid):
    """Parent pointers as arrows parent -> child."""
    for v in range(R.n):
        if pred[v] != v:
            arrow(ax, R.xy[pred[v]], R.xy[v], col=col(pred[v] == true[v]), r0=R.r, r1=R.r + 0.08, gid=gid)
        elif true[v] != v:                                 # "no parent" where there is one
            ring(ax, R.xy[v], R.r + 0.45, col=ANS, lw=0.7, gid=gid, ls=(0, (1.0, 0.8)))


def draw_tree(t, src=True):
    def draw(ax, x0, y0, rings):
        z = Z[t]
        R = rings[t]
        web(ax, R, z["adj"][IDX], f"in:{t}")
        tree(ax, R, z["pred_pi"][IDX], z["pi"][IDX], f"ans:{t}")
        nodes(ax, R, gid=f"node:{t}")
        if src and "s" in z.files:
            ring(ax, R.xy[int(z["s"][IDX])], R.r + 0.5, col=INK, lw=0.5, gid=f"node:{t}")
    return draw


def draw_fw(ax, x0, y0, rings):
    """All pairs: the union of the ten shortest-route trees; an edge's width counts the trees that use it."""
    t = "floyd_warshall"
    z = Z[t]
    R = rings[t]
    P, T = z["pred_Pi"][IDX], z["Pi"][IDX]
    web(ax, R, z["adj"][IDX], f"in:{t}")
    use, bad = {}, set()
    for s in range(10):
        for v in range(10):
            u = int(P[s, v])
            if u == v:
                if T[s, v] != v:
                    bad.add((v, v))
                continue
            e = (min(u, v), max(u, v))
            use[e] = use.get(e, 0) + 1
            if u != T[s, v]:
                bad.add(e)
    for (u, v), k in sorted(use.items()):
        seg(ax, R.xy[u], R.xy[v], col((u, v) not in bad), 0.3 + 0.075 * k, z=4, gid=f"ans:{t}", cap="butt")
    nodes(ax, R, gid=f"node:{t}")
    return use


def draw_edges(t, key):
    """Edge answers (bridges, spanning tree): the chosen edges bold over the faint input."""
    def draw(ax, x0, y0, rings):
        z = Z[t]
        R = rings[t]
        adj, P, T = z["adj"][IDX], z[f"pred_{key}"][IDX], z[key][IDX]
        web(ax, R, adj, f"in:{t}")
        for i, j in und_edges(adj):
            p, tr = int(P[i, j] == 1), int(T[i, j] == 1)
            if p or tr:
                seg(ax, R.xy[i], R.xy[j], col(p == tr), 1.25 if p else 0.6, z=4, gid=f"ans:{t}", cap="butt")
        nodes(ax, R, gid=f"node:{t}")
    return draw


def draw_cut(ax, x0, y0, rings):
    t = "articulation_points"
    z = Z[t]
    R = rings[t]
    P, T = z["pred_is_cut"][IDX], z["is_cut"][IDX]
    web(ax, R, z["adj"][IDX], f"in:{t}")
    nodes(ax, R, fill={k: col(P[k] == T[k]) for k in range(R.n) if P[k] == 1}, gid=f"node:{t}")
    for k in range(R.n):                                   # a cut point the answer missed
        if T[k] == 1 and P[k] != 1:
            ring(ax, R.xy[k], R.r + 0.4, col=ANS, lw=0.7, gid=f"ans:{t}", ls=(0, (1.0, 0.8)))


def hull_path(pts, pad):
    """A rounded hull: the convex hull of `pts` grown by `pad` mm, as a closed polygon."""
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cloud = np.concatenate([p + pad * np.c_[np.cos(th), np.sin(th)] for p in pts])
    # monotone chain
    P = sorted(map(tuple, cloud))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, hi = half(P), half(P[::-1])
    return np.array(lo[:-1] + hi[:-1])


def draw_scc(ax, x0, y0, rings):
    """Components of two or more nodes as pale hulls (the answer's grouping); the input faint."""
    t = "strongly_connected_components"
    z = Z[t]
    R = rings[t]
    P, T = z["pred_scc_id"][IDX], z["scc_id"][IDX]
    web(ax, R, z["adj"][IDX], f"in:{t}")
    groups = {}
    for v in range(R.n):
        groups.setdefault(int(P[v]), []).append(v)
    for g, vs in groups.items():
        if len(vs) > 1:
            ok = all(P[v] == T[v] for v in vs) and sum(T == T[vs[0]]) == len(vs)
            ax.add_patch(Polygon(hull_path(R.xy[vs], R.r + 0.75), closed=True, fc=tone(col(ok), 0.16), ec=tone(col(ok), 0.55),
                                 lw=0.4, zorder=1.5, gid=f"ans:{t}"))
    nodes(ax, R, gid=f"node:{t}")
    return groups


def draw_topo(ax, x0, y0, rings):
    """The order as one chain: each node points to the node after it (the library stores the one before); the first
    node is ringed."""
    t = "topological_sort"
    z = Z[t]
    R = rings[t]
    P, T, hp, ht = z["pred_topo"][IDX], z["topo"][IDX], int(z["pred_topo_head"][IDX]), int(z["topo_head"][IDX])
    web(ax, R, z["adj"][IDX], f"in:{t}")
    for v in range(R.n):
        if P[v] != v:
            arrow(ax, R.xy[P[v]], R.xy[v], col=col(P[v] == T[v]), r0=R.r, r1=R.r + 0.08, gid=f"ans:{t}")
    nodes(ax, R, gid=f"node:{t}")
    ring(ax, R.xy[hp], R.r + 0.5, col=col(hp == ht), lw=0.6, gid=f"node:{t}")


# ---------------------------------------------------------------------------------------------------- arrays
BP, BW = 1.98, 1.28                                    # bar pitch and width
HB = 6.0                                               # tallest bar (sorts)
ARC_C = 0.6                                            # least clearance of an arc over the bars it passes, mm
ARC_C0, ARC_C1 = 0.35, 0.22                            # control height above the tallest bar: ARC_C0 + ARC_C1 * span
ARC_LW = 0.6


def bar_x(x0, n=10):
    return x0 + WC / 2 + (np.arange(n) - (n - 1) / 2) * BP


def bars(ax, t, x0, base, key, hmax=HB, fill=None):
    """Bars standing on `base` (canvas y grows down); returns their x centres and top y."""
    xs = bar_x(x0, len(key))
    h = key / key.max() * hmax
    for k in range(len(key)):
        f = (fill or {}).get(k)
        rect(ax, xs[k] - BW / 2, base - h[k], BW, h[k], fc=tone(f, 0.85) if f else "white", ec=f or INK,
             lw=0.5 if f else NODE_LW, gid=f"bar:{t}:{k}")
    seg(ax, (xs[0] - BW / 2 - 0.5, base), (xs[-1] + BW / 2 + 0.5, base), INK, NODE_LW, z=3.5, gid=f"base:{t}", cap="butt")
    return xs, base - h


def _bezier(p0, p1, p2, p3, n=81):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def order_arcs(xs, tops, pred, base):
    """The sorted order as arcs: each a cubic Bezier from the top centre of one bar to the top centre of the next larger
    key's bar, both control points directly above their endpoints (vertical take-off and landing) at one height H.
    H starts at ARC_C0 + ARC_C1 * span (in bars) mm above the tallest bar from one end to the other (ends included), so
    shorter links nest under longer ones; because the curve itself stays below its control points, H is then raised in
    0.05 mm steps until the drawn curve is at least ARC_C mm from every bar it passes (outline and inside). checks_f5.py
    verifies the same on the drawn paths. Returns {v: (u, polyline [m, 2], lift)}, lift = how far H had to be raised."""
    out = {}
    for v in range(len(xs)):
        u = int(pred[v])
        if u == v:
            continue
        a, b = sorted((u, v))
        nb = (abs(xs[b] - xs[a])) / BP
        H0 = min(tops[a:b + 1]) - ARC_C0 - ARC_C1 * nb      # y grows down: the smallest y is the tallest bar
        H = H0
        while True:
            P = _bezier(np.array([xs[a], tops[a]]), np.array([xs[a], H]), np.array([xs[b], H]), np.array([xs[b], tops[b]]))
            segs = np.stack([P[:-1], P[1:]], 1)
            near = [float(kit._seg_rect(segs, xs[k] - BW / 2, tops[k], xs[k] + BW / 2, base).min()) for k in range(a + 1, b)]
            if not near or min(near) >= ARC_C:
                break
            H -= 0.05
        out[v] = (u, P, H0 - H)
    return out


def draw_sort(t):
    def draw(ax, x0, y0, rings):
        z = Z[t]
        key, P, T = z["x_key"][IDX], z["pred_pred"][IDX], z["pred"][IDX]
        xs, tops = bars(ax, t, x0, y0 + HC - 0.2, key)
        arcs = order_arcs(xs, tops, P, y0 + HC - 0.2)
        for v, (u, poly, lift) in arcs.items():
            ax.plot(poly[:, 0], poly[:, 1], color=col(P[v] == T[v]), lw=ARC_LW, zorder=4, solid_capstyle="butt",
                    solid_joinstyle="round", gid=f"arc:{t}:{v}:{u}")
        return dict(xs=xs, tops=tops, arcs=arcs)
    return draw


def marker(ax, x, ytip, c, gid):
    ax.add_patch(Polygon([[x - 0.62, ytip - 1.0], [x + 0.62, ytip - 1.0], [x, ytip]], closed=True, fc=c, ec=c, lw=0.2,
                         zorder=6, joinstyle="miter", gid=gid))


def draw_pick(t, k):
    def draw(ax, x0, y0, rings):
        z = Z[t]
        p, tr = int(z[f"pred_{k}"][IDX]), int(z[k][IDX])
        xs, tops = bars(ax, t, x0, y0 + HC - 0.2, z["x_key"][IDX], hmax=HB + 2.2, fill={p: col(p == tr)})
        marker(ax, xs[p], tops[p] - 0.6, col(p == tr), f"ans:{t}")
    return draw


def draw_search(ax, x0, y0, rings):
    """Sorted keys; the target as a dashed line at its height; the position the answer returns marked."""
    t = "binary_search"
    z = Z[t]
    key, tg, p, tr = z["x_key"][IDX], float(z["x_target"][IDX]), int(z["pred_return"][IDX]), int(z["return"][IDX])
    base = y0 + HC - 0.2
    xs, tops = bars(ax, t, x0, base, key, hmax=HB + 2.2, fill={p: col(p == tr)})
    y = base - tg / key.max() * (HB + 2.2)
    ax.plot([xs[0] - BW / 2 - 0.5, xs[-1] + BW / 2 + 0.5], [y, y], color=SOFT, lw=0.5, zorder=5, dashes=(1.6, 1.1),
            solid_capstyle="butt", gid=f"ans:{t}")
    marker(ax, xs[p], tops[p] - 0.6, col(p == tr), f"ans:{t}")
    return dict(target_mm_above_base=base - y)


def draw_kadane(ax, x0, y0, rings):
    t = "find_maximum_subarray_kadane"
    z = Z[t]
    key = z["x_key"][IDX]
    ps, pe, ts, te = int(z["pred_start"][IDX]), int(z["pred_end"][IDX]), int(z["start"][IDX]), int(z["end"][IDX])
    ok = (ps, pe) == (ts, te)
    xs = bar_x(x0)
    s = 5.0 / np.abs(key).max()
    y0b = y0 + 1.6 + 5.0 + 1.6
    for k in range(10):
        on = ps <= k <= pe
        h = key[k] * s
        rect(ax, xs[k] - BW / 2, y0b - max(h, 0), BW, abs(h), fc=tone(col(ok), 0.22) if on else "white",
             ec=col(ok) if on else INK, lw=0.5 if on else NODE_LW, gid=f"bar:{t}:{k}")
    seg(ax, (xs[0] - BW / 2 - 0.5, y0b), (xs[-1] + BW / 2 + 0.5, y0b), INK, NODE_LW, z=3.5, gid=f"base:{t}", cap="butt")
    yb = y0b - key[ps:pe + 1].max() * s - 1.0
    xa, xb = xs[ps] - BW / 2, xs[pe] + BW / 2
    ax.plot([xa, xa, xb, xb], [yb + 0.55, yb, yb, yb + 0.55], color=col(ok), lw=ANS_LW, zorder=5, solid_joinstyle="miter",
            solid_capstyle="butt", gid=f"ans:{t}")


# ---------------------------------------------------------------------------------------------------- intervals
def spans(ax, t, x0, y0, lo, hi, P, T, width=None):
    xm = lambda v: x0 + 0.6 + v * (WC - 1.2)              # noqa: E731
    n = len(lo)
    ys = y0 + 0.7 + np.arange(n) * (HC - 2.2) / (n - 1)
    seg(ax, (xm(0), y0 + HC - 0.2), (xm(1), y0 + HC - 0.2), INK, NODE_LW, z=2, gid=f"base:{t}", cap="butt")
    for k in range(n):
        on = P[k] == 1 or T[k] == 1
        c = col(P[k] == T[k]) if on else FAINT
        if width is not None:
            lw = 0.35 + 1.25 * width[k]
        else:
            lw = 1.0 if on else 0.6
        g = f"ans:{t}" if on else f"in:{t}"
        seg(ax, (xm(lo[k]), ys[k]), (xm(hi[k]), ys[k]), c, lw, z=4 if on else 3, gid=g, cap="butt")
        for v in (lo[k], hi[k]):
            seg(ax, (xm(v), ys[k] - 0.38), (xm(v), ys[k] + 0.38), c, 0.45, z=4 if on else 3, gid=g, cap="butt")


def draw_activity(ax, x0, y0, rings):
    t = "activity_selector"
    z = Z[t]
    spans(ax, t, x0, y0, z["x_s"][IDX], z["x_f"][IDX], z["pred_selected"][IDX], z["selected"][IDX])


def draw_tasks(ax, x0, y0, rings):
    """Each task may run until its deadline (the span's right end); its penalty is the stroke's weight."""
    t = "task_scheduling"
    z = Z[t]
    d, w = z["x_d"][IDX], z["x_w"][IDX]
    spans(ax, t, x0, y0, np.zeros(len(d)), d / d.max(), z["pred_selected"][IDX], z["selected"][IDX], width=w / w.max())


# ---------------------------------------------------------------------------------------------------- tables
CELL_GAP = 2 * QPX                                     # white gap between table cells: 2 px at 300 dpi, 4 at 600


def draw_triangle(t, key, lo, hi, first):
    """Cell (i, j): the split the answer chooses for the sub-problem i..j, tinted by where it falls in its range. Cells
    sit on the 300-dpi pixel grid (pitch and origin in whole pixels), filled without strokes and separated by a
    CELL_GAP of white, so no edge is drawn twice and no sliver appears between cells."""
    def draw(ax, x0, y0, rings):
        z = Z[t]
        P, T = z[f"pred_{key}"][IDX], z[key][IDX]
        n = hi - lo
        s = math.floor(min((HC - 0.4) / n, 1.75) / QPX) * QPX
        X0, Y0 = snap(x0 + WC / 2 - n * s / 2), snap(y0 + (HC - n * s) / 2)
        wrong = []
        for i in range(lo, hi):
            for j in range(i + first, hi + 1):
                k, ok = int(P[i, j]), P[i, j] == T[i, j]
                rel = (k - i) / max(j - first - i, 1) if j - first > i else 0.0
                cx, cy = X0 + (j - lo - first) * s + CELL_GAP / 2, Y0 + (i - lo) * s + CELL_GAP / 2
                if ok:
                    rect(ax, cx, cy, s - CELL_GAP, s - CELL_GAP, fc=tone(ANS, 0.07 + 0.50 * float(np.clip(rel, 0, 1))),
                         ec="none", lw=0, gid=f"ans:{t}")
                else:                                        # the wrong cell: a navy cross on white, outlined
                    cross_cell(ax, cx, cy, s - CELL_GAP, f"ans:{t}")
                    wrong.append((i, j, k, int(T[i, j])))
        return dict(wrong=wrong, cell_px=round(s / QPX, 6), origin_px=(round(X0 / QPX, 6), round(Y0 / QPX, 6)))
    return draw


LETTER_DY = 0.0                                        # (the glyph outlines centre themselves)
GLYPH_MM = LABEL / PT                                  # 6.5 pt in mm: the strings' letters are set at the label size
_FP = FontProperties(family=plt.rcParams["font.sans-serif"][0])
_XH = float(np.ptp(TextPath((0, 0), "x", size=GLYPH_MM, prop=_FP).vertices[:, 1]))   # x-height, mm


def letter(fig, ax, x, y, s, gid, color=INK):
    """A letter of a string in a table cell (or beside a table), drawn as its glyph outline (a mark of the table, not a
    label): centred on x, its x-height band centred on y (page mm, y down)."""
    tp = TextPath((0, 0), s, size=GLYPH_MM, prop=_FP)
    v = tp.vertices.copy()
    v[:, 1] = -v[:, 1]                                  # the canvas runs y down
    v[:, 0] += x - (v[:, 0].min() + v[:, 0].max()) / 2
    v[:, 1] += y + _XH / 2
    pp = PathPatch(MPath(v, tp.codes), fc=color, ec="none", lw=0, zorder=8, gid=gid)
    ax.add_patch(pp)
    return pp


def draw_lcs(fig):
    def draw(ax, x0, y0, rings):
        t = "lcs_length"
        z = Z[t]
        stg, key = z["x_string"][IDX], z["x_key"][IDX].argmax(-1)
        X, Y = np.flatnonzero(stg == 0), np.flatnonzero(stg == 1)
        P, T = z["pred_b"][IDX], z["b"][IDX]
        s = math.floor(2.15 / QPX) * QPX                    # cell pitch in whole pixels
        X0 = snap(x0 + WC / 2 - (len(Y) * s) / 2 + 1.0)
        Y0 = snap(y0 + HC - len(X) * s - 0.1)
        path, i, j = set(), len(X) - 1, len(Y) - 1          # the trace the answer table gives
        while i >= 0 and j >= 0:
            path.add((i, j))
            b = int(P[X[i], Y[j]])
            i, j = (i - 1, j - 1) if b == 0 else (i - 1, j) if b == 1 else (i, j - 1)
        g = f"cells:{t}"
        for a in range(len(X)):
            letter(fig, ax, X0 - 1.3, Y0 + (a + 0.5) * s + LETTER_DY, "abcd"[key[X[a]]], g, color=SOFT)
            for b_ in range(len(Y)):
                ok = P[X[a], Y[b_]] == T[X[a], Y[b_]]
                on = (a, b_) in path
                if on:                                       # path cells tinted; fill only (the grid is drawn once)
                    rect(ax, X0 + b_ * s, Y0 + a * s, s, s, fc=tone(col(ok), 0.16), ec="none", lw=0, gid=g)
                c = np.array([X0 + (b_ + 0.5) * s, Y0 + (a + 0.5) * s])
                dvec = {0: np.array([-1, -1]) * 0.47, 1: np.array([0, -1]) * 0.62, 2: np.array([-1, 0]) * 0.62}[int(P[X[a], Y[b_]])]
                arrow(ax, c - dvec, c + dvec, col=col(ok) if (on or not ok) else tone(ANS, 0.45), lw=0.45, z=5, gid=g,
                      head=0.6, hw=0.5)
        GRID_LW = 2 * QPX * PT                               # 2 px at 300 dpi, centred on pixel boundaries (0.48 pt)
        for a in range(len(X) + 1):
            seg(ax, (X0, Y0 + a * s), (X0 + len(Y) * s, Y0 + a * s), tone(INK, 0.28), GRID_LW, z=3.2, gid=g, cap="butt")
        for b_ in range(len(Y) + 1):
            seg(ax, (X0 + b_ * s, Y0), (X0 + b_ * s, Y0 + len(X) * s), tone(INK, 0.28), GRID_LW, z=3.2, gid=g,
                cap="butt")
        for b_ in range(len(Y)):
            letter(fig, ax, X0 + (b_ + 0.5) * s, Y0 - 1.4, "abcd"[key[Y[b_]]], g, color=SOFT)
        return dict(path=sorted(path), cell_px=round(s / QPX, 6), origin_px=(round(X0 / QPX, 6), round(Y0 / QPX, 6)))
    return draw


# ---------------------------------------------------------------------------------------------------- strings
def draw_match(fig, t):
    """The text as a row of letters; the window where the answer says the pattern first appears boxed; the pattern
    under it."""
    def draw(ax, x0, y0, rings):
        z = Z[t]
        stg, key = z["x_string"][IDX], z["x_key"][IDX].argmax(-1)
        txt, pat = key[stg == 0], key[stg == 1]
        p, tr = int(z["pred_match"][IDX]), int(z["match"][IDX])
        s, bh, gap = 2.1, 2.6, 2.4                      # letter pitch, box height, gap between the rows
        X0 = x0 + WC / 2 - len(txt) * s / 2
        yc0 = y0 + HC / 2 - (bh + gap) / 2              # centre line of the text row
        yc1 = yc0 + bh + gap                            # centre line of the pattern row
        g = f"cells:{t}"
        c = col(p == tr)
        for k, ch in enumerate(txt):
            letter(fig, ax, X0 + (k + 0.5) * s, yc0 + LETTER_DY, "abcd"[ch], g)
        seg(ax, (X0, yc0 + bh / 2), (X0 + len(txt) * s, yc0 + bh / 2), tone(INK, 0.3), 0.35, z=2, gid=g, cap="butt")
        rect(ax, X0 + p * s, yc0 - bh / 2, len(pat) * s, bh, fc="none", ec=c, lw=ANS_LW, z=6, gid=g)
        rect(ax, X0 + p * s, yc1 - bh / 2, len(pat) * s, bh, fc=tone(c, 0.14), ec=c, lw=0.5, z=4, gid=g)
        for k, ch in enumerate(pat):
            letter(fig, ax, X0 + (p + k + 0.5) * s, yc1 + LETTER_DY, "abcd"[ch], g)
            seg(ax, (X0 + (p + k + 0.5) * s, yc0 + bh / 2 + 0.35), (X0 + (p + k + 0.5) * s, yc1 - bh / 2 - 0.35), c, 0.45,
                z=3, gid=g, cap="butt")
    return draw


# ---------------------------------------------------------------------------------------------------- points
def fitpts(x0, y0, x, y, pad=1.2):
    P = np.stack([x, -y], 1).astype(float)
    P = P - (P.max(0) + P.min(0)) / 2
    s = min((WC - 2 * pad) / np.ptp(P[:, 0]), (HC - 2 * pad) / np.ptp(P[:, 1]))
    return P * s + [x0 + WC / 2, y0 + HC / 2]


def draw_hull(t):
    def draw(ax, x0, y0, rings):
        z = Z[t]
        Q = fitpts(x0, y0, z["x_x"][IDX], z["x_y"][IDX], pad=0.9)
        P, T = z["pred_in_hull"][IDX], z["in_hull"][IDX]
        on = np.flatnonzero(P == 1)
        c0 = Q[on].mean(0)
        on = on[np.argsort(np.arctan2(*(Q[on] - c0).T[::-1]))]
        good = bool((P == T).all())
        ax.add_patch(Polygon(Q[on], closed=True, fc=tone(col(good), 0.10), ec=col(good), lw=ANS_LW, zorder=3,
                             joinstyle="round", gid=f"ans:{t}"))
        for k in sorted(range(len(Q)), key=lambda k: P[k]):  # the points inside first, the hull's points over them
            ok = P[k] == T[k]
            node(ax, Q[k], 0.42, fc=col(ok) if P[k] else "white", ec=col(ok) if (P[k] or not ok) else INK, gid=f"node:{t}")
    return draw


def draw_segments(ax, x0, y0, rings):
    t = "segments_intersect"
    z = Z[t]
    Q = fitpts(x0, y0, z["x_x"][IDX], z["x_y"][IDX], pad=1.6)
    p, tr = int(z["pred_intersect"][IDX]), int(z["intersect"][IDX])
    for a, b in ((0, 1), (2, 3)):
        seg(ax, Q[a], Q[b], INK, 0.6, z=3, gid=f"in:{t}")
    for k in range(4):
        node(ax, Q[k], 0.36, gid=f"node:{t}")
    d1, d2 = Q[1] - Q[0], Q[3] - Q[2]
    den = d1[0] * d2[1] - d1[1] * d2[0]
    tt = ((Q[2][0] - Q[0][0]) * d2[1] - (Q[2][1] - Q[0][1]) * d2[0]) / den
    X = Q[0] + tt * d1
    if p and tr:                                          # "they cross", right: the crossing as a cobalt dot
        node(ax, X, 0.42, fc=ANS, ec=ANS, z=7, gid=f"ans:{t}")
    elif p:                                               # "they cross", wrong
        ring(ax, Q.mean(0), 1.05, col=ANS, lw=0.6, gid=f"ans:{t}", ls=(0, (1.0, 0.8)))
    elif tr:
        ring(ax, X, 1.05, col=ANS, lw=0.6, gid=f"ans:{t}", ls=(0, (1.0, 0.8)))
    return dict(crossing_to_nearest_end_mm=float(np.min(np.hypot(*(Q - X).T))))


# ---------------------------------------------------------------------------------------------------- panel a
def rings_for():
    """Ring order for every distinct graph (drawing choice; see order10 / order24)."""
    out, info = {}, {}
    shared = ["bfs", "bellman_ford", "dijkstra", "mst_prim", "floyd_warshall"]
    adj = Z["bellman_ford"]["adj"][IDX]
    assert all((Z[t]["adj"][IDX] == adj).all() for t in shared)
    slot, sc = order10(und_edges(adj), [])
    for t in shared:
        out[t] = slot
    info["shared"] = sc

    def ptr_edges(pred):
        return [(int(v), int(pred[v])) for v in range(len(pred)) if pred[v] != v]
    for t, ans in (("dfs", ptr_edges(Z["dfs"]["pred_pi"][IDX])),
                   ("dag_shortest_paths", ptr_edges(Z["dag_shortest_paths"]["pred_pi"][IDX])),
                   ("topological_sort", ptr_edges(Z["topological_sort"]["pred_topo"][IDX])),
                   ("strongly_connected_components", ptr_edges(Z["strongly_connected_components"]["pred_scc_id"][IDX]))):
        out[t], info[t] = order10(und_edges(Z[t]["adj"][IDX]), ans)
    adj24 = Z["articulation_points"]["adj"][IDX]
    assert all((Z[t]["adj"][IDX] == adj24).all() for t in SPARSE_T)
    slot24, sc24 = sparse_layout(len(adj24), und_edges(adj24))
    for t in SPARSE_T:
        out[t] = slot24
    info["sparse24"] = sc24
    return out, info


def build_draw(fig):
    return {
        "bfs": draw_tree("bfs"), "dfs": draw_tree("dfs", src=False), "topological_sort": draw_topo,
        "articulation_points": draw_cut, "bridges": draw_edges("bridges", "is_bridge"),
        "strongly_connected_components": draw_scc,
        "bellman_ford": draw_tree("bellman_ford"), "dijkstra": draw_tree("dijkstra"),
        "dag_shortest_paths": draw_tree("dag_shortest_paths"), "floyd_warshall": draw_fw,
        "mst_kruskal": draw_edges("mst_kruskal", "in_mst"), "mst_prim": draw_tree("mst_prim"),
        "insertion_sort": draw_sort("insertion_sort"), "bubble_sort": draw_sort("bubble_sort"),
        "heapsort": draw_sort("heapsort"), "quicksort": draw_sort("quicksort"), "minimum": draw_pick("minimum", "min"),
        "binary_search": draw_search, "quickselect": draw_pick("quickselect", "median"),
        "activity_selector": draw_activity, "task_scheduling": draw_tasks,
        "find_maximum_subarray_kadane": draw_kadane,
        "matrix_chain_order": draw_triangle("matrix_chain_order", "s", 1, 9, 1),
        "optimal_bst": draw_triangle("optimal_bst", "root", 0, 10, 1), "lcs_length": draw_lcs(fig),
        "naive_string_matcher": draw_match(fig, "naive_string_matcher"), "kmp_matcher": draw_match(fig, "kmp_matcher"),
        "segments_intersect": draw_segments, "graham_scan": draw_hull("graham_scan"), "jarvis_march": draw_hull("jarvis_march"),
    }


