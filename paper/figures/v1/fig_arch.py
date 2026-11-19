"""The architecture figure, "The cold check is the block": what a cold attention block is, and how this paper trains and
runs it, on one real example throughout: the CLRS-30 breadth-first-search check, fitted at the completed answer, run
cold on a ten-node input of the library's generator (data/arch_bfs.json, from data/arch_bfs.py, which uses
code/clrs/run.py and cold.py unchanged). 183 x H mm. Cobalt: the cold block's bits, counts and pointers; rose: the warm
read of the same scores (for contrast) and the warm search on bits the cold stage leaves open; navy: inputs.

a  The block. Left, the input graph with the answer at rest (each node's parent; the source filled). Middle, the pair
   state: one bit per ordered pair (u, v), "u is the parent of v" (cobalt), with the completion register beside it
   (depth: the number of ancestors along the pointers, recomputed from the bits every loop). Right, one loop of the block
   at rest, every table computed from the state: the pair's own tests (four bit tables), the witness test (row u of
   leg 1 meets column v of leg 2 over the third node k: the count N(u, v), an exact whole number, and the bit [N = 0]),
   their conjunction, and the read at zero temperature, which keeps for every v the best-scoring u that passes and the
   number that pass. Inset: the same scores of node 8's candidates read warm (softmax, T = 1) and cold.
b  The fit at the answer: one training answer (7 nodes), completed from itself (depth in each node); each pair bit is
   predicted from the rest, and the fit keeps every condition that held on every training parent: the check.
c  The check run cold from nothing on the input of a: the pair state after each loop, until it stops changing.
d  Where bits stay open, a search: the 9 x 9 Sudoku-Extreme puzzle of Fig. 4a after its cold stage, and solved; the
   check is the stopping test.
    ./run.sh fig_arch.py
"""
import json
import sys
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, TITLE, tone, text, head, W
from draw import seg, arrow, curve_arrow, node, ring
import tiles
import results as R

D = json.load(open(kit.DATA / "arch_bfs.json"))
PB = json.load(open(kit.DATA / "puzzle_boards.json"))
RUN = D["run"]
N = RUN["n"]
SRC = RUN["source"]
ANS = np.array(RUN["answer"])
E = np.zeros((N, N), bool)
for a, b in RUN["edges"]:
    E[a, b] = E[b, a] = True
LAST = RUN["loops"][-1]
DEPTH = np.array([np.nan if x is None else x for x in LAST["depth"]], float)
assert (np.array(LAST["pointers"]) == ANS).all()
PTR = np.zeros((N, N), bool)
for v in range(N):
    if ANS[v] != v:
        PTR[ANS[v], v] = True
LEG1 = np.array(D["witness"]["leg1"]); LEG2 = np.array(D["witness"]["leg2"]); NW = np.array(D["witness"]["count"])
PASS = np.array(D["read"]["passes"])
OFF = ~np.eye(N, dtype=bool)
OWN = [("edge(u, v)", E & OFF),
       ("depth u < depth v", (DEPTH[:, None] < DEPTH[None, :]) & OFF),
       ("u has a depth", np.isfinite(DEPTH)[:, None] & OFF),
       ("v not the source", (np.arange(N) != SRC)[None, :] & OFF)]
WIT0 = (NW == 0) & OFF
ALL = OWN[0][1] & OWN[1][1] & OWN[2][1] & OWN[3][1] & WIT0
assert (ALL == PASS).all()                                        # the five tables multiply to the check's own decision
assert (PASS == PTR).all()                                        # and at rest the check passes exactly the answer
FIT_MS = R.clrs("bfs")["fit_ms"]
LOOPS = RUN["loops_to_rest"]

H = 136.0
fig = kit.figure(H)
cv = kit.canvas(fig, 0, 0, W, H)
GAP = 0.25                                                        # white gap inside bit tables, mm


def table(x, y, M, cell, col=COB, frame=True, vals=None, valcol=None, gid=None):
    """An N x N bit table at (x, y), rows u, columns v; filled cells in `col`."""
    n = M.shape[0]
    cv.add_patch(Rectangle((x, y), n * cell, n * cell, fc="white", ec="none", zorder=1))
    for i in range(n):
        for j in range(n):
            if i == j:
                cv.add_patch(Rectangle((x + j * cell + GAP / 2, y + i * cell + GAP / 2), cell - GAP, cell - GAP,
                                       fc=tone(INK, 0.06), ec="none", zorder=2))
            elif M[i, j]:
                cv.add_patch(Rectangle((x + j * cell + GAP / 2, y + i * cell + GAP / 2), cell - GAP, cell - GAP,
                                       fc=col, ec="none", zorder=2))
            if vals is not None and i != j:
                cv.text(x + (j + 0.5) * cell, y + (i + 0.5) * cell, str(vals[i, j]), ha="center", va="center",
                        fontsize=LABEL, color=valcol(vals[i, j]) if valcol else INK, zorder=4)
    if frame:
        cv.add_patch(Rectangle((x, y), n * cell, n * cell, fc="none", ec=tone(INK, 0.55), lw=0.4, zorder=3))


def graph_xy(cx, cy, R):
    slot = tiles.rings_for()[0]["bfs"]
    return tiles.ring_points(N, cx, cy, R)[slot]


NUMS = dict(__import__("re").findall(r"\\nmdef\{([^}]*)\}\{([^}]*)\}", (kit.ROOT / "paper" / "numbers.tex").read_text()))
LOOPED_MIN = float(NUMS["zero.gen.minutes"])                      # our earlier paper: the general block's median training time per task
assert LOOPED_MIN == 64, LOOPED_MIN
assert abs(float(NUMS["bfs.clrs.fit"]) - round(R.clrs("bfs")["fit_ms"])) < 0.5, NUMS["bfs.clrs.fit"]


def lab2(x, y, l1, l2, col=INK):
    text(fig, x, y, l1, ha="center", color=col)
    text(fig, x, y + 3.1, l2, ha="center", color=col)


def panel_a():
    y0 = 4.4
    head(fig, 0, y0, "a", "The cold block")
    text(fig, 4.6, y0 + 4.2, "one bit for every ordered pair, read through every third node, rebuilt in every loop", color=SOFT)
    P = graph_xy(13.5, y0 + 27.0, 10.0)
    for a, b in RUN["edges"]:
        if not (ANS[b] == a or ANS[a] == b):
            seg(cv, P[a], P[b], tone(GREY, 0.6), HAIR, z=2)
    for v in range(N):
        if ANS[v] != v:
            arrow(cv, P[ANS[v]], P[v], INK, 0.7, r0=1.35, r1=1.35, head=0.85, hw=0.7, gid=f"e{v}")
    for v in range(N):
        node(cv, P[v], 1.3, fc=INK if v == SRC else "white", gid=f"g{v}")
        kit.names(fig, cv.text(*P[v], str(v), ha="center", va="center", fontsize=LABEL, color="white" if v == SRC else INK,
                               zorder=9), f"g{v}")
    lab2(13.5, y0 + 43.0, "input graph and, at rest,", "each node's parent", SOFT)
    cell = 2.0
    X, Y = 32.5, y0 + 16.5
    table(X, Y, PTR, cell)
    for i in range(N):
        cv.text(X - 0.9, Y + (i + 0.5) * cell, str(i), ha="right", va="center", fontsize=LABEL, color=SOFT)
        cv.text(X + (i + 0.5) * cell, Y - 1.0, str(i), ha="center", va="baseline", fontsize=LABEL, color=SOFT)
    text(fig, X + N * cell / 2, Y - 5.6, "pair state", ha="center", size=TITLE)
    xd = X + N * cell + 1.4
    for i in range(N):
        cv.add_patch(Rectangle((xd, Y + i * cell + GAP / 2), cell, cell - GAP, fc=tone(COB, 0.12), ec="none", zorder=2))
        cv.text(xd + cell / 2, Y + (i + 0.5) * cell, "" if np.isnan(DEPTH[i]) else str(int(DEPTH[i])), ha="center",
                va="center", fontsize=LABEL, color=COB, zorder=4)
    lab2(X + N * cell / 2, Y + N * cell + 3.6, "$u$ parent of $v$", "(row $u$, column $v$)", COB)
    lab2(xd + cell / 2 + 1.0, Y + N * cell + 3.6, "depth", "", COB)
    bx, by, bw, bh = 62.5, y0 + 6.5, W - 62.5, 60.0
    cv.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=2.0", fc=tone(INK, 0.035),
                                ec="none", zorder=0))
    arrow(cv, (xd + cell + 0.8, Y + N * cell / 2), (bx + 2.2, Y + N * cell / 2), INK, 0.6)
    text(fig, bx + 3.0, by + 4.8, "One block: the fitted check, the same in every loop", size=TITLE)
    c = 1.02
    tw = N * c
    pitch = 18.4
    x0 = bx + 6.0
    yt = by + 12.0
    text(fig, x0, yt - 1.8, "the pair's own tests", color=COB)
    labs = [("an edge", "$u$–$v$"), ("depth $u$", "< depth $v$"), ("$u$ has", "a depth"), ("$v$ is not", "the source")]
    for k, ((l1, l2), (_, M)) in enumerate(zip(labs, OWN)):
        xx = x0 + k * pitch
        table(xx, yt, M, c)
        lab2(xx + tw / 2, yt + tw + 3.6, l1, l2, COB if k == 1 else INK)
    yw = by + 36.0
    text(fig, x0, yw - 1.8, "the witness test, through every third node $k$", color=COB)
    wl = [("$k$ before $u$", "(row $u$, col. $k$)"), ("edge $k$–$v$, $k$", "nearer the source"),
          ("$N(u,v)$: how", "many such $k$"), ("$N = 0$", "")]
    vals = np.where(OFF, NW, 0)
    for k, (l1, l2) in enumerate(wl):
        xx = x0 + k * pitch
        if k == 0:
            table(xx, yw, LEG1, c)
        elif k == 1:
            table(xx, yw, LEG2, c)
        elif k == 2:
            table(xx, yw, np.zeros((N, N), bool), c)
            for i in range(N):
                for j in range(N):
                    if i != j and NW[i, j] > 0:
                        cv.add_patch(Rectangle((xx + j * c + GAP / 2, yw + i * c + GAP / 2), c - GAP, c - GAP,
                                               fc=tone(COB, min(1.0, 0.3 * NW[i, j])), ec="none", zorder=2))
        else:
            table(xx, yw, WIT0, c)
        lab2(xx + tw / 2, yw + tw + 3.6, l1, l2)
        if k < 3:
            arrow(cv, (xx + tw + 1.3, yw + tw / 2), (xx + pitch - 1.3, yw + tw / 2), INK, 0.5, head=0.8, hw=0.6)
    xa = x0 + 4 * pitch + 2.0
    ya = by + 24.0
    arrow(cv, (x0 + 3 * pitch + tw + 1.0, yt + tw / 2), (xa - 1.0, ya + tw / 2 - 1.2), INK, 0.5, head=0.8, hw=0.6)
    arrow(cv, (x0 + 3 * pitch + tw + 1.0, yw + tw / 2), (xa - 1.0, ya + tw / 2 + 1.2), INK, 0.5, head=0.8, hw=0.6)
    table(xa, ya, PASS, c)
    lab2(xa + tw / 2, ya - 4.4, "all five", "", INK)
    text(fig, xa + tw + 3.0, ya + 1.0, "read at zero", color=COB)
    text(fig, xa + tw + 3.0, ya + 3.9, "temperature:", color=COB)
    text(fig, xa + tw + 3.0, ya + 6.8, "for each $v$, the best", color=COB)
    text(fig, xa + tw + 3.0, ya + 9.7, "$u$ that passes, and", color=COB)
    text(fig, xa + tw + 3.0, ya + 12.6, "how many pass", color=COB)
    # write, recompute depth, loop
    yl = by + bh + 4.4
    seg(cv, (xa + tw / 2, ya + tw + 0.8), (xa + tw / 2, yl), COB, 0.7, cap="butt")
    seg(cv, (xa + tw / 2, yl), (X + N * cell / 2, yl), COB, 0.7, cap="butt")
    arrow(cv, (X + N * cell / 2, yl), (X + N * cell / 2, Y + N * cell + 8.6), COB, 0.7)
    text(fig, xa + tw / 2 - 2.0, yl - 1.5, "write the pointers, recompute depth; loop until nothing changes",
         ha="right", color=COB)
    assert all(p == 1 for v, p in enumerate(LAST["passing"]) if ANS[v] != v)


def panel_b(y0):
    head(fig, 0, y0, "b", "The fit at the answer")
    text(fig, 4.6, y0 + 4.2, "each pair bit predicted from the rest", color=SOFT)
    T = D["train"]; n = T["n"]; dep = T["depth"]; ans = T["answer"]
    rows = {}
    for v in range(n):
        rows.setdefault(dep[v], []).append(v)
    P = {}
    for d, vs in rows.items():
        for i, v in enumerate(sorted(vs)):
            P[v] = np.array([12.5 + (i - (len(vs) - 1) / 2) * 5.6, y0 + 11.0 + d * 6.6])
    for a, b in T["edges"]:
        if not (ans[b] == a or ans[a] == b):
            seg(cv, P[a], P[b], tone(GREY, 0.6), HAIR, z=2)
    for v in range(n):
        if ans[v] != v:
            arrow(cv, P[ans[v]], P[v], INK, 0.7, r0=1.45, r1=1.45, head=0.85, hw=0.7, gid=f"te{v}")
    for v in range(n):
        node(cv, P[v], 1.4, fc=INK if v == T["source"] else "white", gid=f"t{v}")
        kit.names(fig, cv.text(*P[v], str(dep[v]), ha="center", va="center", fontsize=LABEL,
                               color="white" if v == T["source"] else COB, zorder=9), f"t{v}")
    lab2(12.5, y0 + 37.4, "a training answer,", "completed: depth", SOFT)
    arrow(cv, (25.0, y0 + 21.0), (29.5, y0 + 21.0), INK, 0.6)
    xc, yc = 31.0, y0 + 11.0
    text(fig, xc, yc - 0.8, "kept: true at every training parent", color=INK)
    lines = ["depth $u$ < depth $v$", "an edge $u$–$v$", "$u$ has a depth", "$v$ is not the source",
             "no $k$ before $u$ that is a neighbour", "    of $v$ nearer the source"]
    assert len(D["fit"]["conditions"]) == 5
    for i, l in enumerate(lines):
        text(fig, xc + 1.6, yc + 3.6 + i * 3.1, l, color=COB if i == 0 else INK)
    seg(cv, (xc, yc + 1.3), (xc, yc + 4.4), COB, 0.6, cap="butt")
    # training time: fitted against looped
    ya = y0 + 35.5
    xl0, xl1 = xc, xc + 40.0
    lo, hi = np.log10(0.01), np.log10(3600.0)
    f = lambda t: xl0 + (np.log10(t) - lo) / (hi - lo) * (xl1 - xl0)  # noqa: E731
    text(fig, xl0, ya - 1.2, f"fitted at the answer: {FIT_MS:.0f} ms, one core", color=COB)
    seg(cv, (xl0, ya + 0.6), (f(FIT_MS / 1e3), ya + 0.6), COB, 1.6, cap="butt")
    text(fig, xl0, ya + 4.6, "trained through its loops (earlier paper): about an hour", color=ROSE)
    seg(cv, (xl0, ya + 6.4), (f(LOOPED_MIN * 60), ya + 6.4), ROSE, 1.6, cap="butt")
    yx = ya + 8.6
    seg(cv, (xl0, yx), (max(xl1, f(LOOPED_MIN * 60)), yx), INK, HAIR)
    for t, l in ((0.01, "10 ms"), (1, "1 s"), (60, "1 min"), (3600, "1 h")):
        seg(cv, (f(t), yx), (f(t), yx + 0.8), INK, HAIR)
        text(fig, f(t), yx + 3.6, l, ha="center")
    return f, ya


def panel_c(x0, y0):
    head(fig, x0, y0, "c", "Run cold from nothing")
    text(fig, x0 + 4.6, y0 + 4.2, "the pair state of a after each loop", color=SOFT)
    c = 0.95
    tw = N * c
    pitch = tw + 3.0
    for li, lp in enumerate(RUN["loops"]):
        M = np.zeros((N, N), bool)
        for v, u in enumerate(lp["pointers"]):
            if u != v:
                M[u, v] = True
        r, k = divmod(li, 4)
        x = x0 + 0.6 + k * pitch
        y = y0 + 9.0 + r * (tw + 7.6)
        table(x, y, M, c)
        text(fig, x + tw / 2, y + tw + 3.4, f"loop {li + 1}" if li + 1 < LOOPS else f"loop {li + 1}: rest", ha="center",
             color=INK if li + 1 < LOOPS else COB)


def panel_d(x0, y0):
    head(fig, x0, y0, "d", "Search where bits stay open")
    rec = PB["sudoku"]["9"]
    sol = np.array(rec["solution"]).reshape(9, 9); kind = np.array(rec["kind"]).reshape(9, 9)
    cell = 2.05
    for bi, show in enumerate(("cold", "solved")):
        X, Y = x0 + 0.6 + bi * (9 * cell + 3.0), y0 + 9.0
        for i in range(9):
            for j in range(9):
                k = kind[i, j]
                if show == "cold" and k == 2:
                    continue
                cv.text(X + (j + 0.5) * cell, Y + (i + 0.5) * cell, str(sol[i, j]), ha="center", va="center_baseline",
                        fontsize=LABEL, color=INK if k == 0 else COB if k == 1 else ROSE, zorder=4)
        for i in range(10):
            lw = 0.5 if i % 3 == 0 else 0.15
            cc = INK if i % 3 == 0 else tone(INK, 0.35)
            seg(cv, (X, Y + i * cell), (X + 9 * cell, Y + i * cell), cc, lw, cap="projecting" if lw > 0.3 else "butt")
            seg(cv, (X + i * cell, Y), (X + i * cell, Y + 9 * cell), cc, lw, cap="projecting" if lw > 0.3 else "butt")
    lab2(x0 + 0.6 + 4.5 * cell, y0 + 9.0 + 9 * cell + 3.6, "after the", "cold stage", SOFT)
    lab2(x0 + 0.6 + 9 * cell + 3.0 + 4.5 * cell, y0 + 9.0 + 9 * cell + 3.6, "solved; the check", f"stops it, {rec['moves']} moves", ROSE)


def build():
    panel_a()
    panel_b(84.0)
    panel_c(86.0, 84.0)
    panel_d(139.5, 84.0)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig_arch")
