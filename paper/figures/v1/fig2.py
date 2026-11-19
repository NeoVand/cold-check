"""Fig. 2, "A check is not yet a procedure": why a rule fitted at the answer can fail when run from nothing, and the
fix. 183 x H mm. Cobalt: the check run with the level condition kept; rose: the same check as fitted, and the fit
without any completion; grey: checks read at the completed answer (not runs); navy: graphs.

a  Training graph 0 (12 nodes; as Fig. 1a), run from "not yet" loop by loop: as fitted (upper row) and with the level
   condition kept (lower row). A node is filled once it has a parent and the parent is the answer's, a ring if the parent
   is wrong, white if unsettled; arrows: the parents given so far (arm colour); grey: the graph's other edges. Rows
   of each graph: distance from the source (drawing choice). From data/bfs_map.json (train0, every loop logged).
b  Parents right at 64 nodes (100 test graphs, 6,300 parents), the ablation of the paper's Table 1, read from
   code/walkthrough/completed_answer_bfs.json (completions depth and order) and fit_at_answer.json (no
   completion: bfs_64). Small marks on the kept row: 256 and 1,024 nodes. Open grey mark on the depth row: a valid
   shortest-path parent (any neighbour one step nearer the source).
c  The 1,024-node graph of Fig. 1d,e: parents right after each loop (solid) and nodes settled (dashed), kept and as
   fitted (data/bfs_map.json; per-loop counts asserted against data/bfs_run.json).
d  Loops against levels: on every test graph of 16, 64 and 256 nodes (220 graphs) and the three of 1,024, the kept run
   stops after exactly as many loops as the answer has levels (distance of the farthest node); disc area = graphs.
    ./run.sh fig2.py
"""
import json
import sys
from collections import Counter
import numpy as np
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, LINE, tone, text, head
from draw import seg, arrow, node, ring

D = json.load(open(kit.DATA / "bfs_map.json"))
REF = json.load(open(kit.DATA / "bfs_run.json"))
REC = json.load(open(kit.CODE / "walkthrough" / "completed_answer_bfs.json"))
FAA = json.load(open(kit.CODE / "walkthrough" / "fit_at_answer.json"))
t0 = D["train0"]; g = D["graph1024"]

H = 104.0
fig = kit.figure(H)
Y1, Y2 = 5.0, 62.0


# ------------------------------------------------------------------------------------------------- a
def mini(cv, x0, y0, loop, arm):
    """Training graph 0 after `loop` loops of the run `arm`, top-left at (x0, y0)."""
    depth = np.array(t0["depth"]); place = np.array(t0["place"]); par = np.array(t0["label"]); s = t0["source"]
    run = t0[arm]
    col = COB if arm == "kept" else ROSE
    sl = np.array(run["settle_loop"]); given = np.array(run["parent"])     # a parent, once given, is kept (asserted)
    P = {}
    gap, dy = 4.3, 4.0
    for d in range(depth.max() + 1):
        row = sorted(np.nonzero(depth == d)[0], key=lambda v: place[v])
        for i, v in enumerate(row):
            P[int(v)] = np.array([x0 + 10 + (i - (len(row) - 1) / 2) * gap, y0 + 1.5 + d * dy])
    for a, b in t0["edges"]:
        seg(cv, P[a], P[b], tone(GREY, .45), 0.35, z=2)
    for v in range(len(par)):
        if v != s and 0 < sl[v] <= loop:
            arrow(cv, P[int(given[v])], P[v], col, 0.55, r0=1.05, r1=1.05, head=0.8, hw=0.65)
    for v in range(len(par)):
        if v == s:
            node(cv, P[v], 0.95, fc=INK, lw=0.4)
        elif 0 < sl[v] <= loop:
            ok = given[v] == par[v]
            node(cv, P[v], 0.95, fc=col if ok else "white", ec=col, lw=0.6)
        else:
            node(cv, P[v], 0.95, fc="white", ec=tone(GREY, .8), lw=0.4)


def panel_a(x0, w):
    head(fig, x0, Y1, "a", "From nothing, loop by loop")
    text(fig, x0 + 4.6, Y1 + 4.2, "a training graph; filled, parent right; ring, wrong", color=SOFT)
    cv = kit.canvas(fig, x0, Y1 + 6, w, 46)
    lx = 16.0
    cw = (w - lx) / 5
    for j in range(5):
        cv.text(x0 + lx + j * cw + 10, Y1 + 10.6, f"loop {j + 1}", ha="center", va="baseline", fontsize=LABEL, color=INK)
        mini(cv, x0 + lx + j * cw, Y1 + 13.0, j + 1, "as_fitted")
        mini(cv, x0 + lx + j * cw, Y1 + 33.5, j + 1, "kept")
    cv.text(x0, Y1 + 21.0, "as fitted", va="center", fontsize=LABEL, color=ROSE)
    cv.text(x0, Y1 + 41.3, "level", va="center", fontsize=LABEL, color=COB)
    cv.text(x0, Y1 + 44.8, "condition kept", va="center", fontsize=LABEL, color=COB)


# ------------------------------------------------------------------------------------------------- b
def pct(a, b):
    return 100.0 * a / b


ROWS = [  # label, value, colour, kind
    ("no completion, run from nothing", pct(FAA["bfs_64"]["right"], FAA["bfs_64"]["total"]), ROSE, "run"),
    ("depth, read at the answer", pct(*REC["depth"]["64"]["at_answer"]), GREY, "read"),
    ("order, read at the answer", pct(*REC["order"]["64"]["at_answer"]), GREY, "read"),
    ("order, run as fitted", pct(*REC["order"]["64"]["top"][:2]), ROSE, "run"),
    ("order, run, level condition kept", pct(*REC["order"]["64"]["top_gate"][:2]), COB, "run"),
]
assert [round(r[1], 1) for r in ROWS] == [45.9, 85.3, 100.0, 38.3, 100.0]
assert REC["depth"]["64"]["at_answer_valid"] == REC["depth"]["64"]["at_answer"][1]           # every parent valid
for n in ("256", "1024"):
    assert REC["order"][n]["top_gate"][0] == REC["order"][n]["top_gate"][1]


def panel_b(x0, w):
    head(fig, x0, Y1, "b", "Parents right, 64 nodes")
    ax = kit.axes_mm(fig, x0 + 31, Y1 + 9, w - 40, 36)
    kit.clean(ax)
    ax.set_xlim(0, 100); ax.set_ylim(len(ROWS) - 0.5, -0.6)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS])
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100]); ax.set_xlabel("parents right (%)")
    for i, (lab, v, c, kind) in enumerate(ROWS):
        ax.plot([0, 100], [i, i], color=GUIDE, lw=0.5, zorder=1, clip_on=False)
        ax.plot([0, v], [i, i], color=tone(c, .55), lw=1.2, zorder=2, solid_capstyle="butt", clip_on=False)
        if kind == "read":
            ax.plot(v, i, "o", ms=3.2, mfc="white", mec=c, mew=0.7, zorder=3, clip_on=False)
        else:
            ax.plot(v, i, "o", ms=3.2, mfc=c, mec=c, mew=0.0, zorder=3, clip_on=False)
        ax.text(108, i, f"{v:.1f}", ha="left", va="center_baseline", fontsize=LABEL, color=c if c != GREY else SOFT)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)
    return ax


# ------------------------------------------------------------------------------------------------- c
def panel_c(x0, w):
    head(fig, x0, Y2, "c", "1,024 nodes, loop by loop")
    ax = kit.axes_mm(fig, x0 + 12, Y2 + 5, w - 14, 28)
    kit.clean(ax)
    K, A = g["kept"], g["as_fitted"]
    for arm, ref in (("kept", "kept"), ("as_fitted", "as_fitted")):
        assert g[arm]["settled"] == REF["runs1024"][0][ref]["settled"] and g[arm]["right"] == REF["runs1024"][0][ref]["right"]
    L = len(A["settled"])
    xs = np.arange(1, L + 1)
    k_right = K["right"] + [K["right"][-1]] * (L - len(K["right"]))
    ax.plot(xs, A["settled"], color=ROSE, lw=LINE * 0.8, ls=(0, (3, 1.6)), zorder=3)
    ax.plot(xs, A["right"], color=ROSE, lw=LINE, zorder=3)
    ax.plot(xs, k_right, color=COB, lw=LINE, zorder=4)
    ax.set_xlim(0, L + 0.5); ax.set_ylim(0, 1060)
    ax.set_xticks([1, 10, 20, 30, 40, 52]); ax.set_yticks([0, 500, 1023]); ax.set_yticklabels(["0", "500", "1,023"])
    ax.set_xlabel("loop"); ax.set_ylabel("nodes")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COB, lw=LINE), Line2D([], [], color=ROSE, lw=LINE),
          Line2D([], [], color=ROSE, lw=LINE * 0.8, ls=(0, (3, 1.6)))]
    ax.legend(hs, ["parents right, level condition kept", "parents right, as fitted", "nodes settled, as fitted"],
              loc="lower right", bbox_to_anchor=(1.0, 0.31), frameon=False, fontsize=LABEL, handlelength=2.2,
              borderaxespad=0.2, labelspacing=0.35)
    assert A["settled"][0] == 776 and A["right"][0] == 222 and A["right"][-1] == 312 and K["right"][8] == 1023
    return ax


# ------------------------------------------------------------------------------------------------- d
def panel_d(x0, w):
    head(fig, x0, Y2, "d", "Loops against levels")
    ax = kit.axes_mm(fig, x0 + 9, Y2 + 5, w - 11, 28)
    kit.clean(ax)
    lv = [(lev, loops) for n, lev, loops, ok in D["loops_levels"]]
    assert all(ok for *_, ok in D["loops_levels"]) and len(lv) == 220
    for r in REF["runs1024"]:
        lv.append((len(r["levels"]) - 1, r["kept"]["loops"]))
    C = Counter(lv)
    assert all(a == b for a, b in C)
    m = max(a for a, _ in C) + 1
    ax.plot([0, m], [0, m], color=tone(GREY, .6), lw=HAIR, zorder=1)
    for (a, b), c in C.items():
        ax.scatter(a, b, s=1.5 + 2.0 * c ** 0.8, color=COB, lw=0, zorder=3)
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xticks(range(0, m + 1, 2)); ax.set_yticks(range(0, m + 1, 2))
    ax.set_xlabel("levels in the answer"); ax.set_ylabel("loops to rest")
    from matplotlib.lines import Line2D
    ks = [1, 10, 50]
    assert max(C.values()) >= 50
    hs = [Line2D([], [], ls="", marker="o", ms=(1.5 + 2.0 * k ** 0.8) ** 0.5, color=COB, mew=0) for k in ks]
    ax.legend(hs, [f"{k}" for k in ks[:-1]] + [f"{ks[-1]} graphs"], loc="lower right", frameon=False, fontsize=LABEL,
              ncol=3, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)
    ax.set_aspect("auto")
    return ax, len(lv)


def build():
    panel_a(0.0, 118.0)
    panel_b(124.0, 59.0)
    panel_c(0.0, 118.0)
    panel_d(124.0, 59.0)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig2")
