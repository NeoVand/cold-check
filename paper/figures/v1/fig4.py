"""Fig. 4, "CLRS-30, fitted at the answer": all thirty tasks, grouped by what is supplied by hand. 183 x H mm.
Cobalt: our fitted checks (filled: comparisons and counts; open: a piece supplied); pale cobalt: our looped blocks
of the earlier paper; rose: FloydNet; navy: the answers in the glyphs; a pale band behind the tasks given a piece.

a  Thirty glyphs in three groups (the paper's Table 3 groups, read from paper/gen/make_clrs_table.py, which is exec'd up
   to its table loop and never run whole): comparisons and counts (15; task scheduling's deadlines in the units
   of positions and LCS's one-cell index shift marked "+ units", "+ shift"), one arithmetic primitive supplied (9), a
   piece of structure supplied (6). Each glyph (tiles.py) is a small input of the library's generator with its true
   answer. Under each: the task, its fit time on one laptop core and the largest held-out size at which every input
   was exact (results.py: code/clrs/final/; segments intersect has a fixed size of four points), and what is supplied.
b  % right on the official 64-node test, per task: our fitted check (code/clrs/final/summary.json, 100 on all 30,
   asserted), our looped blocks (better of the general and cold blocks) and FloydNet's reported score (both
   from arXiv 2611.11937 Supp. Table 4, as make_clrs_table.py's P). Left: comparisons and counts; right: a piece
   supplied.
    ./run.sh fig4.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, TITLE, LETTER, tone, text, head, W

# ------------------------------------------------------------------------------------------------- the numbers
_MCT = kit.ROOT / "paper" / "gen" / "make_clrs_table.py"
_src = _MCT.read_text()
_ns = {"__file__": str(_MCT)}
exec(_src[:_src.index("rows, tier_prev")], _ns)
T, P = _ns["T"], _ns["P"]
assert len(T) == 30
import results as R  # noqa: E402
INFO = {}
for t, (nm, tier, sup) in T.items():
    r = R.clrs(t)
    g, c, fn = P[t]
    INFO[t] = dict(name=nm.replace("--", "–"), tier=tier, sup=sup, fit_ms=r["fit_ms"], exact_to=r["exact_to"],
                   test64=r["test64"], looped=max(v for v in (g, c) if v is not None), floydnet=fn)
assert [sum(i["tier"] == k for i in INFO.values()) for k in (1, 2, 3)] == [15, 9, 6]

# ------------------------------------------------------------------------------------------------- the glyphs
import tiles as F5  # noqa: E402
WC, HC = F5.WC, F5.HC


NOTE1 = {"task_scheduling": "+ units", "lcs_length": "+ shift"}   # make_clrs_table: deadline / n; index shift
assert sorted(t for t, v in INFO.items() if v["tier"] == 1 and v["sup"] != "--") == sorted(NOTE1)
SUPPLIED = {"sum": "+ sum", "orientation": "+ orientation", "products, sums": "+ products, sums",
            "sums": "+ sums", "prefix sums": "+ prefix sums", "deletion closure": "+ closures",
            "``open''": "+ open bit", "``open'', finish order": "+ open bit", "two passes": "+ two passes"}
SHORT = {"Breadth-first search": "breadth-first\nsearch", "Activity selection": "activity\nselection",
         "Task scheduling": "task\nscheduling", "Naive string matcher": "naive string\nmatcher",
         "DAG shortest paths": "DAG shortest\npaths", "Matrix chain order": "matrix chain\norder",
         "Segments intersect": "segments\nintersect", "Maximum subarray": "maximum\nsubarray",
         "Articulation points": "articulation\npoints", "Depth-first search": "depth-first\nsearch",
         "Topological sort": "topological\nsort", "Strong components": "strong\ncomponents",
         "Insertion sort": "insertion\nsort"}
ORDER = {1: ["insertion_sort", "bubble_sort", "heapsort", "quicksort", "minimum", "binary_search", "quickselect",
             "activity_selector", "task_scheduling", "bfs", "mst_kruskal", "mst_prim", "lcs_length",
             "naive_string_matcher", "kmp_matcher"],
         2: ["bellman_ford", "dijkstra", "dag_shortest_paths", "floyd_warshall", "matrix_chain_order", "optimal_bst",
             "segments_intersect", "graham_scan", "jarvis_march"],
         3: ["find_maximum_subarray_kadane", "articulation_points", "bridges", "dfs", "topological_sort",
             "strongly_connected_components"]}
assert sorted(sum(ORDER.values(), [])) == sorted(T)
NCOL = {1: 5, 2: 3, 3: 2}
GAPX = 3.0
PX = (W - 1.4 - 2 * GAPX) / 10                # slot pitch
SC = (PX - 1.6) / WC                          # glyph scale
TW, TH = WC * SC, HC * SC
PY = 26.0                                     # row pitch
YA = 12.0                                     # first tile row top


def label(name):
    s = name[0].lower() + name[1:] if name.split()[0] not in ("Bellman–Ford", "Dijkstra", "Kruskal", "Prim", "Floyd–Warshall",
                                                                "Graham", "Jarvis", "KMP", "LCS", "DAG", "Optimal") else name
    s = SHORT.get(name, s)
    return s if s[0].isupper() or name.split()[0] in ("KMP", "LCS", "DAG") else s[0].lower() + s[1:]


def fmt_ms(ms):
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def tile(fig, t, X, Y):
    """The glyph of task t, scaled by SC, top-left at page (X, Y)."""
    m = 1.6
    ax = kit.axes_mm(fig, X - m * SC, Y - m * SC, (WC + 2 * m) * SC, (HC + 2 * m) * SC)
    ax.set_axis_off(); ax.set_facecolor("none")
    ax.set_xlim(-m, WC + m); ax.set_ylim(HC + m, -m)
    rings = {}
    slots = TILE_SLOTS.get(t)
    if slots is not None:
        rings[t] = F5.Ring(0, 0, slots, r=F5.NODE_R24 if t in F5.SPARSE_T else F5.NODE_R)
    F5.build_draw(fig)[t](ax, 0, 0, rings)
    return ax


def block_x(tier):
    x = 0.7
    for k in (1, 2, 3):
        if k == tier:
            return x
        x += NCOL[k] * PX + GAPX


def panel_a(fig):
    head(fig, 0, 4.4, "a", "Thirty CLRS-30 tasks, fitted at the answer")
    hdr = {1: "comparisons and counts", 2: "one arithmetic primitive supplied",
           3: "a piece of structure supplied"}
    bg = kit.canvas(fig, 0, 0, W, YA + 3 * PY)
    for tier in (1, 2, 3):
        x0 = block_x(tier)
        w = NCOL[tier] * PX
        rows = int(np.ceil(len(ORDER[tier]) / NCOL[tier]))
        if tier > 1:                                              # the ceilings sit on a pale band
            kit.text(fig, x0 + 1.0, YA - 2.6, hdr[tier], color=SOFT)
            from matplotlib.patches import FancyBboxPatch
            bg.add_patch(FancyBboxPatch((x0 + 0.3, YA - 1.0), w - 0.6, rows * PY + 0.4, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=tone(INK, 0.03), ec="none", zorder=0))
        else:
            kit.text(fig, x0 + 1.0, YA - 2.6, hdr[tier], color=SOFT)
        for i, t in enumerate(ORDER[tier]):
            r, c = divmod(i, NCOL[tier])
            X = x0 + c * PX + (PX - TW) / 2
            Y = YA + r * PY + 1.2
            tile(fig, t, X, Y)
            cx = X + TW / 2
            inf = INFO[t]
            y = Y + TH + 3.5
            for line in label(inf["name"]).split("\n"):
                text(fig, cx, y, line, ha="center")
                y += 3.0
            if "\n" not in label(inf["name"]):
                y += 3.0
            y += 0.4
            to = "fixed size" if t == "segments_intersect" else f"{inf['exact_to']:,}"
            text(fig, cx, y, f"{fmt_ms(inf['fit_ms'])} · {to}", ha="center", color=COB)
            if tier > 1:
                text(fig, cx, y + 3.0, SUPPLIED[inf["sup"]], ha="center", color=SOFT)
            elif t in NOTE1:                                       # group one, with a small hand-set adjustment
                text(fig, cx, y + 3.0, NOTE1[t], ha="center", color=SOFT)


# ------------------------------------------------------------------------------------------------- panel b
def panel_b(fig, y0):
    head(fig, 0, y0, "b", "Official test, % right")
    from matplotlib.lines import Line2D
    cols = [(1, 0.0), (0, W / 2 + 1.5)]
    RP = 3.05
    for ci, (which, x0) in enumerate(((1, 0.0), (2, W / 2 + 1.5))):
        tasks = ORDER[1] if which == 1 else ORDER[2] + ORDER[3]
        ax = kit.axes_mm(fig, x0 + 30, y0 + 9.0, W / 2 - 34, RP * 15 + 1.0)
        kit.clean(ax)
        ax.set_xlim(78, 100.6); ax.set_ylim(15 - 0.4, -0.6)
        ax.set_yticks(range(15)); ax.set_yticklabels([label(INFO[t]["name"]).replace("\n", " ") for t in tasks]); ax.tick_params(axis="y", length=0, pad=2.5)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([80, 90, 100]); ax.set_xlabel("% right")
        for i, t in enumerate(tasks):
            inf = INFO[t]
            ceil = inf["tier"] > 1
            ax.plot([78, 100], [i, i], color=GUIDE, lw=0.4, zorder=1)
            ax.plot(inf["floydnet"], i - 0.18, "o", ms=2.6, color=ROSE, mew=0, zorder=3, clip_on=False)
            ax.plot(inf["looped"], i + 0.18, "o", ms=3.2, color=tone(COB, .38), mew=0, zorder=3, clip_on=False)
            ax.plot(inf["test64"], i + 0.18, "o", ms=2.6, mfc="white" if ceil else COB, mec=COB, mew=0.7 if ceil else 0, zorder=4,
                    clip_on=False)
        if ci == 0:
            hs = [Line2D([], [], ls="", marker="o", ms=2.6, color=COB, mew=0),
                  Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=COB, mew=0.7),
                  Line2D([], [], ls="", marker="o", ms=3.2, color=tone(COB, .38), mew=0),
                  Line2D([], [], ls="", marker="o", ms=2.6, color=ROSE, mew=0)]
            fig.legend(hs, ["fitted check", "fitted check, piece supplied", "our looped blocks", "FloydNet"], ncol=4,
                       loc="upper left", bbox_to_anchor=(4.6 / W, 1 - (y0 + 1.4) / fig._h), frameon=False,
                       fontsize=LABEL, handletextpad=0.3, columnspacing=1.6, borderaxespad=0)
    return y0 + 9.0 + RP * 15 + 9


H = 4.4 + YA + 3 * PY + 66
fig = kit.figure(H)
slots, _ = F5.rings_for()
TILE_SLOTS = slots


def build():
    panel_a(fig)
    panel_b(fig, YA + 3 * PY + 4.0)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig4")
