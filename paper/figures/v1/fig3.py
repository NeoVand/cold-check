"""Fig. 3, "One fitted rule on CRAM's benchmarks". 183 x H mm. Cobalt: set by our fitted check run cold (and our
answers and numbers); rose: set by the search (warm), and CRAM's released models and published numbers; pale rose: the
looped models; navy: givens and walls; grey: CRAM's hand-set energy.

a  Sudoku at four sizes, one rule (the 9 x 9 train-s0 fit, unchanged): digits given (navy), set by the cold stage before
   any search (cobalt, tinted cell), set during the search (rose). data/puzzle_boards.json (from the bench
   final engine; see data/puzzle_boards.py). 9 x 9 is a Sudoku-Extreme test puzzle; the others are generated.
d  N-Queens 100 x 100 from 12 given queens (7/8 removed): given queens navy, queens placed by the search rose (the cold
   stage forces none here). Same file.
b  A 29 x 29 tree maze whose dead ends reach the board's edge (h = 15, maze 0 of the 200 of the recorded run, no wall
   ring): the path (cobalt; ours, exact), and the cells CRAM's released Maze-Unique model (seed 0) also leaves on (rose).
   data/released_boards.json (re-run of the recorded case; counts asserted equal to the record).
c  A room-and-corridor maze (Maze-Hard style, room 24, the symmetry where CRAM's released Maze-Hard model, seed 0, first
   leaves the shortest path): ours (cobalt) = the label's shortest path, 49 steps; CRAM's returned path (rose band), 51.
e  Operations per puzzle against solved, on CRAM's test sets: CRAM's Table 2 (Sudoku-Extreme, Maze-Unique) and Table 8
   (Maze-Hard), parsed from code/ext/cram.txt; ours from code/bench/final/summary.json (Sudoku: mean of the three
   fits; mazes) (CRAM's convention, comparisons free; open mark: every comparison counted too).
    ./run.sh fig3.py
"""
import json
import re
import sys
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath
from matplotlib.lines import Line2D
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, LINE, tone, text, head, W
from draw import seg

PB = json.load(open(kit.DATA / "puzzle_boards.json"))
RB = json.load(open(kit.DATA / "released_boards.json"))
import results as R  # noqa: E402
FIN = R.PUZ                                       # code/bench/final/summary.json
TXT = (kit.CODE / "ext" / "cram.txt").read_text().splitlines()
_FP = FontProperties(family=kit.plt.rcParams["font.sans-serif"][0])

H = 196.0
fig = kit.figure(H)
WALL = tone(INK, 0.30)


def glyph(ax, x, y, s, h, color):
    """A digit string as glyph outlines, centred at (x, y) (page mm, y down), cap height h mm (a mark of the board)."""
    tp = TextPath((0, 0), s, size=1.0, prop=_FP)
    v = tp.vertices.copy()
    cap = float(np.ptp(TextPath((0, 0), "8", size=1.0, prop=_FP).vertices[:, 1]))
    v *= h / cap
    v[:, 1] = -v[:, 1]
    v[:, 0] += x - (v[:, 0].min() + v[:, 0].max()) / 2
    v[:, 1] += y + h / 2
    ax.add_patch(PathPatch(MPath(v, tp.codes), fc=color, ec="none", lw=0, zorder=8))


def board(ax, X, Y, side, cell, rec):
    n = side
    b = int(round(n ** 0.5))
    sol = np.array(rec["solution"]).reshape(n, n); kind = np.array(rec["kind"]).reshape(n, n)
    for i in range(n):
        for j in range(n):
            k = kind[i, j]
            if k == 1:                                          # set cold: the few cells worth finding at a glance
                ax.add_patch(Rectangle((X + j * cell, Y + i * cell), cell, cell, fc=tone(COB, 0.16), ec="none", zorder=1))
            col = INK if k == 0 else COB if k == 1 else ROSE
            s = str(sol[i, j])
            glyph(ax, X + (j + 0.5) * cell, Y + (i + 0.5) * cell, s, cell * (0.5 if n < 16 else 0.42), col)
    for i in range(n + 1):
        lw = 0.6 if i % b == 0 else 0.2
        c = INK if i % b == 0 else tone(INK, 0.35)
        seg(ax, (X, Y + i * cell), (X + n * cell, Y + i * cell), c, lw, z=3, cap="projecting" if lw > 0.3 else "butt")
        seg(ax, (X + i * cell, Y), (X + i * cell, Y + n * cell), c, lw, z=3, cap="projecting" if lw > 0.3 else "butt")


def panel_a(y0):
    head(fig, 0, y0, "a", "Sudoku at four sizes, one fitted rule")
    text(fig, 4.6, y0 + 4.2, "navy, given; cobalt, set cold; rose, set by the search", color=SOFT)
    big = 25 * 2.2
    ax = kit.canvas(fig, 0, y0 + 6, W, big + 10)
    yb = y0 + 9.0
    sizes = ((4, 4.4), (9, 3.15), (16, 2.6), (25, 2.2))
    gap = (W - 1.2 - sum(n * c for n, c in sizes)) / 3
    x = 0.6
    out = {}
    for side, cell in sizes:
        rec = PB["sudoku"][str(side)]
        top = yb + (big - side * cell)                        # boards share a baseline
        board(ax, x, top, side, cell, rec)
        kind = np.array(rec["kind"])
        out[side] = [int((kind == k).sum()) for k in (0, 1, 2)]
        text(fig, x, yb + big + 4.2, f"{side} × {side}", color=INK)
        x += side * cell + gap
    return out


def panel_b(x0, y0, w):
    head(fig, x0, y0, "d", "N-Queens, 100 × 100")
    q = PB["queens"]
    N = q["N"]
    ax = kit.canvas(fig, x0, y0 + 2, w, 58)
    s = 50.0 / N
    X, Y = x0 + 0.6, y0 + 7.0
    ax.add_patch(Rectangle((X, Y), N * s, N * s, fc="white", ec=INK, lw=0.4, zorder=1))
    for r in range(0, N + 1, 10):
        seg(ax, (X, Y + r * s), (X + N * s, Y + r * s), GUIDE, 0.25, z=0.5)
        seg(ax, (X + r * s, Y), (X + r * s, Y + N * s), GUIDE, 0.25, z=0.5)
    given = set(q["given_rows"])
    for r, c in enumerate(q["cols"]):
        ax.add_patch(Circle((X + (c + 0.5) * s, Y + (r + 0.5) * s), 0.45, fc=INK if r in given else ROSE, ec="none", zorder=3))
    text(fig, X, Y + 50 + 4.2, f"{len(given)} given, {N - len(given)} placed by the search", color=SOFT)
    assert len(q["cold_rows"]) == 0
    return len(given)


def maze_cells(ax, X, Y, g, s):
    """Walls as filled cells (navy tint); open cells white."""
    H_, W_ = g.shape
    ax.add_patch(Rectangle((X, Y), W_ * s, H_ * s, fc="white", ec="none", zorder=0.5))
    for i in range(H_):
        for j in range(W_):
            if not g[i, j]:
                ax.add_patch(Rectangle((X + j * s, Y + i * s), s, s, fc=WALL, ec=WALL, lw=0.05, zorder=1))


def path_xy(X, Y, cells, s):
    return np.array([[X + (c + 0.5) * s, Y + (r + 0.5) * s] for r, c in cells])


def order_path(onmask, S, G):
    """The on-cells of a simple path from S to G, in order."""
    p, prev = [tuple(S)], None
    while p[-1] != tuple(G):
        r, c = p[-1]
        nxt = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
               if 0 <= r + dr < onmask.shape[0] and 0 <= c + dc < onmask.shape[1] and onmask[r + dr, c + dc]
               and (r + dr, c + dc) != prev]
        prev = p[-1]
        p.append(nxt[0])
    return p


def endpoints(ax, P0, P1, s):
    for p in (P0, P1):
        ax.add_patch(Rectangle((p[0] - s / 2, p[1] - s / 2), s, s, fc=INK, ec="none", zorder=7))


def panel_c(x0, y0):
    head(fig, x0, y0, "b", "Dead ends at the edge")
    m = RB["edge"]["0"]["mazes"][0]
    g = np.array(m["grid"], bool); lab = np.array(m["label"], bool); pred = np.array(m["pred"], bool)
    assert not (lab & ~pred).any()                              # CRAM leaves every path cell on ...
    extra = pred & ~lab                                         # ... and these too
    assert int(extra.sum()) == m["n_diff"] == 159 and not m["solved"]
    s = 50.0 / 29
    X, Y = x0 + 0.6, y0 + 7.0
    ax = kit.canvas(fig, x0, y0 + 2, 62, 58)
    maze_cells(ax, X, Y, g, s)
    for r, c in np.argwhere(extra):
        ax.add_patch(Rectangle((X + c * s, Y + r * s), s, s, fc=tone(ROSE, 0.38), ec="none", zorder=2))
    P = path_xy(X, Y, order_path(lab, m["S"], m["G"]), s)
    ax.plot(*P.T, color=COB, lw=1.1, zorder=5, solid_joinstyle="round", solid_capstyle="round")
    endpoints(ax, P[0], P[-1], s * 0.62)
    text(fig, X, Y + g.shape[0] * s + 4.0, f"CRAM leaves {int(extra.sum())} cells on", color=ROSE)
    return int(extra.sum())


def panel_d(x0, y0):
    head(fig, x0, y0, "c", "A room on the way")
    r = RB["room"][0]
    assert (r["n"], r["extra"], r["sym"]) == (24, 2, 1) and r["valid"] and r["length"] == r["shortest"] + 2 == 51
    g = np.array(r["grid"], bool); b = np.array(r["board"], bool)
    s = 50.0 / 29
    X, Y = x0 + 0.6, y0 + 7.0
    ax = kit.canvas(fig, x0, y0 + 2, 62, 58)
    maze_cells(ax, X, Y, g, s)
    cram = path_xy(X, Y, order_path(b, r["S"], r["G"]), s)
    ours = path_xy(X, Y, [tuple(c) for c in r["label"]], s)
    ax.plot(*cram.T, color=tone(ROSE, 0.45), lw=3.6, zorder=4, solid_joinstyle="round", solid_capstyle="round")
    ax.plot(*ours.T, color=COB, lw=1.1, zorder=5, solid_joinstyle="round", solid_capstyle="round")
    endpoints(ax, ours[0], ours[-1], s * 0.62)
    text(fig, X, Y + g.shape[0] * s + 4.0, f"ours {len(ours) - 1} steps, CRAM {len(cram) - 1}", color=SOFT)
    return len(ours) - 1, len(cram) - 1


# ------------------------------------------------------------------------------------------------- e: the numbers
def table2():
    i0 = next(i for i, l in enumerate(TXT) if l.startswith("Table 2: CRAM trains faster"))
    rows = {}
    pat = re.compile(r"^\s+(\S.*?)\s{2,}(\d+\.\d+)\s*(?:±\s*\d+\.\d+|†)?\s+(\S+)\s+(\d\.\d+e\d+)\s+(\S+(?: \S)?)\s+"
                     r"(\d+\.\d+)\s*(?:±\s*\d+\.\d+|†)?\s+(\S+)\s+(\d\.\d+e\d+)\s+(\S+(?: \S)?)\s*$")
    for l in TXT[i0:i0 + 30]:
        m = pat.match(l)
        if m:
            nm = re.sub(r"\s*\[\d+\]|[§‡†]", "", m.group(1)).strip()
            rows[nm] = dict(sx=(float(m.group(2)), float(m.group(4))), mu=(float(m.group(6)), float(m.group(8))))
    return rows


def table8():
    i0 = next(i for i, l in enumerate(TXT) if l.startswith("Table 8: CRAM+z"))
    rows = {}
    for l in TXT[i0:i0 + 14]:
        m = re.match(r"^\s+(\S+?)†?\s*(?:\[\d+\])?\s+(\d+\.\d+)(?:\s*±\s*\d+\.\d+)?\s+\S+\s+(\d\.\d+e\d+)\s*$", l)
        if m:
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return rows


T2, T8 = table2(), table8()
assert sorted(T2) == sorted(["HRM", "TRM", "GRAM", "EqR", "FRM", "TRMP", "EBR", "CRAM", "(hand-set)"]), sorted(T2)
assert sorted(T8) == sorted(["HRM", "TRM", "FPRM", "NCA", "EBR", "CRAM+z"]), sorted(T8)
_sx, _mu, _mh = FIN["sudoku_extreme"], FIN["maze_unique"]["test"], FIN["maze_hard"]["test"]
OURS = dict(sx=(np.mean([f["ops_mean"] for f in _sx]), np.mean([f["ops_plus_cmp_mean"] for f in _sx]),
                min(f["solved"] for f in _sx) / 412786 * 100),
            mu=(_mu["ops_mean"], _mu["ops_plus_cmp_mean"], 100.0 * _mu["exact"] / _mu["n"]),
            mh=(_mh["ops_mean"], _mh["ops_plus_cmp_mean"], 100.0 * _mh["exact"] / _mh["n"]))
assert OURS["sx"][2] == 100.0 and OURS["mu"][2] == 100.0 and OURS["mh"][2] == 100.0
assert abs(T2["CRAM"]["sx"][1] - 2.02e9) < 1


def panel_e(y0):
    head(fig, 0, y0, "e", "Operations per puzzle and puzzles solved")
    sets = [("Sudoku-Extreme", {k: v["sx"] for k, v in T2.items()}, "CRAM", OURS["sx"]),
            ("Maze-Unique", {k: v["mu"] for k, v in T2.items()}, "CRAM", OURS["mu"]),
            ("Maze-Hard", {k: (v[0], v[1]) for k, v in T8.items()}, "CRAM+z", OURS["mh"])]
    w = (W - 16 - 2 * 9) / 3
    for i, (name, rows, cram, ours) in enumerate(sets):
        x = 12 + i * (w + 9)
        ax = kit.axes_mm(fig, x, y0 + 12.5, w, 26)
        kit.clean(ax)
        ax.set_xscale("log"); ax.set_xlim(1e3, 1e15); ax.set_ylim(30, 103)
        ax.set_xticks([1e3, 1e6, 1e9, 1e12, 1e15])
        ax.set_xticklabels(["$10^{3}$", "$10^{6}$", "$10^{9}$", "$10^{12}$", "$10^{15}$"])
        ax.set_yticks([40, 70, 100])
        ax.minorticks_off()
        if i == 0:
            ax.set_ylabel("solved (%)")
        ax.set_xlabel("operations per puzzle")
        kit.text(fig, x, y0 + 9.5, name, color=INK)
        for nm, (sv, fl) in rows.items():
            if nm == cram:
                ax.plot(fl, sv, "o", ms=3.4, color=ROSE, mew=0, zorder=4, clip_on=False)
            elif nm == "(hand-set)":
                ax.plot(fl, sv, "o", ms=3.0, mfc="white", mec=GREY, mew=0.7, zorder=3, clip_on=False)
            else:
                ax.plot(fl, sv, "o", ms=3.0, color=tone(ROSE, 0.4), mew=0, zorder=3, clip_on=False)
        ax.plot([ours[0], ours[1]], [ours[2], ours[2]], color=tone(COB, 0.5), lw=0.6, zorder=4)
        ax.plot(ours[1], ours[2], "o", ms=3.0, mfc="white", mec=COB, mew=0.7, zorder=5, clip_on=False)
        ax.plot(ours[0], ours[2], "o", ms=3.4, color=COB, mew=0, zorder=6, clip_on=False)
    hs = [Line2D([], [], ls="", marker="o", ms=3.4, color=COB, mew=0),
          Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=COB, mew=0.7),
          Line2D([], [], ls="", marker="o", ms=3.4, color=ROSE, mew=0),
          Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=GREY, mew=0.7),
          Line2D([], [], ls="", marker="o", ms=3.0, color=tone(ROSE, 0.4), mew=0)]
    fig.legend(hs, ["ours", "ours, every comparison counted", "CRAM", "CRAM, hand-set", "looped models"], ncol=5,
               loc="upper left", bbox_to_anchor=(4.6 / W, 1 - (y0 + 1.6) / H), frameon=False, fontsize=LABEL,
               handletextpad=0.3, columnspacing=1.4, borderaxespad=0)


def build():
    panel_a(4.0)
    panel_c(0.0, 79.0)
    panel_d(65.0, 79.0)
    panel_b(130.0, 79.0, 53.0)
    panel_e(147.0)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig3")
