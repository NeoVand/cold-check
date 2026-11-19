"""Fig. 5, "Fitted in milliseconds, exact far beyond training". 183 x H mm. Cobalt: ours (open: CLRS-30 tasks given a
piece by hand); rose: CRAM; pale rose: the looped models.

a  The time to fit one rule, one row per benchmark (log axis: fit times run from 15 ms to 9 hours). Within a row, marks
   that would overlap are stacked; only their horizontal position is data.
b  The sizes at which one fitted rule is exact (log axis): a line from our largest training size (tick) to the largest
   held-out size at which every test was exact (dot); CRAM fits and tests at one size.
Numbers: costdata.py (CRAM's tables parsed from code/ext/cram.txt; ours from code/bench and code/clrs) and fig4.py's
reading of paper/gen/make_clrs_table.py for the tasks given a piece.
    ./run.sh fig5.py
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
sys.path.insert(0, ".")
import kit
from kit import INK, SOFT, GREY, COB, ROSE, GUIDE, HAIR, LABEL, tone, text, head, W

import costdata as _cd  # noqa: E402
clrs, ours, comp, cram_size, LOOPED = _cd.clrs, _cd.ours, _cd.comp, _cd.cram_size, _cd.LOOPED
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("fig4mod", "fig4.py")
_f4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_f4)
INFO = _f4.INFO
assert len(clrs) == 15 and all(INFO[t]["tier"] == 1 for t in clrs)
CEIL = {t: v for t, v in INFO.items() if v["tier"] > 1}
TRAIN_CLRS = max(_cd.SIZES)

ROWS = [("sudoku", "Sudoku-Extreme", 1), ("unique", "Maze-Unique", 1), ("hard", "Maze-Hard", 1),
        ("queens", "N-Queens", 1), ("clrs1", "CLRS-30, comparisons and counts", 2.6), ("clrs2", "CLRS-30, a piece supplied", 2.6)]
RH = 6.4                                         # row pitch (mm) for a weight-1 row
Y0 = 13.0
H = Y0 + sum(w for *_, w in ROWS) * RH + 14.0
fig = kit.figure(H)
TOP = {}
_y = Y0
for k, _, wgt in ROWS:
    TOP[k] = (_y, _y + wgt * RH)
    _y += wgt * RH
XA0, XA1 = 32.0, 106.0                           # panel a axes, mm
XB0, XB1 = 116.0, W - 1.0                        # panel b axes, mm
LA = (np.log10(0.008), np.log10(1.3e5))
LB = (np.log10(10), np.log10(2.6e4))
DOT = 1.05                                       # marker diameter, mm (for the swarm)


def xa(v):
    return XA0 + (np.log10(v) - LA[0]) / (LA[1] - LA[0]) * (XA1 - XA0)


def xb(v):
    return XB0 + (np.log10(v) - LB[0]) / (LB[1] - LB[0]) * (XB1 - XB0)


def swarm(xs, yc, step=DOT * 1.08):
    """Exact x; marks that would overlap move up or down in the row (y only)."""
    out = [None] * len(xs)
    placed = []
    for i in np.argsort(xs, kind="stable"):
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            y = yc + k * step / 2
            if all((xs[i] - a) ** 2 + (y - b) ** 2 >= step ** 2 for a, b in placed):
                break
        placed.append((xs[i], y)); out[i] = y
    return out


def dot(cv, x, y, col, fill=True, shape="o", d=DOT):
    from matplotlib.patches import Circle, RegularPolygon, Rectangle
    if shape == "o":
        p = Circle((x, y), d / 2)
    else:
        p = Rectangle((x - d / 2, y - d / 2), d, d)
    p.set_facecolor(col if fill else "white"); p.set_edgecolor(col); p.set_linewidth(0.0 if fill else 0.55)
    p.set_zorder(5)
    cv.add_patch(p)


def axis(cv, x0, x1, ticks, f, labels, title):
    yb = TOP["clrs2"][1] + 1.0
    seg(cv, (x0, yb), (x1, yb), INK, HAIR)
    for t, l in zip(ticks, labels):
        seg(cv, (f(t), yb), (f(t), yb + 0.9), INK, HAIR)
        text(fig, f(t), yb + 3.9, l, ha="center")
        seg(cv, (f(t), Y0 - 1.0), (f(t), yb), GUIDE, 0.4, z=0.5)
    text(fig, (x0 + x1) / 2, yb + 7.8, title, ha="center")


def build():
    from draw import seg as _seg
    global seg
    seg = _seg
    cv = kit.canvas(fig, 0, 0, W, H)
    head(fig, 0, 4.4, "a", "Time to fit one rule")
    head(fig, XB0 - 3.5, 4.4, "b", "Sizes at which one rule is exact")
    for k, name, wgt in ROWS:
        y0, y1 = TOP[k]
        yc = (y0 + y1) / 2
        text(fig, XA0 - 2.0, yc + 0.9, name if wgt == 1 else name.split(", ")[0], ha="right")
        if wgt > 1:
            text(fig, XA0 - 2.0, yc + 3.9, name.split(", ")[1], ha="right", color=SOFT)
        if k != "sudoku":
            seg(cv, (2.0, y0), (W, y0), GUIDE, 0.4, z=0.4)
    axis(cv, XA0, XA1, [0.01, 1, 60, 3600, 86400], xa, ["10 ms", "1 s", "1 min", "1 h", "1 day"], "fit time, one core (ours) or one H200 GPU")
    axis(cv, XB0, XB1, [16, 64, 256, 1024, 4096, 16384], xb, ["16", "64", "256", "1,024", "4,096", "16,384"], "nodes or cells")
    # CRAM's benchmarks
    for k in ("sudoku", "unique", "hard", "queens"):
        y0, y1 = TOP[k]; yc = (y0 + y1) / 2
        o, c = ours[k], comp[k]
        lx = [xa(v) for v in c["looped"].values()]
        for x, y in zip(lx, swarm(lx, yc)):
            dot(cv, x, y, tone(ROSE, 0.42))
        if c["cram"] is not None:
            dot(cv, xa(c["cram"]), yc, ROSE)
        dot(cv, xa(o["fit_s"]), yc, COB)
        # b: training size (tick) to the largest size exact
        ya = yc - 0.9
        seg(cv, (xb(o["fit_size"]), ya), (xb(max(o["exact"])), ya), COB, 0.8)
        seg(cv, (xb(o["fit_size"]), ya - 0.9), (xb(o["fit_size"]), ya + 0.9), INK, 0.6)
        dot(cv, xb(max(o["exact"])), ya, COB)
        for sz in cram_size[k]:
            dot(cv, xb(sz), yc + 1.0, ROSE)
    # CLRS-30
    for k, items, filled in (("clrs1", [(v["fit_s"], 16 if v["fit_size"] == 16 else v["fit_size"], v["exact_to"]) for v in clrs.values()], True),
                             ("clrs2", [(v["fit_ms"] / 1e3, TRAIN_CLRS, v["exact_to"]) for v in CEIL.values()], False)):
        y0, y1 = TOP[k]; yc = (y0 + y1) / 2
        xs = [xa(f) for f, _, _ in items]
        for x, y in zip(xs, swarm(xs, yc)):
            dot(cv, x, y, COB, fill=filled)
        rows_b = sorted(items, key=lambda r: (-r[2], r[1]))
        n = len(rows_b)
        for i, (_, tr, ex) in enumerate(rows_b):
            yy = y0 + 1.2 + i * (y1 - y0 - 2.4) / (n - 1)
            if ex is None or ex <= tr:                     # segments intersect: a fixed size of four points
                continue
            seg(cv, (xb(tr), yy), (xb(ex), yy), COB if filled else tone(COB, 0.45), 0.45)
            dot(cv, xb(ex), yy, COB, fill=filled, d=0.8)
        trs = sorted({tr for _, tr, _ in items})
        for tr in trs:
            seg(cv, (xb(tr), y0 + 0.6), (xb(tr), y1 - 0.6), INK, 0.6)
    # legend
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker="o", ms=3.0, color=COB, mew=0),
          Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=COB, mew=0.6),
          Line2D([], [], ls="", marker="o", ms=3.0, color=ROSE, mew=0),
          Line2D([], [], ls="", marker="o", ms=3.0, color=tone(ROSE, 0.42), mew=0),
          Line2D([], [], ls="", marker="|", ms=4.0, color=INK, mew=0.7)]
    fig.legend(hs, ["ours", "ours, a piece supplied", "CRAM", "looped models", "our largest training size"], ncol=5,
               loc="upper left", bbox_to_anchor=(XA0 / W, 1 - 6.6 / H), frameon=False, fontsize=LABEL,
               handletextpad=0.3, columnspacing=1.5, borderaxespad=0)
    return fig


if __name__ == "__main__":
    build()
    for p in kit.check(fig):
        print("CHECK", p)
    kit.save(fig, "fig5")
    print({k: (round(v["fit_s"] * 1e3, 1), max(v["exact"])) for k, v in ours.items()})
    print({k: (v["cram"], cram_size[k]) for k, v in comp.items()})
