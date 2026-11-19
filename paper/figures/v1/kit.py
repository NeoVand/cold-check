"""House style for the figures of "When the Check Is the Algorithm": the style of our earlier paper's figures.

    page        183 mm wide, laid out in millimetres from the top-left corner
    type        panel letter 9 pt bold, panel title 8 pt, labels / ticks / grey subtitles 6.5 pt, nothing else
    math        every formula, variable and symbol as $...$ mathtext in Computer Modern; words in Helvetica Neue
    ink         navy #1B2A4A hairlines 0.5 pt; cobalt #2557A7 ours, run cold; rose #D6336C the comparison; grey; pale tints
    plots       lines 1.0 pt, markers 2.2 pt

and the geometry checks (texts at least 0.8 mm apart and 0.8 mm from any mark they do not name, nothing outside the
page, nothing clipped by its axes). house/layout.py supplies the fonts (Helvetica Neue faces extracted from the system
font into house/.fonts, never committed) and millimetre axes.

Data: every result file is read under CODE, the repository's code/ tree (or $COLDCHECK_DATA if set).
"""
import os
import pathlib
import sys

WIT = pathlib.Path(__file__).resolve().parent / "house"                        # layout and fonts
sys.path.insert(0, str(WIT))
import layout  # noqa: E402  (fonts, house mplstyle, axes_mm)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb, to_hex  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection, PathCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"                                     # staged renders (the paper builds from figures/, never written)
ZOOM = OUT / "zoom"
DATA = HERE / "data"                                   # this folder's own data (extracted boards, glyph inputs)
ROOT = HERE.parents[2]                                 # the repository
CODE = pathlib.Path(os.environ.get("COLDCHECK_DATA", ROOT / "code"))

# ---------------------------------------------------------------------------------------------------- house style
W = 183.0
MM = 1 / 25.4
LETTER, TITLE, LABEL = 9.0, 8.0, 6.5                   # the three type sizes
SIZES = (LETTER, TITLE, LABEL)
INK = "#1B2A4A"                                        # navy hairlines and text
SOFT = "#4C5B78"                                       # grey subtitles and notes (the Witnesses ink_soft)
GREY = "#8A94A6"                                       # other people's models, the laws
ROSE = "#D6336C"                                       # warm
COB = "#2557A7"                                        # cold
GUIDE = "#E2E7EF"                                      # row guides
HAIR = 0.5                                             # structure hairline, pt
LINE = 1.0                                             # plotted lines, pt
MARK = 2.2                                             # marker diameter, pt
GAP = 0.8                                              # the least clearance between a label and anything else, mm

plt.rcParams.update({
    "font.size": LABEL, "axes.labelsize": LABEL, "xtick.labelsize": LABEL, "ytick.labelsize": LABEL,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "mathtext.fontset": "cm", "mathtext.fallback": "cm",   # every $...$ in Computer Modern, as in a LaTeX paper
    "axes.unicode_minus": True, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": HAIR, "xtick.major.width": HAIR,
    "ytick.major.width": HAIR, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.6,
    "ytick.major.pad": 1.6, "axes.labelpad": 3.0, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
    "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.bbox": "standard", "path.simplify": False,
    "svg.hashsalt": "cold-check"})


# ---------------------------------------------------------------------------------------------------- colour
def tone(col, a):
    """`col` blended with white: a = 1 the colour itself, a = 0 white."""
    r = np.array(to_rgb(col))
    return to_hex(1 - a * (1 - r))


def _lab(h):
    c = np.array(to_rgb(h))
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    x = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
    x = x / np.array([0.95047, 1.0, 1.08883])
    f = np.where(x > (6 / 29) ** 3, np.cbrt(x), x / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _hex_lab(L):
    fy = (L[0] + 16) / 116
    f = np.array([fy + L[1] / 500, fy, fy - L[2] / 200])
    x = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    c = np.linalg.solve(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]), x)
    c = np.clip(c, 0, 1)
    return to_hex(np.clip(np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055), 0, 1))


def blend(c0, c1, t):
    """Perceptual (CIELAB) blend of two colours: t = 0 gives c0, t = 1 gives c1."""
    return _hex_lab((1 - t) * _lab(c0) + t * _lab(c1))


# ---------------------------------------------------------------------------------------------------- page
def figure(h):
    fig = plt.figure(figsize=(W * MM, h * MM))
    fig.patch.set_facecolor("white")
    fig._h = h
    fig.names = {}                                     # id(text) -> gids of the marks the text names (for checks)
    return fig


def text(fig, x, y, s, size=LABEL, **kw):
    """Free text on the page at (x, y) mm from the top-left corner."""
    kw.setdefault("color", INK)
    kw.setdefault("va", "baseline")
    return fig.text(x / W, 1 - y / fig._h, s, fontsize=size, **kw)


def axes_mm(fig, x, y, w, h):
    return layout.axes_mm(fig, x, y, w, h)


def canvas(fig, x, y, w, h):
    """A blank axes in millimetres (y down, equal aspect) for drawn panels; its own coordinates are page mm."""
    ax = axes_mm(fig, x, y, w, h)
    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.set_xlim(x, x + w)
    ax.set_ylim(y + h, y)
    ax.set_aspect("equal")
    return ax


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(HAIR)
    ax.tick_params(colors=INK, length=2.0, width=HAIR, pad=1.6, labelsize=LABEL)
    ax.set_facecolor("none")


def head(fig, x, y, letter, title, sub=None):
    """Panel letter (9 pt bold) and title (8 pt) on one baseline at y; an optional grey subtitle (6.5 pt) below."""
    text(fig, x, y, letter, size=LETTER, weight="bold")
    text(fig, x + 4.6, y, title, size=TITLE)
    if sub:
        text(fig, x + 4.6, y + 4.2, sub, size=LABEL, color=SOFT)


def names(fig, t, *gids):
    """Record that text `t` names the marks with these gids (it may sit on or near them)."""
    fig.names[id(t)] = set(gids)
    return t


# ---------------------------------------------------------------------------------------------------- checks
def _texts(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    out += [t for t in (tick.label1, tick.label2) if t.get_visible()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def _marks(fig, r):
    """Every drawn mark in every axes as (gid, label, segments [n, 2, 2] in px, filled paths in px, half width px)."""
    px = fig.dpi / 72
    out = []
    for ax in fig.axes:
        arts = list(ax.lines) + list(ax.patches) + list(ax.collections)
        for a in arts:
            if not a.get_visible():
                continue
            gid = a.get_gid()
            segs, fills, hw = [], [], 0.0
            if isinstance(a, Line2D):
                xy = a.get_transform().transform(a.get_xydata())
                if a.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
                    segs.append(np.stack([xy[:-1], xy[1:]], 1))
                    hw = a.get_linewidth() * px / 2
                if a.get_marker() not in (None, "", "None", " "):
                    rad = a.get_markersize() * px / 2
                    th = np.linspace(0, 2 * np.pi, 25)
                    for p in xy:
                        ring = p + rad * np.c_[np.cos(th), np.sin(th)]
                        fills.append(ring)
                        segs.append(np.stack([ring[:-1], ring[1:]], 1))
            elif isinstance(a, Patch):
                path = a.get_path().transformed(a.get_transform())
                polys = path.to_polygons(closed_only=False)
                for q in polys:
                    if len(q) > 1:
                        segs.append(np.stack([q[:-1], q[1:]], 1))
                    if a.get_fill() and a.get_facecolor()[3] > 0:
                        fills.append(q)
                hw = a.get_linewidth() * px / 2 if a.get_edgecolor()[3] > 0 else 0.0
            elif isinstance(a, (PolyCollection, LineCollection)):
                tr = a.get_transform()
                lws = a.get_linewidths()
                hw = (float(np.max(lws)) if len(lws) else 0.0) * px / 2
                for p in a.get_paths():
                    for q in p.transformed(tr).to_polygons(closed_only=False):
                        if len(q) > 1:
                            segs.append(np.stack([q[:-1], q[1:]], 1))
                        if isinstance(a, PolyCollection) and len(a.get_facecolor()) and a.get_facecolor()[0][3] > 0:
                            fills.append(q)
            elif isinstance(a, PathCollection):
                offs = a.get_offset_transform().transform(a.get_offsets())
                s = np.broadcast_to(a.get_sizes() if len(a.get_sizes()) else [0.0], (len(offs),))
                th = np.linspace(0, 2 * np.pi, 25)
                for p, ss in zip(offs, s):
                    ring = p + np.sqrt(ss) / 2 * px * np.c_[np.cos(th), np.sin(th)]
                    fills.append(ring)
                    segs.append(np.stack([ring[:-1], ring[1:]], 1))
            else:
                continue
            seg = np.concatenate(segs, 0) if segs else np.zeros((0, 2, 2))
            out.append((gid, f"{type(a).__name__}({gid})", seg, fills, hw, a, ax))
    return out


def _seg_rect(seg, x0, y0, x1, y1):
    """Least distance (px) from each segment to the axis-aligned rectangle; 0 where they meet."""
    if not len(seg):
        return np.zeros(0)
    a, b = seg[:, 0], seg[:, 1]

    def pt_rect(p):
        dx = np.maximum(np.maximum(x0 - p[:, 0], 0), p[:, 0] - x1)
        dy = np.maximum(np.maximum(y0 - p[:, 1], 0), p[:, 1] - y1)
        return np.hypot(dx, dy)

    def pt_seg(c):
        d = b - a
        L2 = (d ** 2).sum(1)
        t = np.clip(((c[None] - a) * d).sum(1) / np.where(L2 > 0, L2, 1), 0, 1)
        q = a + t[:, None] * d
        return np.hypot(*(q - c[None]).T)

    d = np.minimum(pt_rect(a), pt_rect(b))
    for c in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        d = np.minimum(d, pt_seg(np.array(c, float)))
    # a segment that passes through the rectangle with both ends outside: Liang-Barsky slabs
    dd = b - a
    t_in, t_out = np.zeros(len(a)), np.ones(len(a))
    for k, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
        flat = dd[:, k] == 0
        inside = (a[:, k] >= lo) & (a[:, k] <= hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - a[:, k]) / dd[:, k]
            t2 = (hi - a[:, k]) / dd[:, k]
        t_in = np.where(flat, np.where(inside, t_in, np.inf), np.maximum(t_in, np.minimum(t1, t2)))
        t_out = np.where(flat, np.where(inside, t_out, -np.inf), np.minimum(t_out, np.maximum(t1, t2)))
    return np.where(t_in <= t_out, 0.0, d)


def check(fig, gap=GAP):
    """Every problem with the drawn figure, as strings: texts closer than `gap` mm to each other or to a mark they do
    not name, anything outside the page, any mark or text clipped by its axes, any type size outside the three."""
    from matplotlib.path import Path as MPath
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_mm = fig.dpi / 25.4
    g = gap * px_mm
    W_px, H_px = fig.bbox.width, fig.bbox.height
    probs = []
    texts = _texts(fig)
    boxes = [(t, Text.get_window_extent(t, r)) for t in texts]
    # 1. type sizes
    for t, _ in boxes:
        if round(t.get_fontsize(), 2) not in SIZES:
            probs.append(f"type size {t.get_fontsize()} pt: {t.get_text()!r}")
    # 2. text against text: boxes closer than the gap
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
            dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
            if np.hypot(dx, dy) < g:
                probs.append(f"texts {np.hypot(dx, dy) / px_mm:.2f} mm apart: {ta.get_text()!r} / {tb.get_text()!r}")
    # 3. text against marks it does not name
    marks = _marks(fig, r)
    for t, b in boxes:
        named = fig.names.get(id(t), set())
        for gid, lab, seg, fills, hw, art, ax in marks:
            if gid is not None and gid in named:
                continue
            d = _seg_rect(seg, b.x0, b.y0, b.x1, b.y1)
            dmin = (float(d.min()) if len(d) else np.inf) - hw
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            if any(MPath(q).contains_point((cx, cy)) for q in fills if len(q) > 2):
                dmin = 0.0
            if dmin < g:
                probs.append(f"text {max(dmin, 0) / px_mm:.2f} mm from {lab}: {t.get_text()!r}")
    # 4. outside the page
    for t, b in boxes:
        if b.x0 < -0.01 or b.y0 < -0.01 or b.x1 > W_px + 0.01 or b.y1 > H_px + 0.01:
            probs.append(f"text outside the page: {t.get_text()!r} ({b.x0 / px_mm:.1f}..{b.x1 / px_mm:.1f} mm, "
                         f"{(H_px - b.y1) / px_mm:.1f}..{(H_px - b.y0) / px_mm:.1f} mm from the top)")
    for gid, lab, seg, fills, hw, art, ax in marks:
        if not len(seg):
            continue
        p = seg.reshape(-1, 2)
        lo, hi = p.min(0) - hw, p.max(0) + hw
        if lo[0] < -0.01 or lo[1] < -0.01 or hi[0] > W_px + 0.01 or hi[1] > H_px + 0.01:
            probs.append(f"mark outside the page: {lab}")
        # 5. clipped by its own axes
        if art.get_clip_on():
            bb = ax.bbox
            if lo[0] < bb.x0 - 0.5 or lo[1] < bb.y0 - 0.5 or hi[0] > bb.x1 + 0.5 or hi[1] > bb.y1 + 0.5:
                probs.append(f"mark clipped by its axes: {lab} ({(bb.x0 - lo[0]) / px_mm:.2f}, {(bb.y0 - lo[1]) / px_mm:.2f}, "
                             f"{(hi[0] - bb.x1) / px_mm:.2f}, {(hi[1] - bb.y1) / px_mm:.2f} mm past left/bottom/right/top)")
    over = layout.overflow(fig, tol=0.0)
    if over:
        probs.append(f"drawn content past the page edge (mm): {over}")
    return probs


def save(fig, name):
    name += os.environ.get("FIG_SUFFIX", "")             # e.g. FIG_SUFFIX=_v2: a new name beside the one the paper uses
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    print("wrote", OUT / f"{name}.pdf", "| .svg | .png (300 dpi)")


def zoom(fig, name, boxes, dpi=1200):
    """Crops of the page (mm boxes from the top-left) rendered at `dpi` (1200 dpi = 400% of the 300 dpi proof)."""
    from PIL import Image
    ZOOM.mkdir(parents=True, exist_ok=True)
    name += os.environ.get("FIG_SUFFIX", "")
    big = OUT / f"{name}_{dpi}dpi.png"
    fig.savefig(big, dpi=dpi)
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(big)
    k = dpi / 25.4
    paths = []
    for tag, (x0, y0, x1, y1) in boxes.items():
        c = im.crop((int(x0 * k), int(y0 * k), int(x1 * k), int(y1 * k)))
        p = ZOOM / f"{name}_{tag}.png"
        c.save(p)
        paths.append(p)
    big.unlink()
    return paths
