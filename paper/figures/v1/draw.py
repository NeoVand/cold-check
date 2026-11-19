"""Small drawing helpers shared by the v1 figures (page millimetres, y down), in the grammar of our earlier paper's figures."""
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, FancyBboxPatch
from kit import INK, HAIR

NODE_R, NODE_LW = 1.25, 0.5
HEAD, HEADW = 0.95, 0.75


def seg(ax, p, q, col=INK, lw=HAIR, z=3, gid=None, cap="round", **kw):
    return ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, zorder=z, solid_capstyle=cap, gid=gid, **kw)[0]


def arrow(ax, p, q, col=INK, lw=0.7, r0=0.0, r1=0.0, z=5, gid=None, head=HEAD, hw=HEADW):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    u = d / float(np.hypot(*d))
    a, b = p + u * r0, q - u * r1
    nrm = np.array([-u[1], u[0]])
    base = b - u * head
    ax.add_patch(Polygon([b, base + nrm * hw / 2, base - nrm * hw / 2], closed=True, fc=col, ec=col, lw=0.2,
                         zorder=z + 0.1, joinstyle="miter", gid=gid))
    seg(ax, a, base + u * 0.15, col, lw, z, gid=gid, cap="butt")


def curve_arrow(ax, pts, col=INK, lw=0.7, z=5, gid=None, head=HEAD, hw=HEADW):
    """A polyline (already sampled curve) with an arrowhead at its last point."""
    pts = np.asarray(pts, float)
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    keep = L <= L[-1] - head * 0.9
    tip = pts[-1]
    d = tip - pts[keep][-1]
    u = d / np.hypot(*d)
    nrm = np.array([-u[1], u[0]])
    base = tip - u * head
    ax.plot(*np.vstack([pts[keep], base + u * 0.15]).T, color=col, lw=lw, zorder=z, gid=gid, solid_capstyle="butt")
    ax.add_patch(Polygon([tip, base + nrm * hw / 2, base - nrm * hw / 2], closed=True, fc=col, ec=col, lw=0.2,
                         zorder=z + 0.1, gid=gid))


def node(ax, p, r=NODE_R, fc="white", ec=INK, lw=NODE_LW, z=6, gid=None):
    ax.add_patch(Circle(p, r, fc=fc, ec=ec, lw=lw, zorder=z, gid=gid))


def ring(ax, p, r, col=INK, lw=0.5, z=6, gid=None, ls="-"):
    ax.add_patch(Circle(p, r, fc="none", ec=col, lw=lw, zorder=z, gid=gid, ls=ls))


def rect(ax, x, y, w, h, fc="white", ec=INK, lw=NODE_LW, z=3, gid=None, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="miter", gid=gid, **kw))


def bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def arc(p, q, h, n=60):
    """A symmetric arc from p to q bulging by h (mm; negative = up on the page)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    t = np.linspace(0, 1, n)[:, None]
    c = m + np.array([0, h]) * 4 / 3
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * c + t ** 2 * q
