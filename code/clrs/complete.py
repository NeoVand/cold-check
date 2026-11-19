"""The shared completion library: what is added to an answer, computed from the answer (or an input relation) alone.

Three operations, used by every task that needs one:
  depth(pointer)       the number of ancestors along a pointer (closure of the pointer, then a count); "not yet" (inf)
                       for an item whose chain does not end at a marked root, or never ends. Pointer doubling,
                       ceil(log2 n) + 1 steps.
  bottleneck(weights)  closure of a weighted relation in (min, max): the smallest possible largest weight on a path.
                       Repeated min-max squaring (ceil(log2 n) steps); for n > 256 the same closure by single linkage
                       (checked equal to the squaring on every 64-node test input, see check_bottleneck).
  reach(relation)      transitive closure (reflexive), by repeated boolean squaring.
No step of any task's algorithm is used: these are the closure and the count, applied to the answer or the input.
"""
import numpy as np

INF = np.inf


def depth(pi, root_ok=None):
    """pi[v] = pointer (v itself or -1 = a root). root_ok: boolean mask of roots that count (None: every root)."""
    n = len(pi); pi = np.where(pi < 0, np.arange(n), pi)
    if root_ok is not None:                                  # a marked root is a root whatever its own pointer says
        pi = np.where(root_ok, np.arange(n), pi)
    is_root = pi == np.arange(n)
    nxt, dist = pi.copy(), (~is_root).astype(np.int64)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        dist = dist + np.where(is_root[nxt], 0, dist[nxt]); nxt = nxt[nxt]
    ok = is_root[nxt] & (True if root_ok is None else root_ok[nxt])
    return np.where(ok, dist.astype(np.float64), INF)


def dist(pi, W, root_ok=None):
    """The sum of the edge weights along the pointer, W[pi[v], v] per step (weighted depth); inf where depth is."""
    n = len(pi); pi = np.where(pi < 0, np.arange(n), pi)
    if root_ok is not None:
        pi = np.where(root_ok, np.arange(n), pi)
    is_root = pi == np.arange(n)
    step = np.where(is_root, 0.0, W[pi, np.arange(n)])
    step = np.where(np.isnan(step), np.inf, step)
    nxt, tot = pi.copy(), step.copy()
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        tot = tot + np.where(is_root[nxt], 0.0, tot[nxt]); nxt = nxt[nxt]
    ok = is_root[nxt] & (True if root_ok is None else root_ok[nxt])
    return np.where(ok, tot, np.inf)


def bottleneck(W):
    """W: (n, n) weights, nan = no edge. Returns B (n, n): min over paths of the max weight; inf if no path; the
    diagonal is -inf (the empty path)."""
    n = len(W); E = np.where(np.isnan(W), INF, W)
    if n > 256:
        return _bottleneck_linkage(E)
    B = E.copy(); np.fill_diagonal(B, -INF)
    for _ in range(int(np.ceil(np.log2(max(n, 2))))):
        B = np.minimum(B, np.min(np.maximum(B[:, :, None], B[None, :, :]), axis=1))
    return B


def _bottleneck_linkage(E):
    """The same closure by single linkage: add edges by increasing weight; when two groups join, every pair across
    them gets that weight."""
    n = len(E); iu = np.triu_indices(n, 1); w = E[iu]; keep = np.isfinite(w)
    a, b, w = iu[0][keep], iu[1][keep], w[keep]; order = np.argsort(w, kind='stable')
    B = np.full((n, n), INF); np.fill_diagonal(B, -INF)
    comp = np.arange(n); members = {i: [i] for i in range(n)}
    for e in order:
        x, y = comp[a[e]], comp[b[e]]
        if x == y:
            continue
        mx, my = np.array(members[x]), np.array(members[y])
        B[np.ix_(mx, my)] = w[e]; B[np.ix_(my, mx)] = w[e]
        if len(mx) < len(my):
            x, y, mx, my = y, x, my, mx
        comp[my] = x; members[x].extend(members.pop(y))
    return B


def reach(R):
    n = len(R); C = R | np.eye(n, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(n, 2))))):
        C = (C.astype(np.float32) @ C.astype(np.float32)) > 0.5
    return C


def check_bottleneck(Ws):
    """Squaring and single linkage agree (the two ways of computing the one closure)."""
    for W in Ws:
        E = np.where(np.isnan(W), INF, W); B1 = E.copy(); np.fill_diagonal(B1, -INF)
        for _ in range(int(np.ceil(np.log2(len(W))))):
            B1 = np.minimum(B1, np.min(np.maximum(B1[:, :, None], B1[None, :, :]), axis=1))
        assert np.array_equal(B1, _bottleneck_linkage(E))
    return True
