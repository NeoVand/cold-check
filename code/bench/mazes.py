"""Mazes in our setting: a check fitted at the labelled answer, run cold; no hand-built layer.

Items are cells; a cell's token is wall / open / S / G; its value is off / on (on = on the path). As in CRAM, walls are
given off and S, G given on; only open cells are decided.

Relation vocabulary (position-free): "side neighbours" (|dr| + |dc| = 1) and "corner neighbours" (|dr| = |dc| = 1),
counted only over cells that exist, so a cell at the board's edge simply has fewer neighbours.

1. The check (Maze-Unique). Pseudolikelihood at the labelled boards: each open cell's value predicted, by logistic
   regression, from n_r(i, u), the number of its r-neighbours holding value u, for r in the vocabulary and u in
   {off, on}, plus a bias. L1 soft-threshold after each gradient step, from zero. A relation is kept when any of its
   weights is non-zero.
2. Kept counts (the same rule as Sudoku): for every kept relation r, token t, own value a and neighbour value u, the
   range [lo, hi] of n_r(i, u) over every cell of every training answer. Neighbourhoods are bounded, so the ranges
   mention no size.
3. Cold derivation: bounds on those counts (engine_nbr), from the top: every open cell starts with both values
   possible. With the side relation and the range {2} for an on open cell this is dead-end filling, derived.
4. Maze-Hard. The clock: the cold reach run from G over the kept relation; clock(i) = the loop at which cell i is first
   reached (a generic register of a looped cold block; it equals the number of steps to G, and is labelled as such).
   The path rule is fitted at the labelled paths: for a path cell u and each side neighbour w, features
   [w open], [clock(w) < clock(u)], [clock(w) = clock(u)], [clock(w) > clock(u)], [w unreached], and the direction of w
   (down, left, right, up); softmax over the four neighbours, true next cell = the path neighbour that comes next
   along the label from S. Gradient steps from zero, L2. Kept conditions: every binary feature that held on all
   training positives is required. Read cold: from S, step to the best-scoring neighbour that meets the kept
   conditions, until G.
"""
import time
from collections import deque

import numpy as np
from numba import njit

SIDE = ((1, 0), (0, -1), (0, 1), (-1, 0))  # down, left, right, up
CORNER = ((1, 1), (1, -1), (-1, 1), (-1, -1))
DIRNAMES = ('down', 'left', 'right', 'up')
RELS = (SIDE, CORNER)
RELNAMES = ('side', 'corner')


def neighbours(H, W, rels=RELS):
    """(H*W, R, 4) int64 neighbour index per relation, -1 where the neighbour does not exist."""
    nb = -np.ones((H * W, len(rels), 4), np.int64)
    for r in range(H):
        for c in range(W):
            for k, offs in enumerate(rels):
                for j, (dr, dc) in enumerate(offs):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        nb[r * W + c, k, j] = rr * W + cc
    return nb


def counts(val, nb):
    """val (H*W,) 0/1 -> n (H*W, R, 2): neighbours per relation with value off/on."""
    v = np.concatenate([val, [-1]])
    x = v[nb]  # -1 for missing
    return np.stack([(x == 0).sum(-1), (x == 1).sum(-1)], -1)


# ------------------------------------------------------------------------------------------------- the check fit --
def fit_check(toks, paths, steps=20, lr=1.0, l1=1e-3):
    """toks (N, H, W) int8, paths (N, H, W) bool. Returns the check dict and fit seconds.
    Counts are of answer members only (neighbours that are on), as Sudoku counts cells holding a value: a count of
    off neighbours would also count the board's edge (an edge cell has fewer neighbours), which edge-free training
    mazes cannot show."""
    t0 = time.perf_counter()
    N, H, W = toks.shape
    R = len(RELS)
    t = toks.reshape(N, -1)
    v8 = paths.astype(np.int8)
    v = v8.reshape(N, -1)
    pad = np.pad(v8, ((0, 0), (1, 1), (1, 1)))  # a cell beyond the edge does not exist, so it is never on
    n_on = np.zeros((N, H, W, R), np.int8)
    for r, offs in enumerate(RELS):
        for dr, dc in offs:
            n_on[..., r] += pad[:, 1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    n_on = n_on.reshape(N, H * W, R)
    o = t == 1
    # the pseudolikelihood depends on a cell only through its feature row, so identical rows are pooled (exact)
    key = n_on[o].astype(np.int64) @ (9 ** np.arange(R))
    yo = v[o].astype(np.float64)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ysum = np.bincount(inv, weights=yo)
    feats = np.stack([(uk // 9 ** r) % 9 for r in range(R)], 1).astype(np.float64)
    X = np.concatenate([feats, np.ones((len(uk), 1))], 1)
    th = np.zeros(R + 1)
    M = cnt.sum()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (cnt * p - ysum) / M
        th = th - lr * g
        th[:-1] = np.sign(th[:-1]) * np.maximum(np.abs(th[:-1]) - lr * l1, 0)  # L1 on the relation weights
    kept = [r for r in range(R) if th[r] != 0]
    # kept count ranges lo/hi[token, own value, relation] of on-neighbours over every cell of every training answer
    lo = np.zeros((4, 2, R), np.int64)
    hi = np.full((4, 2, R), 8, np.int64)
    grp = (t.astype(np.int64) * 2 + v).ravel()
    for r in range(R):
        h = np.bincount(grp * 9 + n_on[..., r].ravel(), minlength=72).reshape(4, 2, 9)
        for tok in range(4):
            for a in range(2):
                nz = np.flatnonzero(h[tok, a])
                if len(nz):
                    lo[tok, a, r], hi[tok, a, r] = nz[0], nz[-1]
    for r in range(R):
        if r not in kept:
            lo[:, :, r] = 0
            hi[:, :, r] = 8
    dt = time.perf_counter() - t0
    return dict(theta=th, kept=kept, lo=lo, hi=hi), dt


# ------------------------------------------------------------------------------- cold derivation on neighbour counts --
@njit(cache=True, nogil=True)
def propagate_nbr(dom, tok, nb, kept, lo, hi, stats):
    """dom (n,) masks over {off = bit 0, on = bit 1}. Bounds on the kept count ranges, a worklist of items.
    Returns False on a contradiction. stats[0] ops (count increments), stats[1] comparisons."""
    n = dom.shape[0]
    R = nb.shape[1]
    inq = np.ones(n, np.bool_)
    q = np.arange(n)
    qn = n
    head = 0
    ring = np.zeros(n, np.int64)
    for i in range(n):
        ring[i] = i
    size = n
    while size > 0:
        i = ring[head]
        head = (head + 1) % n
        size -= 1
        inq[i] = False
        t = tok[i]
        for a in range(2):
            if not (dom[i] >> a) & 1:
                continue
            bad = False
            for kk in range(kept.shape[0]):
                r = kept[kk]
                u = 1
                L = 0
                U = 0
                for j in range(4):
                    w = nb[i, r, j]
                    if w >= 0:
                        stats[1] += 1
                        if (dom[w] >> u) & 1:
                            U += 1
                            stats[0] += 1
                            if dom[w] == (1 << u):
                                L += 1
                                stats[0] += 1
                stats[1] += 2
                if L > hi[t, a, r] or U < lo[t, a, r]:
                    bad = True
            if bad:
                dom[i] &= ~(1 << a)
                if dom[i] == 0:
                    return False
                for kk in range(kept.shape[0]):
                    r = kept[kk]
                    for j in range(4):
                        w = nb[i, r, j]
                        if w >= 0 and not inq[w]:
                            inq[w] = True
                            ring[(head + size) % n] = w
                            size += 1
                if not inq[i]:
                    inq[i] = True
                    ring[(head + size) % n] = i
                    size += 1
        # forcing on neighbours when my value is fixed
        if dom[i] == 1 or dom[i] == 2:
            a = 0 if dom[i] == 1 else 1
            for kk in range(kept.shape[0]):
                r = kept[kk]
                u = 1
                L = 0
                U = 0
                for j in range(4):
                    w = nb[i, r, j]
                    if w >= 0:
                        if (dom[w] >> u) & 1:
                            U += 1
                            stats[0] += 1
                            if dom[w] == (1 << u):
                                L += 1
                                stats[0] += 1
                stats[1] += 2
                newv = -1
                if U == lo[t, a, r] and L < U:
                    newv = 1 << u  # every neighbour that can be on must be
                elif L == hi[t, a, r] and U > L:
                    newv = 0  # no other neighbour may be on
                if newv >= 0:
                    for j in range(4):
                        w = nb[i, r, j]
                        if w >= 0 and (dom[w] >> u) & 1 and dom[w] != (1 << u):
                            if newv > 0:
                                dom[w] = newv
                            else:
                                dom[w] &= ~(1 << u)
                            if dom[w] == 0:
                                return False
                            if not inq[w]:
                                inq[w] = True
                                ring[(head + size) % n] = w
                                size += 1
                            # w changed, so every cell that counts w must be looked at again
                            for kk2 in range(kept.shape[0]):
                                r2 = kept[kk2]
                                for j2 in range(4):
                                    x = nb[w, r2, j2]
                                    if x >= 0 and not inq[x]:
                                        inq[x] = True
                                        ring[(head + size) % n] = x
                                        size += 1
    return True


def cold_unique(check, tok):
    """Cold derivation on one maze (tok (H, W)). Returns (board bool (H, W) or None if not fully decided, ops, cmp,
    number of open cells left undecided)."""
    H, W = tok.shape
    nb = neighbours(H, W)
    t = tok.ravel().astype(np.int64)
    dom = np.where(t == 0, 1, np.where(t >= 2, 2, 3)).astype(np.int64)
    stats = np.zeros(2, np.int64)
    ok = propagate_nbr(dom, t, nb, np.array(check['kept'], np.int64), check['lo'], check['hi'], stats)
    und = int(((dom == 3)).sum())
    board = (dom == 2).reshape(H, W) if ok and und == 0 else None
    return board, int(stats[0]), int(stats[1]), und, ok


def passes(check, tok, board):
    """The fitted check on a full board: every kept count in its range."""
    H, W = tok.shape
    nb = neighbours(H, W)
    v = board.ravel().astype(np.int64)
    n = np.concatenate([v, [0]])[nb].sum(-1)  # (HW, R) on-neighbours
    t = tok.ravel()
    for r in check['kept']:
        if ((n[:, r] < check['lo'][t, v, r]) | (n[:, r] > check['hi'][t, v, r])).any():
            return False
    return True


# ---------------------------------------------------------------------------------------- Maze-Hard: clock + rule --
@njit(cache=True, nogil=True)
def reach_clock(openm, nb, r, src, stats):
    """Cold reach from src over relation r, run loop by loop; clock[i] = the loop at which i is first reached (-1 never).
    One loop = every open cell with a reached r-neighbour becomes reached."""
    n = openm.shape[0]
    clock = -np.ones(n, np.int64)
    clock[src] = 0
    frontier = np.zeros(n, np.int64)
    frontier[0] = src
    fn = 1
    t = 0
    nxt = np.zeros(n, np.int64)
    while fn > 0:
        t += 1
        nn = 0
        for k in range(fn):
            i = frontier[k]
            for j in range(4):
                w = nb[i, r, j]
                stats[1] += 1
                if w >= 0 and openm[w] and clock[w] < 0:
                    clock[w] = t
                    stats[0] += 1  # the loop counter written into the cell
                    nxt[nn] = w
                    nn += 1
        for k in range(nn):
            frontier[k] = nxt[k]
        fn = nn
        stats[0] += 1  # the loop counter
    return clock


def rule_features(openm, clock, nb, u):
    """(4, 9) features of the side neighbours of u: open, less, equal, greater, unreached, dir one-hot (4)."""
    F = np.zeros((4, 9))
    for j in range(4):
        w = nb[u, 0, j]
        if w < 0 or not openm[w]:
            F[j, 4] = 1 if w >= 0 and clock[w] < 0 else 0
        if w >= 0 and openm[w]:
            F[j, 0] = 1
            if clock[w] < 0:
                F[j, 4] = 1
            elif clock[w] < clock[u]:
                F[j, 1] = 1
            elif clock[w] == clock[u]:
                F[j, 2] = 1
            else:
                F[j, 3] = 1
        F[j, 5 + j] = 1
    return F


def label_order(path, S, G, nb, H, W):
    """The label's cells in order from S to G (the label is a simple path)."""
    on = set(np.flatnonzero(path.ravel()).tolist())
    seq = [S]
    prev = -1
    while seq[-1] != G:
        u = seq[-1]
        nxts = [nb[u, 0, j] for j in range(4) if nb[u, 0, j] >= 0 and nb[u, 0, j] in on and nb[u, 0, j] != prev]
        assert len(nxts) == 1
        prev = u
        seq.append(int(nxts[0]))
    return seq


@njit(cache=True, nogil=True)
def walk_features(on, openm, clock, nb, S, G):
    """Walk the label from S to G (each step to the one side neighbour on the label that is not the previous cell)
    and return, for each step, the features of the four side neighbours (rule_features) and the index of the true
    next cell."""
    n = on.shape[0]
    F = np.zeros((n, 4, 9))
    y = np.zeros(n, np.int64)
    m = 0
    u = S
    prev = -1
    while u != G and m < n:
        nxt = -1
        for j in range(4):
            w = nb[u, 0, j]
            F[m, j, 5 + j] = 1
            if w >= 0 and openm[w]:
                F[m, j, 0] = 1
                if clock[w] < 0:
                    F[m, j, 4] = 1
                elif clock[w] < clock[u]:
                    F[m, j, 1] = 1
                elif clock[w] == clock[u]:
                    F[m, j, 2] = 1
                else:
                    F[m, j, 3] = 1
            if w >= 0 and on[w] and w != prev:
                nxt = j
        y[m] = nxt
        prev = u
        u = nb[u, 0, nxt]
        m += 1
    return F[:m], y[:m]


def fit_rule(toks, paths, steps=100, lr=1.0, l2=1e-3, rel=0):
    """The path rule, fitted at the labelled paths (see the module docstring). The move relation of a path is the
    side relation (a path on a grid steps between side neighbours: given by the label format, not fitted)."""
    t0 = time.perf_counter()
    N, H, W = toks.shape
    nb = neighbours(H, W)
    Fs, ys = [], []
    stats = np.zeros(2, np.int64)
    for tk, p in zip(toks, paths):
        t = tk.ravel()
        openm = t > 0
        S = int(np.flatnonzero(t == 2)[0]); G = int(np.flatnonzero(t == 3)[0])
        clock = reach_clock(openm, nb, rel, G, stats)
        F, y = walk_features(p.ravel(), openm, clock, nb, S, G)
        Fs.append(F); ys.append(y)
    F = np.concatenate(Fs)  # (M, 4, 9)
    y = np.concatenate(ys)
    pos = F[np.arange(len(y)), y]  # (M, 9) features of the true next cells
    # identical (4 x 9) blocks are pooled (exact for the softmax loss)
    code = F.reshape(len(F), -1).astype(np.int64) @ (np.int64(1) << np.arange(36, dtype=np.int64))  # features are 0/1
    _, first, inv, cnt = np.unique(code, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Fu = F[first]
    Yc = np.zeros((len(Fu), 4))
    np.add.at(Yc, (inv, y), 1.0)
    M = len(y)
    w = np.zeros(9)
    for _ in range(steps):
        z = Fu @ w
        z -= z.max(1, keepdims=True)
        pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
        g = (np.einsum('u,uj,ujf->f', cnt, pr, Fu) - np.einsum('uj,ujf->f', Yc, Fu)) / M
        w -= lr * (g + l2 * w)
    binary = list(range(5))  # open, nearer, level, farther, unreached
    required = [f for f in binary if (pos[:, f] == 1).all()]
    forbidden = [f for f in binary if (pos[:, f] == 0).all()]
    dt = time.perf_counter() - t0
    return dict(w=w, required=required, forbidden=forbidden, n_steps=int(M), n_patterns=int(len(Fu))), dt


def solve_hard(rule, tok, rel=0, nb=None):
    """Cold read: clock from G, then from S the best neighbour meeting the kept conditions, until G.
    Returns (board or None, ops, cmp)."""
    H, W = tok.shape
    nb = neighbours(H, W) if nb is None else nb
    t = tok.ravel()
    openm = t > 0
    S = int(np.flatnonzero(t == 2)[0]); G = int(np.flatnonzero(t == 3)[0])
    stats = np.zeros(2, np.int64)
    clock = reach_clock(openm, nb, rel, G, stats)
    ops, cmp = int(stats[0]), int(stats[1])
    board = np.zeros(H * W, bool)
    board[S] = True
    u = S
    for _ in range(H * W):
        if u == G:
            break
        F = rule_features(openm, clock, nb, u)
        ok = np.ones(4, bool)
        for f in rule['required']:
            ok &= F[:, f] == 1
        for f in rule['forbidden']:
            ok &= F[:, f] == 0
        cmp += 4 * (len(rule['required']) + len(rule['forbidden']))
        if not ok.any():
            return None, ops, cmp
        s = F @ rule['w']
        ops += 4 * int((F != 0).sum(1).max())  # score additions
        s[~ok] = -np.inf
        u = int(nb[u, 0, int(np.argmax(s))])
        board[u] = True
    if u != G:
        return None, ops, cmp
    return board.reshape(H, W), ops, cmp


# ------------------------------------------------------------------------- maze builders (as in released/modal_cram_check.py, verbatim) --
def L_maze(h, w, extra):
    off = 1 if extra else 0
    H, W = h + 2, w + 2 + off
    g = np.zeros((H, W), bool)
    g[0:h, 0:w] = True
    g[0, w] = True
    S = (0, w + 1)
    g[0:h + 2, w + 1] = True
    if extra:
        g[h, 0] = True; g[h + 1, 0] = True; G = (h + 1, 1); g[h + 1, 1:w + 2] = True
    else:
        g[h, 0] = True; G = (h + 1, 0); g[h + 1, 0:w + 2] = True
    g = np.pad(g, 1)
    return g, (S[0] + 1, S[1] + 1), (G[0] + 1, G[1] + 1)


def sym(g, S, G, k):
    m = np.zeros(g.shape, np.int8); m[S] = 1; m[G] = 2
    g2, m2 = np.rot90(g, k % 4), np.rot90(m, k % 4)
    if k >= 4: g2, m2 = g2.T, m2.T
    S2 = tuple(int(x) for x in np.argwhere(m2 == 1)[0]); G2 = tuple(int(x) for x in np.argwhere(m2 == 2)[0])
    return np.ascontiguousarray(g2), S2, G2


def tree_maze(h, rng):
    """Uniform random spanning tree (Wilson's algorithm) of an h x h lattice of rooms on a (2h+1) x (2h+1) grid."""
    g = np.zeros((2 * h + 1, 2 * h + 1), bool)
    cells = [(r, c) for r in range(h) for c in range(h)]
    intree = np.zeros((h, h), bool)
    root = cells[rng.integers(len(cells))]
    intree[root] = True
    g[2 * root[0] + 1, 2 * root[1] + 1] = True
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for start in rng.permutation(len(cells)):
        u = cells[start]
        if intree[u]:
            continue
        nxt = {}
        cur = u
        while not intree[cur]:
            while True:
                d = steps[rng.integers(4)]
                v = (cur[0] + d[0], cur[1] + d[1])
                if 0 <= v[0] < h and 0 <= v[1] < h:
                    break
            nxt[cur] = v
            cur = v
        cur = u
        while not intree[cur]:
            v = nxt[cur]
            intree[cur] = True
            g[2 * cur[0] + 1, 2 * cur[1] + 1] = True
            g[cur[0] + v[0] + 1, cur[1] + v[1] + 1] = True
            cur = v
    return g


def tokens(g, S, G):
    t = g.astype(np.int8)
    t[S] = 2; t[G] = 3
    return t


def bfs_dist(open_, src):
    H, W = open_.shape; d = np.full((H, W), -1); d[src] = 0; q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in SIDE:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and open_[rr, cc] and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1; q.append((rr, cc))
    return d


def label_rule(open_, S, G):
    """The benchmark's label (released/pointer_test.py greedy): from S, the side neighbour nearest G, ties by down, left,
    right, up. Used only to label our own generated mazes, never by the solver."""
    dG = bfs_dist(open_, G)
    H, W = open_.shape
    p = [S]
    while p[-1] != G:
        r, c = p[-1]
        best = None
        for k, (dr, dc) in enumerate(SIDE):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and open_[rr, cc] and dG[rr, cc] >= 0:
                key = (dG[rr, cc], k)
                if best is None or key < best[0]:
                    best = (key, (rr, cc))
        p.append(best[1])
    lab = np.zeros(open_.shape, bool)
    for c in p:
        lab[c] = True
    return lab


def path_cells_tree(g, S, G):
    """The unique S-G path of a tree maze, as a mask."""
    d = bfs_dist(g, G)
    lab = np.zeros(g.shape, bool)
    cur = S
    lab[cur] = True
    while cur != G:
        r, c = cur
        for dr, dc in SIDE:
            rr, cc = r + dr, c + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] and d[rr, cc] == d[r, c] - 1:
                cur = (rr, cc)
                break
        lab[cur] = True
    return lab
