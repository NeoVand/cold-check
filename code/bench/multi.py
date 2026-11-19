"""N-Queens and graph colouring in the same setting, one fit per task, tested at 10x the training size.

Data are ours (CRAM's sets come from GRAM's builder, third-party code we do not run):
- N-Queens: the fit sees the 92 solutions of 8 x 8. Test boards are random solutions of N x N (min-conflicts, seeded)
  with queens removed; the task is to complete them.
- Graph colouring: planted 3-colourable random graphs (vertices split into three colour classes, each cross-class pair
  an edge with probability p); the fit sees one proper colouring of each training graph.

The check is the Sudoku one with one generalisation: a same-value cost per (pair type, value), since here the values
are not symmetric (queen / empty); colours are symmetric, so the colouring costs are tied over colours as in Sudoku.
Pair types: N-Queens, the agreement pattern of (r, c, r - c, r + c); colouring, whether the pair is an edge of the input
graph. Kept relations become units (classes of a relation; for colouring each edge is a unit of two). Count
conditions per (relation, value): n_v >= 1 and n_v <= 1, kept when they held in every unit of every training answer.
Engine: the bounds of engine.py per (unit, value); the warm choice is what the check requires to exist (the open
(unit, value) with n_v >= 1 and the fewest places), else the item with the fewest values.
"""
import time

import numpy as np
from numba import njit


# ------------------------------------------------------------------------------------------------------- the fit --
def fit_pl(examples, ntypes, V, tied, steps=8, lr=4.0, l1=1e-3, members=None, lr_bias=0.5):
    """examples: list of (sol (M,) values 0..V-1, tid (M, M) pair type ids, -1 on the diagonal). Same-value costs
    w[type, value] >= 0 (tied over values if `tied`; only for the answer-member values `members` if given, as in the
    mazes: a cost on two empties of one row is a density term that grows with the row) and a free bias per value
    (CRAM's input cost). Pseudolikelihood at the answers, full batch, gradient from zero, L1 soft-threshold on w,
    projection to w >= 0. All examples are stacked into one list of pairs. Returns ((w, bias), seconds)."""
    t0 = time.perf_counter()
    I, J, TT, X = [], [], [], []
    off = 0
    for sol, tid in examples:
        ii, jj = np.nonzero(tid >= 0)
        I.append(ii + off); J.append(jj + off); TT.append(tid[ii, jj]); X.append(sol)
        off += len(sol)
    I = np.concatenate(I); J = np.concatenate(J); TT = np.concatenate(TT); X = np.concatenate(X)
    Y = np.eye(V)[X]
    YJ = Y[J]
    npairs = np.maximum(np.bincount(TT, minlength=ntypes), 1)
    mask = np.ones(V) if members is None else np.isin(np.arange(V), members).astype(float)
    w = np.zeros((ntypes, V))
    bias = np.zeros(V)
    n = len(X)
    for _ in range(steps):
        cp = w[TT] * YJ  # (pairs, V)
        cost = np.stack([np.bincount(I, weights=cp[:, v], minlength=n) for v in range(V)], 1)
        z = -(cost + bias)
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        contrib = (Y[I] - p[I]) * YJ
        g = np.stack([np.bincount(TT, weights=contrib[:, v], minlength=ntypes) for v in range(V)], 1) / npairs[:, None]
        gb = (Y - p).mean(0)
        if tied:
            g[:] = g.mean(1, keepdims=True)
            gb[:] = 0
        w = np.maximum(w - lr * g - lr * l1, 0.0) * mask
        bias = bias - lr_bias * gb
    return (w, bias), time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------------- the engine --
@njit(cache=True, nogil=True)
def popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True, nogil=True)
def propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, inq, stats):
    nu = unit_size.shape[0]
    ring = np.zeros(nu, np.int64)
    size = 0
    for u in range(nu):
        if inq[u]:
            ring[size] = u
            size += 1
    head = 0
    while size > 0:
        u = ring[head]
        head = (head + 1) % nu
        size -= 1
        inq[u] = False
        s = unit_size[u]
        for v in range(V):
            g = ge1[u, v]
            l = le1[u, v]
            if not (g or l):
                continue
            bit = np.int64(1) << v
            L = 0
            U = 0
            last = -1
            for k in range(s):
                d = dom[unit_items[u, k]]
                stats[1] += 1
                if d & bit:
                    U += 1
                    stats[0] += 1
                    last = unit_items[u, k]
                    if d == bit:
                        L += 1
                        stats[0] += 1
            stats[1] += 4
            if (l and L >= 2) or (g and U == 0):
                return False
            if l and L == 1 and U > 1:
                for k in range(s):
                    it = unit_items[u, k]
                    d = dom[it]
                    if (d & bit) and d != bit:
                        nd = d & ~bit
                        if nd == 0:
                            return False
                        dom[it] = nd
                        for j in range(item_units.shape[1]):
                            w = item_units[it, j]
                            if w >= 0 and not inq[w]:
                                inq[w] = True
                                ring[(head + size) % nu] = w
                                size += 1
            elif g and U == 1 and dom[last] != bit:
                dom[last] = bit
                for j in range(item_units.shape[1]):
                    w = item_units[last, j]
                    if w >= 0 and not inq[w]:
                        inq[w] = True
                        ring[(head + size) % nu] = w
                        size += 1
    return True


@njit(cache=True, nogil=True)
def choose(dom, unit_items, unit_size, ge1, V):
    """The warm choice: first what the check requires to exist, the open (unit, value) with n_v >= 1 required and the
    fewest places (branch over its places); if there is none, the item with the fewest values (branch over them).
    Returns (kind 1 or 0, index, value, size); kind -1 = nothing open."""
    n = dom.shape[0]
    best_c = 1 << 30
    kind = -1
    idx = -1
    val = -1
    if True:
        for u in range(unit_size.shape[0]):
            for v in range(V):
                if not ge1[u, v]:
                    continue
                bit = np.int64(1) << v
                U = 0
                fixed = False
                for k in range(unit_size[u]):
                    d = dom[unit_items[u, k]]
                    if d & bit:
                        U += 1
                        if d == bit:
                            fixed = True
                if not fixed and U > 1 and U < best_c:
                    best_c = U; kind = 1; idx = u; val = v
    if kind < 0:
        for i in range(n):
            c = popcount(dom[i])
            if c > 1 and c < best_c:
                best_c = c; kind = 0; idx = i
    return kind, idx, val, best_c


@njit(cache=True, nogil=True)
def solve(dom0, unit_items, unit_size, item_units, ge1, le1, V, seed, budget, out):
    """Cold derivation, then warm depth-first search on the tightest count, random order, undo on contradiction.
    out: 0 status (1 solved, 0 budget, -1 none), 1 moves, 2 ops, 3 cmp, 4 solved cold."""
    n = dom0.shape[0]
    nu = unit_size.shape[0]
    stats = np.zeros(2, np.int64)
    inq = np.ones(nu, np.bool_)
    dom = dom0.copy()
    for k in range(5):
        out[k] = 0
    if not propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, inq, stats):
        out[0] = -1
        return dom
    kind, idx, val, c = choose(dom, unit_items, unit_size, ge1, V)
    if kind < 0:
        out[0] = 1; out[4] = 1; out[2] = stats[0]; out[3] = stats[1]
        return dom
    rng = np.uint64(seed * 2654435761 + 1)
    maxd = n + 1
    sd = np.zeros((maxd, n), np.int64)
    skind = np.zeros(maxd, np.int64); sidx = np.zeros(maxd, np.int64); sval = np.zeros(maxd, np.int64)
    sopts = np.zeros((maxd, max(V, unit_items.shape[1])), np.int64); sn = np.zeros(maxd, np.int64)
    depth = 0
    moves = 0
    # push root
    sd[0] = dom; skind[0] = kind; sidx[0] = idx; sval[0] = val
    m = 0
    if kind == 0:
        for v in range(V):
            if (dom[idx] >> v) & 1:
                sopts[0, m] = v; m += 1
    else:
        bit = np.int64(1) << val
        for k in range(unit_size[idx]):
            if dom[unit_items[idx, k]] & bit:
                sopts[0, m] = unit_items[idx, k]; m += 1
    sn[0] = m
    result = dom
    while depth >= 0:
        if moves >= budget:
            out[0] = 0
            break
        if sn[depth] == 0:
            depth -= 1
            continue
        rng ^= rng << np.uint64(13); rng ^= rng >> np.uint64(7); rng ^= rng << np.uint64(17)
        p = int(rng % np.uint64(sn[depth]))
        opt = sopts[depth, p]
        sopts[depth, p] = sopts[depth, sn[depth] - 1]
        sn[depth] -= 1
        moves += 1
        d2 = sd[depth].copy()
        if skind[depth] == 0:
            it = sidx[depth]; d2[it] = np.int64(1) << opt
        else:
            it = opt; d2[it] = np.int64(1) << sval[depth]
        for j in range(item_units.shape[1]):
            w = item_units[it, j]
            if w >= 0:
                inq[w] = True
        ok = propagate(d2, unit_items, unit_size, item_units, ge1, le1, V, inq, stats)
        for u in range(nu):
            inq[u] = False
        if not ok:
            continue
        kind, idx, val, c = choose(d2, unit_items, unit_size, ge1, V)
        if kind < 0:
            out[0] = 1
            result = d2
            break
        depth += 1
        sd[depth] = d2; skind[depth] = kind; sidx[depth] = idx; sval[depth] = val
        m = 0
        if kind == 0:
            for v in range(V):
                if (d2[idx] >> v) & 1:
                    sopts[depth, m] = v; m += 1
        else:
            bit = np.int64(1) << val
            for k in range(unit_size[idx]):
                if d2[unit_items[idx, k]] & bit:
                    sopts[depth, m] = unit_items[idx, k]; m += 1
        sn[depth] = m
    if depth < 0:
        out[0] = -1
    out[1] = moves; out[2] = stats[0]; out[3] = stats[1]
    return result


def build_units(classes_per_rel, conds, M, V):
    """classes_per_rel: list (per kept relation) of lists of item arrays; conds[k] = (ge1 (V,), le1 (V,))."""
    U = [(k, np.asarray(c)) for k, cl in enumerate(classes_per_rel) for c in cl if len(c) >= 1]
    smax = max(len(c) for _, c in U)
    ui = -np.ones((len(U), smax), np.int64)
    us = np.zeros(len(U), np.int64)
    ge = np.zeros((len(U), V), np.bool_)
    le = np.zeros((len(U), V), np.bool_)
    lists = [[] for _ in range(M)]
    for u, (k, c) in enumerate(U):
        ui[u, :len(c)] = c; us[u] = len(c)
        ge[u], le[u] = conds[k]
        for x in c:
            lists[x].append(u)
    m = max(1, max(len(x) for x in lists))
    iu = -np.ones((M, m), np.int64)
    for x, l in enumerate(lists):
        iu[x, :len(l)] = l
    return ui, us, iu, ge, le


# ------------------------------------------------------------------------------------------------------ N-Queens --
QF = ('r', 'c', 'r-c', 'r+c')


def queen_funcs(N):
    r, c = np.divmod(np.arange(N * N), N)
    return np.stack([r, c, r - c, r + c], 1)


def queen_types(N):
    f = queen_funcs(N)
    t = ((f[:, None, :] == f[None, :, :]) * (1 << np.arange(4))).sum(-1)
    np.fill_diagonal(t, -1)
    return t  # type id = the 4-bit pattern (0..15)


def all_queens(N):
    sols = []
    cols = [0] * N

    def rec(r, used, d1, d2):
        if r == N:
            sols.append(cols.copy()); return
        for c in range(N):
            if c not in used and (r - c) not in d1 and (r + c) not in d2:
                cols[r] = c
                rec(r + 1, used | {c}, d1 | {r - c}, d2 | {r + c})
    rec(0, frozenset(), frozenset(), frozenset())
    return sols


@njit(cache=True, nogil=True)
def min_conflicts(N, seed, max_steps):
    """A random N-Queens solution (one queen per row; columns by min-conflicts). Returns cols or all -1."""
    np.random.seed(seed)
    cols = np.random.permutation(N)
    colc = np.zeros(N, np.int64); d1 = np.zeros(2 * N, np.int64); d2 = np.zeros(2 * N, np.int64)
    for r in range(N):
        colc[cols[r]] += 1; d1[r - cols[r] + N] += 1; d2[r + cols[r]] += 1
    for step in range(max_steps):
        bad = np.zeros(N, np.int64); nb = 0
        for r in range(N):
            c = cols[r]
            if colc[c] > 1 or d1[r - c + N] > 1 or d2[r + c] > 1:
                bad[nb] = r; nb += 1
        if nb == 0:
            return cols
        r = bad[np.random.randint(nb)]
        c = cols[r]
        colc[c] -= 1; d1[r - c + N] -= 1; d2[r + c] -= 1
        best = 1 << 30; cand = np.zeros(N, np.int64); ncand = 0
        for cc in range(N):
            k = colc[cc] + d1[r - cc + N] + d2[r + cc]
            if k < best:
                best = k; ncand = 0
            if k == best:
                cand[ncand] = cc; ncand += 1
        cc = cand[np.random.randint(ncand)]
        cols[r] = cc
        colc[cc] += 1; d1[r - cc + N] += 1; d2[r + cc] += 1
    return -np.ones(N, np.int64)


def queens_check(steps=32, lr=1.0):
    """Fit at the 92 solutions of 8 x 8; returns the check (kept relations as function indices, conditions)."""
    N = 8
    T = queen_types(N)
    exs = []
    for cols in all_queens(N):
        sol = np.zeros(N * N, np.int64)
        sol[np.arange(N) * N + np.array(cols)] = 1
        exs.append((sol, T))
    t0 = time.perf_counter()
    (w, bias), _ = fit_pl(exs, 16, 2, tied=False, steps=steps, lr=lr, members=[1])
    kept_types = [(t, v) for t in range(16) for v in range(2) if w[t, v] > 0]
    # kept relations: single functions all of whose types (with that bit) are kept for some value
    rels = []
    for f in range(4):
        for v in range(2):
            need = [t for t in range(16) if (t >> f) & 1 and t in np.unique(T[T >= 0])]
            if need and all(w[t, v] > 0 for t in need) and f not in rels:
                rels.append(f)
    f8 = queen_funcs(N)
    conds = []
    for f in rels:
        ge = np.ones(2, bool); le = np.ones(2, bool)
        for sol, _ in exs:
            for key in np.unique(f8[:, f]):
                cells = np.flatnonzero(f8[:, f] == key)
                for v in range(2):
                    n = int((sol[cells] == v).sum())
                    ge[v] &= n >= 1; le[v] &= n <= 1
        conds.append((ge, le))
    dt = time.perf_counter() - t0
    return dict(w=w, bias=bias, kept_types=kept_types, rels=rels, conds=conds), dt


def queens_engine(ck, N):
    f = queen_funcs(N)
    classes = []
    for fi in ck['rels']:
        classes.append([np.flatnonzero(f[:, fi] == key) for key in np.unique(f[:, fi])])
    return build_units(classes, ck['conds'], N * N, 2)


def queens_valid(board, N):
    q = np.argwhere(board.reshape(N, N) == 1)
    if len(q) != N:
        return False
    r, c = q[:, 0], q[:, 1]
    return len(set(r)) == N and len(set(c)) == N and len(set(r - c)) == N and len(set(r + c)) == N


# --------------------------------------------------------------------------------------------- graph colouring --
def planted_graph(n, p, rng, k=3):
    """Planted k-colourable graph: a random colouring, every cross-colour pair an edge with probability p.
    Returns (adjacency (n, n) bool, colouring)."""
    col = rng.integers(k, size=n)
    while len(set(col.tolist())) < k:
        col = rng.integers(k, size=n)
    A = (rng.random((n, n)) < p) & (col[:, None] != col[None, :])
    A = np.triu(A, 1)
    A = A | A.T
    return A, col


def colour_check(graphs, steps=8, lr=4.0):
    """graphs: list of (A, colouring). Pair types: 0 non-edge, 1 edge. Costs tied over colours."""
    exs = []
    for A, col in graphs:
        tid = A.astype(np.int64)
        np.fill_diagonal(tid, -1)
        exs.append((col.astype(np.int64), tid))
    t0 = time.perf_counter()
    (w, bias), _ = fit_pl(exs, 2, 3, tied=True, steps=steps, lr=lr)
    kept = [t for t in range(2) if (w[t] > 0).any()]
    # conditions on edge units (classes of size two): n_v <= 1 held everywhere? n_v >= 1?
    ge = np.ones(3, bool); le = np.ones(3, bool)
    for A, col in graphs:
        i, j = np.nonzero(np.triu(A, 1))
        for v in range(3):
            n = (col[i] == v).astype(int) + (col[j] == v).astype(int)
            ge[v] &= bool((n >= 1).all()); le[v] &= bool((n <= 1).all())
    dt = time.perf_counter() - t0
    return dict(w=w, kept=kept, conds=[(ge, le)]), dt


def colour_engine(ck, A):
    n = len(A)
    if 1 not in ck['kept']:
        raise ValueError('edge relation not kept')
    i, j = np.nonzero(np.triu(A, 1))
    classes = [[np.array([a, b]) for a, b in zip(i, j)]]
    return build_units(classes, ck['conds'], n, 3)


# ----------------------------------------------------------------------------- final version: Luby restarts too --
@njit(cache=True, nogil=True)
def luby(i):
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i = i - (1 << (k - 1)) + 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


@njit(cache=True, nogil=True)
def solve_r(dom0, unit_items, unit_size, item_units, ge1, le1, V, seed, budget, out, restart_base):
    """As solve, restarted on Luby's schedule (restart_base moves times luby(run); 0 = never). The final engine.
    out: 0 status (1 solved, 0 budget, -1 none), 1 moves, 2 ops, 3 cmp, 4 solved cold, 5 restarts."""
    n = dom0.shape[0]
    nu = unit_size.shape[0]
    stats = np.zeros(2, np.int64)
    inq = np.ones(nu, np.bool_)
    root = dom0.copy()
    for k in range(6):
        out[k] = 0
    if not propagate(root, unit_items, unit_size, item_units, ge1, le1, V, inq, stats):
        out[0] = -1
        return root
    kind0, idx0, val0, c = choose(root, unit_items, unit_size, ge1, V)
    if kind0 < 0:
        out[0] = 1; out[4] = 1; out[2] = stats[0]; out[3] = stats[1]
        return root
    rng = np.uint64(seed * 2654435761 + 1)
    maxd = n + 1
    sd = np.zeros((maxd, n), np.int64)
    skind = np.zeros(maxd, np.int64); sidx = np.zeros(maxd, np.int64); sval = np.zeros(maxd, np.int64)
    sopts = np.zeros((maxd, max(V, unit_items.shape[1])), np.int64); sn = np.zeros(maxd, np.int64)
    moves = 0
    run = 0
    result = root
    solved = False
    exhausted = False
    while moves < budget and not solved and not exhausted:
        run += 1
        limit = budget if restart_base <= 0 else restart_base * luby(run)
        run_moves = 0
        depth = 0
        sd[0] = root; skind[0] = kind0; sidx[0] = idx0; sval[0] = val0
        m = 0
        if kind0 == 0:
            for v in range(V):
                if (root[idx0] >> v) & 1:
                    sopts[0, m] = v; m += 1
        else:
            bit = np.int64(1) << val0
            for k in range(unit_size[idx0]):
                if root[unit_items[idx0, k]] & bit:
                    sopts[0, m] = unit_items[idx0, k]; m += 1
        sn[0] = m
        while depth >= 0:
            if moves >= budget or run_moves >= limit:
                break
            if sn[depth] == 0:
                depth -= 1
                continue
            rng ^= rng << np.uint64(13); rng ^= rng >> np.uint64(7); rng ^= rng << np.uint64(17)
            p = int(rng % np.uint64(sn[depth]))
            opt = sopts[depth, p]
            sopts[depth, p] = sopts[depth, sn[depth] - 1]
            sn[depth] -= 1
            moves += 1
            run_moves += 1
            d2 = sd[depth].copy()
            if skind[depth] == 0:
                it = sidx[depth]; d2[it] = np.int64(1) << opt
            else:
                it = opt; d2[it] = np.int64(1) << sval[depth]
            for j in range(item_units.shape[1]):
                w = item_units[it, j]
                if w >= 0:
                    inq[w] = True
            ok = propagate(d2, unit_items, unit_size, item_units, ge1, le1, V, inq, stats)
            for u in range(nu):
                inq[u] = False
            if not ok:
                continue
            kind, idx, val, c = choose(d2, unit_items, unit_size, ge1, V)
            if kind < 0:
                solved = True
                result = d2
                break
            depth += 1
            sd[depth] = d2; skind[depth] = kind; sidx[depth] = idx; sval[depth] = val
            m = 0
            if kind == 0:
                for v in range(V):
                    if (d2[idx] >> v) & 1:
                        sopts[depth, m] = v; m += 1
            else:
                bit = np.int64(1) << val
                for k in range(unit_size[idx]):
                    if d2[unit_items[idx, k]] & bit:
                        sopts[depth, m] = unit_items[idx, k]; m += 1
            sn[depth] = m
        if not solved and depth < 0:
            exhausted = True
    out[0] = 1 if solved else (-1 if exhausted else 0)
    out[1] = moves; out[2] = stats[0]; out[3] = stats[1]; out[5] = run - 1
    return result
