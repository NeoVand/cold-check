"""Completed-answer test on BFS parents (predictions registered before this ran).
Label: parent[v] = the neighbour that discovered v (CLRS-30's rule: queue order, neighbours scanned by index).
Completion from the label alone: depth (number of ancestors) or place in the order of discovery (number of nodes before).
Fit at the answer: logistic regression on cold features of a pair (u, v): its own comparison bits, and yes/no bits
'is there a third node k passing a test on (k, u) and a test on (k, v)'. No loop in training."""
import json, time, itertools
import numpy as np
rng = np.random.default_rng(1)
INF = 10 ** 9

def ugraph(n, extra=1.0):
    """A connected random graph: a random tree (each node joins a random earlier node, in a random order) plus
    random extra edges, about `extra` per node. (A plain random graph of average degree 3 is almost never connected at
    256 nodes, which made the first version of this script stall.)"""
    order = rng.permutation(n); A = np.zeros((n, n), bool)
    for i in range(1, n):
        j = order[rng.integers(i)]; A[order[i], j] = A[j, order[i]] = True
    B = np.triu(rng.random((n, n)) < 2 * extra / n, 1); A |= B | B.T
    return A
def bfs(A, s):
    n = len(A); par = np.full(n, -1); par[s] = s; q = [s]
    for u in q:
        for v in np.nonzero(A[u])[0]:
            if par[v] < 0: par[v] = u; q.append(v)
    return par, np.array(q)
def completion(par, s):
    """From the label alone: depth = number of ancestors; order by (parent's place, index), level by level."""
    n = len(par); depth = np.zeros(n, np.int64)
    for v in range(n):
        u = v
        while u != s: u = par[u]; depth[v] += 1
    place = np.full(n, INF); place[s] = 0; pp = np.full(n, -1)
    for d in range(1, depth.max() + 1):
        lev = np.nonzero(depth == d)[0]; pp[lev] = place[par[lev]]
        order = sorted(lev, key=lambda v: (pp[v], v))
        base = (depth < d).sum()
        for i, v in enumerate(order): place[v] = base + i
    return depth, place, pp

def pair_bits(reg):
    """Comparison bits of one node register on every ordered pair (a, b): a<b, a=b, a>b ('not yet' = INF compares as later)."""
    a, b = reg[:, None], reg[None, :]
    return [(a < b), (a == b) & (a < INF), (a > b)]
def features(A, regs, idx):
    """Cold features of each ordered pair (u, v). regs: list of node registers (depth or place); idx: node indices."""
    n = len(A); own = [A] ; first = []; names = ['edge']
    cmp_idx = pair_bits(idx)
    for r in regs:
        cb = pair_bits(r); own += cb; names += ['lt', 'eq', 'gt']
        first += cb + [cb[1] & cmp_idx[0]]                    # tests on the leg (k, u): k before/level/after u; level and lower index
    own.append(cmp_idx[0]); names.append('idx_lt')
    first.append(cmp_idx[0])
    second = [A]                                              # test on the leg (k, v): k is a neighbour of v
    for r in regs: second.append(A & pair_bits(r)[0])        # ... and k is before v
    F = [o.astype(np.float64) for o in own]
    for f1 in first:
        for f2 in second:
            N = f1.T.astype(np.float64) @ f2.astype(np.float64)   # N[u, v] = #k with f1[k, u] and f2[k, v]
            F.append((N >= 1).astype(np.float64))
    F.append(np.ones((n, n)))
    return np.stack(F, -1)
def logreg(X, y, steps=60, l2=1e-3):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y) / len(y) + l2 * w)
    return w

def dataset(kind, graphs):
    X, y = [], []
    for A, s, par in graphs:
        n = len(A); depth, place, pp = completion(par, s); idx = np.arange(n)
        F = features(A, [depth] if kind == 'depth' else [place], idx)
        P = np.zeros((n, n), bool); P[par, idx] = True; P[s, s] = False
        m = ~np.eye(n, dtype=bool); m[:, s] = False            # every pair (u, v), u != v, v not the source
        X.append(F[m]); y.append(P[m])
    return np.concatenate(X), np.concatenate(y).astype(np.float64)
def read_at_answer(kind, w, A, s, par):
    n = len(A); depth, place, pp = completion(par, s); idx = np.arange(n)
    sc = features(A, [depth] if kind == 'depth' else [place], idx) @ w
    np.fill_diagonal(sc, -1e9); pred = sc.argmax(0)
    ok = [(pred[v] == par[v]) for v in range(n) if v != s]
    valid = [A[pred[v], v] and depth[pred[v]] == depth[v] - 1 for v in range(n) if v != s]
    return sum(ok), sum(valid), n - 1

def run(kind, w, wb, A, s, top, gate=None):
    """Run the fitted rules from nothing. top=True: places start at 'not yet' (INF) except the source, and a node keeps
    its parent once it has one. top=False: every count starts at 0 and everything is recomputed each loop."""
    n = len(A); idx = np.arange(n); par = np.full(n, -1); par[s] = s
    reg = np.full(n, INF if top else 0); reg[s] = 0; pp = np.full(n, INF if top else 0); pp[s] = -1
    for loop in range(min(n + 2, 80)):
        F = features(A, [reg], idx); sc = F @ w; np.fill_diagonal(sc, -1e9)
        if gate is not None: sc = np.where(F[..., gate].all(-1), sc, -1e9)   # keep every own bit that held on all training parents
        new_par = par.copy(); changed = False
        for v in range(n):
            if v == s or (top and par[v] >= 0): continue
            u = int(sc[:, v].argmax())
            if sc[u, v] > 0 and new_par[v] != u: new_par[v] = u; changed = True
            elif sc[u, v] <= 0 and not top and new_par[v] != -1: new_par[v] = -1; changed = True
        par = new_par
        if kind == 'depth':                                    # depth of a node = its parent's depth + 1 (hop count along the pointer)
            new = reg.copy()
            for v in range(n):
                if v != s and par[v] >= 0 and reg[par[v]] < INF: new[v] = reg[par[v]] + 1
        else:                                                  # place = number of nodes before; 'before' is the fitted rule on (parent's place, index)
            pp = np.where((par >= 0) & (reg[np.maximum(par, 0)] < INF), reg[np.maximum(par, 0)], INF if top else 0); pp[s] = -1
            has = pp < INF
            cb = pair_bits(pp) ; ci = pair_bits(idx)
            Fb = np.stack([cb[0], cb[1], cb[2], ci[0], cb[1] & ci[0], np.ones((n, n), bool)], -1).astype(np.float64)
            before = (Fb @ wb > 0) & has[:, None] & has[None, :]; np.fill_diagonal(before, False)
            new = np.where(has, before.sum(0), INF if top else 0); new[s] = 0
        if not changed and (new == reg).all(): break
        reg = new
    return par, loop

graphs = lambda n, m: [(lambda A, s: (A, s, bfs(A, s)[0]))(ugraph(n), int(rng.integers(n))) for _ in range(m)]
train = [g for n in rng.integers(8, 17, 30) for g in graphs(int(n), 1)]
# the fitted 'before' rule (order completion): before(a, b) from comparisons of (parent's place, index)
Xb, yb = [], []
for A, s, par in train:
    n = len(A); depth, place, pp = completion(par, s); idx = np.arange(n); cb = pair_bits(pp); ci = pair_bits(idx)
    Fb = np.stack([cb[0], cb[1], cb[2], ci[0], cb[1] & ci[0], np.ones((n, n), bool)], -1).astype(np.float64)
    m = ~np.eye(n, dtype=bool); Xb.append(Fb[m]); yb.append((place[:, None] < place[None, :])[m])
Xb, yb = np.concatenate(Xb), np.concatenate(yb).astype(np.float64)
t0 = time.perf_counter(); wb = logreg(Xb, yb); tb = time.perf_counter() - t0
print(f"'before' rule: wrong pairs at the answer {int(((Xb @ wb > 0) != (yb > 0)).sum())} of {len(yb)}; fit {tb * 1000:.0f} ms")
out = {}
for kind in ('order', 'depth'):
    X, y = dataset(kind, train)
    gate = [j for j in range(5) if X[y > 0][:, j].all()]
    print(kind, 'gate (own bits true on every training parent; 0 edge, 1 before, 2 level, 3 after, 4 lower index):', gate)
    t0 = time.perf_counter(); w = logreg(X, y); tf = time.perf_counter() - t0
    print(f'{kind} completion: {X.shape[1]} features, {len(y)} pairs from {len(train)} graphs; fit {tf * 1000:.0f} ms; wrong pairs at the answer (training) {int(((X @ w > 0) != (y > 0)).sum())}')
    out[kind] = dict(fit_ms=tf * 1000)
    for n, reps in ((16, 100), (64, 100), (256, 20), (1024, 3)):
        gs = graphs(n, reps)
        a = np.array([read_at_answer(kind, w, *g) for g in gs]).sum(0)
        line = f'   {n:5d} nodes, {reps:3d} graphs | at the answer: parents right {a[0]}/{a[2]} ({100 * a[0] / a[2]:.2f}%), a valid BFS parent {100 * a[1] / a[2]:.2f}%'
        res = dict(at_answer=[int(a[0]), int(a[2])], at_answer_valid=int(a[1]))
        for top, g in ((False, None), (True, None), (True, gate)):
            if n > 64 and g is None: continue
            r = v = t = 0; loops = []
            for A, s, par in gs:
                p, k = run(kind, w, wb, A, s, top, g); loops.append(k)
                d = completion(par, s)[0]
                r += int((p == par).sum()) - 1; t += n - 1
                v += sum(1 for x in range(n) if x != s and p[x] >= 0 and A[p[x], x] and d[p[x]] == d[x] - 1)
            line += f" | from nothing, {('top+gate' if g is not None else 'top') if top else 'zero'} start: {r}/{t} ({100 * r / t:.2f}%), valid {100 * v / t:.2f}%, loops <= {max(loops)}"
            res[('top_gate' if g is not None else 'top') if top else 'zero'] = [r, t, v, max(loops)]
        print(line, flush=True); out[kind][n] = res
json.dump(out, open('walkthrough/completed_answer_bfs.json', 'w'), indent=1, default=float)
