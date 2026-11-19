"""Candidate C, first experiment: fit a pair rule at the answer, each pair predicted from the others (pseudolikelihood
on pairs), with no loop in training; then run the fitted rule cold (hard threshold) from the empty state until rest.
Task 1, reachability: answer R[a,b] = 'b can be reached from a' (a != b). Features of a pair: its own edge, and the
number of third nodes k with R[a,k] and R[k,b] (the witness count of the cold layer). The pair's own bit is never used.
Task 2, BFS parents (the depth is hidden): answer P[u,v] = 'u is v's parent'. Features: the edge, the source marks, and
witness counts over the legs' edge and parent bits. Predictions were written down before this ran."""
import time, json
import numpy as np
rng = np.random.default_rng(0)

def digraph(n, deg=1.3):
    A = (rng.random((n, n)) < deg / n).astype(np.float64); np.fill_diagonal(A, 0); return A
def closure(A):
    R = A.copy()
    while True:
        R2 = np.minimum(1, R + (R @ R > 0)); np.fill_diagonal(R2, 0)
        if (R2 == R).all(): return R
        R = R2
def logreg(X, y, steps=400, l2=1e-4):
    w = np.zeros(X.shape[1])
    for _ in range(steps):                                   # Newton steps on the logistic loss
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w

# ---- task 1: reachability ----
t0 = time.perf_counter()
X, y = [], []
train = [digraph(int(rng.integers(8, 17))) for _ in range(10)]
for A in train:
    R = closure(A); n = len(A); off = ~np.eye(n, dtype=bool)
    C = R @ R                                               # zero diagonal in R, so k = a and k = b never count
    X.append(np.stack([A[off], C[off], np.ones(off.sum())], 1)); y.append(R[off])
X, y = np.concatenate(X), np.concatenate(y)
w = logreg(X, y); fit_s = time.perf_counter() - t0
print(f'reachability: fitted on {len(train)} graphs of 8-16 nodes, {len(y)} pairs, in {fit_s * 1000:.1f} ms; weights edge {w[0]:.2f}, witness count {w[1]:.2f}, bias {w[2]:.2f};'
      f' wrong pairs at the answer (training graphs): {int(((X @ w > 0) != (y > 0)).sum())}')
res = dict(fit_ms=fit_s * 1000, weights=w.tolist(), sizes={})
for n, reps in ((64, 100), (256, 100), (1024, 20)):
    wrong = pairs = 0; loops = []
    for _ in range(reps):
        A = digraph(n); R = closure(A); S = np.zeros_like(A); k = 0
        while True:
            S2 = ((w[0] * A + w[1] * (S @ S) + w[2]) > 0).astype(np.float64); np.fill_diagonal(S2, 0); k += 1
            if (S2 == S).all(): break
            S = S2
        wrong += int((S != R).sum()); pairs += n * (n - 1); loops.append(k)
    print(f'   {n:5d} nodes, {reps} graphs: wrong pairs {wrong} of {pairs}; loops to rest: median {int(np.median(loops))}, max {max(loops)}; share of pairs reachable {R.sum() / (n * (n - 1)):.2f}')
    res['sizes'][n] = dict(graphs=reps, wrong=wrong, pairs=pairs, loops_max=max(loops))

# ---- task 2: BFS parents, depth hidden ----
def ugraph(n, deg=3.0):
    while True:
        A = np.triu((rng.random((n, n)) < deg / n), 1); A = (A | A.T).astype(np.float64)
        R = closure(A)
        if R.sum() == n * (n - 1): return A
def bfs_parents(A, s):
    n = len(A); par = np.full(n, -1); par[s] = s; q = [s]
    for u in q:
        for v in np.nonzero(A[u])[0]:
            if par[v] < 0: par[v] = u; q.append(v)
    P = np.zeros((n, n)); 
    for v in range(n):
        if v != s: P[par[v], v] = 1
    return P, par
def feats(A, P, s):
    n = len(A); src_u = np.zeros((n, n)); src_u[s, :] = 1; src_v = np.zeros((n, n)); src_v[:, s] = 1
    F = [A, src_u, src_v, A @ A, A @ P, P @ A, P @ P, P.T @ A, A @ P.T, np.ones((n, n))]
    # counts whose legs would read the pair's own bit P[u,v] do not occur: k ranges over third nodes and P has zero diagonal
    off = ~np.eye(n, dtype=bool)
    return np.stack([f[off] for f in F], 1), off
X, y = [], []
for _ in range(30):
    n = int(rng.integers(8, 17)); A = ugraph(n); s = int(rng.integers(n)); P, _ = bfs_parents(A, s)
    f, off = feats(A, P, s); X.append(f); y.append(P[off])
X, y = np.concatenate(X), np.concatenate(y)
t0 = time.perf_counter(); w2 = logreg(X, y); fit2 = time.perf_counter() - t0
print(f'BFS parents: fitted on 30 graphs, {len(y)} pairs, in {fit2 * 1000:.1f} ms; wrong pairs at the answer {int(((X @ w2 > 0) != (y > 0)).sum())} of {len(y)} ({int(y.sum())} are parent pairs)')
for n in (16, 64):
    right = tot = 0
    for _ in range(50):
        A = ugraph(n); s = int(rng.integers(n)); P, par = bfs_parents(A, s)
        S = np.zeros((n, n))
        for k in range(n):
            f, off = feats(A, S, s); sc = np.full((n, n), -1e9); sc[off] = f @ w2
            S2 = (sc > 0).astype(np.float64)
            if (S2 == S).all(): break
            S = S2
        f, off = feats(A, S, s); sc = np.full((n, n), -1e9); sc[off] = f @ w2
        sc = np.where(A > 0, sc, -1e9)                      # a parent must be a neighbour
        pred = sc.argmax(0); ok = [pred[v] == par[v] for v in range(n) if v != s]
        right += sum(ok); tot += len(ok)
    print(f'   {n:3d} nodes: parents right {right} of {tot} ({100 * right / tot:.1f}%)')
    res[f'bfs_{n}'] = dict(right=right, total=tot)
json.dump(res, open('walkthrough/fit_at_answer.json', 'w'), indent=1, default=float)
