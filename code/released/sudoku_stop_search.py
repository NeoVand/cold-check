"""Search for a board that breaks a Sudoku rule and still passes CRAM's stopping test E(x) <= tau + 1e-3 (released weights).
Two peer cells are held at one digit (the cheapest conflict); every other cell is annealed to minimise the energy."""
import zipfile, itertools, json, sys
import numpy as np
TAU = {0: -4.9e-6, 1: -3.1e-6, 2: -1.65e-4}     # highest training-solution energy per seed (their tracked record)
res = {}
for seed in (0, 1, 2):
    z = zipfile.ZipFile(f'ext/CRAM/checkpoints/sudoku-extreme/seed{seed}/model.pt')
    st = [np.frombuffer(z.read(n), '<f4') for n in z.namelist() if '/data/' in n and not n.endswith('.pkl')]
    W = [s for s in st if s.size == 3240 * 81][0].reshape(3240, 9, 9).astype(np.float64)
    assert np.count_nonzero(W - np.einsum('paa->pa', W)[:, :, None] * np.eye(9)) <= 1
    pairs = list(itertools.combinations(range(81), 2))
    D = np.zeros((81, 81, 9))
    for q, (i, j) in enumerate(pairs): D[i, j] = D[j, i] = np.diag(W[q])
    def energy(x): return 0.5 * sum(D[i, j, x[i]] for i in range(81) for j in range(81) if i != j and x[i] == x[j])
    def conflicts(x):
        c = 0
        for i, j in pairs:
            r1, c1, r2, c2 = i // 9, i % 9, j // 9, j % 9
            if x[i] == x[j] and (r1 == r2 or c1 == c2 or (r1 // 3 == r2 // 3 and c1 // 3 == c2 // 3)): c += 1
        return c
    diag = np.einsum('paa->pa', W)
    pos = np.where(diag > 0.2, diag, np.inf)
    q, a = np.unravel_index(pos.argmin(), pos.shape); u, v = pairs[q]
    rng = np.random.default_rng(seed)
    best = (1e9, None)
    for rep in range(40):
        x = rng.integers(0, 9, 81); x[u] = x[v] = a
        onehot = np.zeros((81, 9)); onehot[np.arange(81), x] = 1
        F = np.einsum('ija,ja->ia', D, onehot)          # field: cost of digit a at cell i
        free = [i for i in range(81) if i not in (u, v)]
        steps = 60000
        for t in range(steps):
            T = 0.25 * (0.004 / 0.25) ** (t / steps)
            i = free[rng.integers(len(free))]
            pr = np.exp(-(F[i] - F[i].min()) / T); pr /= pr.sum()
            b = rng.choice(9, p=pr)
            if b != x[i]:
                F[:, x[i]] -= D[:, i, x[i]]; F[:, b] += D[:, i, b]; x[i] = b
        for _ in range(3):                               # zero-temperature polish
            for i in free:
                b = int(F[i].argmin())
                if F[i, b] < F[i, x[i]] - 1e-12: F[:, x[i]] -= D[:, i, x[i]]; F[:, b] += D[:, i, b]; x[i] = b
        e = energy(x)
        if e < best[0]: best = (e, x.copy())
    e, x = best
    thr = TAU[seed] + 1e-3
    print(f'seed {seed}: cheapest conflict costs {diag[q, a]:.3f} (cells {u},{v}, digit {a + 1}); lowest energy found with it held: {e:.4f};'
          f' rule conflicts on that board: {conflicts(x)}; stop threshold {thr:.5f}; passes the stop test: {e <= thr}')
    res[seed] = dict(energy=float(e), conflicts=int(conflicts(x)), threshold=thr, passes=bool(e <= thr), board=(x + 1).reshape(9, 9).tolist(), held=[int(u), int(v), int(a + 1)], cost=float(diag[q, a]))
json.dump(res, open('released/sudoku_stop_search.json', 'w'), indent=1)
