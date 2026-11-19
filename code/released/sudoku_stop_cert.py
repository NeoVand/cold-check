"""Is CRAM's Sudoku stopping test sound for every board, judged from the released weights alone?
Stop rule: E(x) <= tau + 1e-3, with E(x) = sum_(pairs) W_p[x_i, x_j] + sum_i B[i, x_i].
A board with a rule conflict pays at least c_min (smallest positive cost). It can earn back at most the negative costs.
Crude bound: every pair's most negative entry at once. Tighter: each cell holds one digit, so solve the earning problem."""
import zipfile, json, itertools
import numpy as np
for seed in (0, 1, 2):
    p = f'ext/CRAM/checkpoints/sudoku-extreme/seed{seed}/'
    z = zipfile.ZipFile(p + 'model.pt')
    st = [np.frombuffer(z.read(n), '<f4') for n in z.namelist() if '/data/' in n and not n.endswith('.pkl')]
    W = [s for s in st if s.size == 3240 * 81][0].reshape(3240, 9, 9).astype(np.float64)
    Bs = [s for s in st if s.size == 81 * 9]
    fin = json.load(open(p + 'final.json'))
    tau = None
    def find(d, k):
        if isinstance(d, dict):
            for kk, v in d.items():
                if kk == k: return v
                r = find(v, k)
                if r is not None: return r
        if isinstance(d, list):
            for v in d:
                r = find(v, k)
                if r is not None: return r
        return None
    tau = find(fin, 'tau')
    pairs = list(itertools.combinations(range(81), 2))
    def peer(i, j):
        r1, c1, r2, c2 = i // 9, i % 9, j // 9, j % 9
        return r1 == r2 or c1 == c2 or (r1 // 3 == r2 // 3 and c1 // 3 == c2 // 3)
    P = np.array([peer(i, j) for i, j in pairs])
    diag = np.einsum('paa->pa', W)
    off = W.copy(); off[:, np.arange(9), np.arange(9)] = 0
    cmin = diag[P].min()
    neg_pairs = np.where((W < 0).any((1, 2)))[0]
    crude = np.minimum(W.reshape(3240, -1).min(1), 0).sum()
    Bmax = max(np.abs(b).max() for b in Bs)
    print(f'seed {seed}: tau {tau}; peer pairs {P.sum()}, their equal-digit costs in [{diag[P].min():.3f}, {diag[P].max():.3f}], all positive: {(diag[P] > 0).all()}'
          f'; off-diagonal nonzeros {int((off != 0).sum())}; |B| max {Bmax:.2e}')
    print(f'   negative entries {int((W < 0).sum())} on {len(neg_pairs)} pairs (peer pairs among them: {int(P[neg_pairs].sum())}); most negative {W.min():.4f};'
          f' sum of each pair\'s most negative entry {crude:.3f}; sum of all negative entries {W[W < 0].sum():.3f}')
    print(f'   crude certificate: c_min + (all negatives at once) = {cmin + crude:.3f}  vs threshold {0 if tau is None else tau + 1e-3:.5f}  ->', 'SOUND' if tau is not None and cmin + crude > tau + 1e-3 else 'not proved by the crude bound')
    # tighter: maximum earning = min over digit assignments of the negative part only (a small Potts problem on the 'negative' graph).
    # Upper-bound the earning per connected component by brute force when small, else by coordinate descent (a lower bound on E, not a proof).
    import collections
    adj = collections.defaultdict(set)
    for q in neg_pairs:
        i, j = pairs[q]; adj[i].add(j); adj[j].add(i)
    seen, comps = set(), []
    for v in adj:
        if v in seen: continue
        comp, stack = [], [v]; seen.add(v)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in adj[u]:
                if w not in seen: seen.add(w); stack.append(w)
        comps.append(sorted(comp))
    print('   negative graph: cells', len(adj), 'components', len(comps), 'sizes', sorted((len(c) for c in comps), reverse=True)[:8])
    json.dump(dict(seed=seed, tau=tau, cmin=float(cmin), crude=float(crude), n_neg=int((W < 0).sum())), open(f'released/sudoku_stop_seed{seed}.json', 'w'))
