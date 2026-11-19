"""Certificate attempt for CRAM's Sudoku stop test, from the released weights.
Lemma (counting): a complete board that is not a valid Sudoku has at least two conflicting peer pairs (if exactly one
pair clashed, counting that digit by rows and by columns, or by boxes, would give 10 and 9).
So a wrong board pays at least the two cheapest conflict costs. It can earn back at most
LB = 1/2 sum_i min_a sum_j min(0, min_b S[i,j,a,b])  +  sum_i min_a B[i,a]   (each cell holds one digit).
If  2 c_min + LB > tau + 1e-3  then no wrong board passes the stop test, for any puzzle."""
import zipfile, itertools, json
import numpy as np
TAU = {0: -4.9e-6, 1: -3.1e-6, 2: -1.65e-4}
out = {}
for seed in (0, 1, 2):
    z = zipfile.ZipFile(f'ext/CRAM/checkpoints/sudoku-extreme/seed{seed}/model.pt')
    st = [np.frombuffer(z.read(n), '<f4') for n in z.namelist() if '/data/' in n and not n.endswith('.pkl')]
    W = [s for s in st if s.size == 3240 * 81][0].reshape(3240, 9, 9).astype(np.float64)
    Bc = [s for s in st if s.size == 81 * 9]
    pairs = list(itertools.combinations(range(81), 2))
    S = np.zeros((81, 81, 9, 9))
    for q, (i, j) in enumerate(pairs): S[i, j] = W[q]; S[j, i] = W[q].T
    def peer(i, j):
        r1, c1, r2, c2 = i // 9, i % 9, j // 9, j % 9
        return r1 == r2 or c1 == c2 or (r1 // 3 == r2 // 3 and c1 // 3 == c2 // 3)
    P = np.array([[i != j and peer(i, j) for j in range(81)] for i in range(81)])
    eq = np.einsum('ijaa->ija', S)
    costs = np.sort(eq[P].ravel())                       # every conflict cost appears twice (i,j) and (j,i)
    two = costs[0] + costs[2]                            # the two cheapest distinct conflicts
    neg = np.minimum(S, 0).min(3)                        # [i, j, a]: the most negative entry cell j can give cell i at digit a
    LB_pairs = 0.5 * neg.sum(1).min(1).sum()
    # which B is the model's: bound with the worse of the stored copies
    LB_B = min(b.reshape(81, 9).astype(np.float64).min(1).sum() for b in Bc)
    thr = TAU[seed] + 1e-3
    lhs = two + LB_pairs + LB_B
    print(f'seed {seed}: two cheapest conflicts {two:.4f}; most a board can earn back from negative pair costs {LB_pairs:.4f}; from input costs {LB_B:.5f};'
          f' so every wrong board has E >= {lhs:.4f}; stop threshold {thr:.5f}  ->', 'PROVED SOUND' if lhs > thr else 'not proved')
    out[seed] = dict(two=float(two), LB_pairs=float(LB_pairs), LB_B=float(LB_B), bound=float(lhs), threshold=thr, proved=bool(lhs > thr))
json.dump(out, open('released/sudoku_stop_cert.json', 'w'), indent=1)
