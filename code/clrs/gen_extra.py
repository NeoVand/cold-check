"""The library's samplers and labels for the tasks added after the freeze (a new file: gen.py stays frozen).
Same RandomState calls in the same order as dm-clrs 2.0.3; `python gen_extra.py verify` reproduces the official test
splits and relabels every stored split, as gen.py's verify does.

Tasks: graham_scan, jarvis_march, segments_intersect, find_maximum_subarray_kadane, articulation_points, bridges.
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np
import gen

HULL = ('graham_scan', 'jarvis_march')
GRAPH_X = ('articulation_points', 'bridges')


def _collinear_any(xs, ys, eps=1e-12):
    """The sampler's test: is any triple collinear (|cross| < eps)? Vectorised over the triples in the library's order
    (only the yes/no answer matters, and it is the same)."""
    n = len(xs)
    for a in range(n):
        for b in range(a + 1, n):
            c = np.arange(b + 1, n)
            if not len(c):
                continue
            v1x, v1y = xs[b] - xs[a], ys[b] - ys[a]; v2x, v2y = xs[c] - xs[a], ys[c] - ys[a]
            if (np.abs(v1x * v2y - v1y * v2x) < eps).any():
                return True
    return False


def sample_inputs(task, rng, n, p=None):
    if task in HULL:                                          # ConvexHullSampler: a disk of radius 2, collinear resampled
        for _ in range(1000):
            th = rng.uniform(0.0, 2.0 * np.pi, size=(n,)); rs = 2.0 * np.sqrt(rng.uniform(0.0, 1.0, size=(n,)))
            xs, ys = rs * np.cos(th), rs * np.sin(th)
            if not _collinear_any(xs, ys):
                return [xs, ys]
        raise RuntimeError('collinear')
    if task == 'segments_intersect':                          # SegmentsSampler (n is ignored: 4 points)
        def ccw(xa, ya, xb, yb, xc, yc):
            return (yc - ya) * (xb - xa) > (yb - ya) * (xc - xa)
        def inter(xs, ys):
            return ccw(xs[0], ys[0], xs[2], ys[2], xs[3], ys[3]) != ccw(xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]) and \
                   ccw(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]) != ccw(xs[0], ys[0], xs[1], ys[1], xs[3], ys[3])
        coin = rng.binomial(1, 0.5); xs = rng.uniform(0., 1., size=(4,)); ys = rng.uniform(0., 1., size=(4,))
        while inter(xs, ys) != coin:
            xs = rng.uniform(0., 1., size=(4,)); ys = rng.uniform(0., 1., size=(4,))
        return [xs, ys]
    if task == 'find_maximum_subarray_kadane':
        return [rng.uniform(-1., 1., size=(n,))]
    if task == 'matrix_chain_order':                          # SortingSampler
        return [rng.uniform(0., 1., size=(n,))]
    if task == 'optimal_bst':                                 # OptimalBSTSampler(length = n - 1 keys)
        L = n - 1; arr = rng.uniform(0.0, 1.0, size=(2 * L + 1,)); arr /= np.sum(arr)
        return [arr[:L], arr[L:]]
    if task in GRAPH_X:                                       # ArticulationSampler, p = 0.2 by default
        return [gen.er_graph(rng, n, rng.choice(p or (0.2,)))]
    raise ValueError(task)


def label(task, inp):
    if task == 'graham_scan':
        xs, ys = inp; n = len(xs); inh = np.zeros(n, np.int64); sp = np.arange(n); at = np.zeros(n)
        ccw = lambda i, j, k: ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i])) <= 0
        best = 0
        for i in range(n):
            if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]): best = i
        inh[best] = 1; last = best
        for i in range(n):
            if i != best: at[i] = math.atan2(ys[i] - ys[best], xs[i] - xs[best])
        at[best] = -123456789; ind = np.argsort(at)
        for i in range(1, n):
            if i >= 3:
                while ccw(sp[last], last, ind[i]):
                    pl = last; last = sp[last]; sp[pl] = pl; inh[pl] = 0
            inh[ind[i]] = 1; sp[ind[i]] = last; last = ind[i]
        return dict(in_hull=inh)
    if task == 'jarvis_march':
        xs, ys = inp; n = len(xs); inh = np.zeros(n, np.int64)
        def ccw(i, j, k):
            if k == i or k == j: return False
            return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i])) <= 0
        best = 0
        for i in range(n):
            if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]): best = i
        inh[best] = 1; last = best; end = 0
        while True:
            for i in range(n):
                if end == last or ccw(last, end, i): end = i
            if inh[end] > 0: break
            inh[end] = 1; last = end; end = 0
        return dict(in_hull=inh)
    if task == 'segments_intersect':
        xs, ys = inp
        cp = lambda x1, y1, x2, y2: x1 * y2 - x2 * y1
        di = lambda i, j, k: cp(xs[k] - xs[i], ys[k] - ys[i], xs[j] - xs[i], ys[j] - ys[i])
        on = lambda i, j, k: (min(xs[i], xs[j]) <= xs[k] <= max(xs[i], xs[j])) and (min(ys[i], ys[j]) <= ys[k] <= max(ys[i], ys[j]))
        d = [di(2, 3, 0), di(2, 3, 1), di(0, 1, 2), di(0, 1, 3)]
        o = [on(2, 3, 0), on(2, 3, 1), on(0, 1, 2), on(0, 1, 3)]
        r = 0
        if ((d[0] > 0 and d[1] < 0) or (d[0] < 0 and d[1] > 0)) and ((d[2] > 0 and d[3] < 0) or (d[2] < 0 and d[3] > 0)): r = 1
        elif any(d[q] == 0 and o[q] for q in range(4)): r = 1
        return dict(intersect=r)
    if task == 'find_maximum_subarray_kadane':
        A = inp[0]; bl = bh = 0; bs = A[0]; i = 0; s = A[0]
        for j in range(1, len(A)):
            x = A[j]
            if s + x >= x: s += x
            else: i = j; s = x
            if s > bs: bl, bh, bs = i, j, s
        return dict(start=bl, end=bh)
    if task == 'matrix_chain_order':                          # the library's iterations converge to the DP: s = first argmin
        p = inp[0]; n = len(p); m = np.zeros((n, n)); s_ = np.zeros((n, n), np.int64)
        for l in range(1, n - 1):
            for i in range(1, n - l):
                j = i + l; ks = np.arange(i, j); q = m[i, ks] + m[ks + 1, j] + p[i - 1] * p[ks] * p[j]
                b = int(np.argmin(q)); m[i, j] = q[b]; s_[i, j] = ks[b]
        return dict(s=s_)
    if task == 'optimal_bst':
        p, q = inp; n = len(q); pc = np.zeros(n); pc[:-1] = p; e = np.zeros((n, n)); w = np.zeros((n, n)); root = np.zeros((n, n), np.int64)
        for i in range(n): e[i, i] = q[i]; w[i, i] = q[i]
        for l in range(1, len(p) + 1):
            for i in range(len(p) - l + 1):
                j = i + l; e[i, j] = 1e9; w[i, j] = w[i, j - 1] + p[j - 1] + q[j]
                for r in range(i, j):
                    t = e[i, r] + e[r + 1, j] + w[i, j]
                    if t < e[i, j]: e[i, j] = t; root[i, j] = r
        return dict(root=root)
    if task in GRAPH_X:                                        # by definition: removing it disconnects its component
        A = inp[0]; n = len(A); E = (A != 0); np.fill_diagonal(E, False)
        def comps(E, alive):
            lab = np.full(n, -1); c = 0
            for s in range(n):
                if alive[s] and lab[s] < 0:
                    st = [s]; lab[s] = c
                    while st:
                        u = st.pop()
                        for v in np.nonzero(E[u] & alive)[0]:
                            if lab[v] < 0: lab[v] = c; st.append(v)
                    c += 1
            return c, lab
        base, _ = comps(E, np.ones(n, bool))
        if task == 'articulation_points':
            cut = np.zeros(n, np.int64)
            for v in range(n):
                al = np.ones(n, bool); al[v] = False
                cut[v] = int(comps(E, al)[0] > base - (0 if E[v].any() else 1))
            return dict(is_cut=cut)
        br = np.where(E, 0, -1).astype(np.int64); np.fill_diagonal(br, 0)      # the library stores 0 on the diagonal
        for a, b in zip(*np.nonzero(np.triu(E, 1))):
            E2 = E.copy(); E2[a, b] = E2[b, a] = False
            if comps(E2, np.ones(n, bool))[0] > base: br[a, b] = br[b, a] = 1
        return dict(is_bridge=br)
    raise ValueError(task)


def pack(task, inps, labs):
    N = len(inps)
    if task in GRAPH_X:
        n = len(inps[0][0]); out = dict(n=np.full(N, n, np.int16), pos=np.tile((np.arange(n) / n).astype(np.float32), (N, 1)),
                                        A=np.stack([i[0] for i in inps]).astype(np.float32))
        out['adj'] = ((out['A'] + np.eye(n, dtype=np.float32)) != 0).astype(np.uint8)
    else:
        n = len(inps[0][-1]); out = dict(n=np.full(N, n, np.int16), x_pos=np.tile(np.arange(n) / n, (N, 1)))
        names = ['x', 'y'] if task in HULL + ('segments_intersect',) else ['key']
        if task == 'matrix_chain_order':
            names = ['p']
        if task == 'optimal_bst':
            n = len(inps[0][1]); out = dict(n=np.full(N, n, np.int16), x_pos=np.tile(np.arange(n) / n, (N, 1)),
                                              x_p=np.array([np.r_[i[0], 0.0] for i in inps]), x_q=np.array([i[1] for i in inps]))
            names = []
        for j, nm in enumerate(names):
            out['x_' + nm] = np.array([i[j] for i in inps], np.float64)
    for k in labs[0]:
        out[k] = np.array([l[k] for l in labs]).astype(np.int16)
    return out


def make(task, n, num, seed, p=None):
    rng = np.random.RandomState(seed); inps = [sample_inputs(task, rng, n, p) for _ in range(num)]
    return pack(task, inps, [label(task, i) for i in inps])


def inputs_from_split(task, d, i):
    n = int(d['n'][i])
    if task in GRAPH_X:
        return [d['A'][i, :n, :n].astype(np.float64)]
    if task == 'find_maximum_subarray_kadane':
        return [d['x_key'][i, :n]]
    if task == 'matrix_chain_order':
        return [d['x_p'][i, :n]]
    if task == 'optimal_bst':
        return [d['x_p'][i, :n - 1], d['x_q'][i, :n]]
    return [d['x_x'][i, :n], d['x_y'][i, :n]]


def verify(tasks):
    rep = {}
    for task in tasks:
        off = dict(np.load(gen.DATA / task / 'test.npz')); num = len(off['n'])
        mine = make(task, 4 if task == 'segments_intersect' else 65 if task == 'optimal_bst' else 64, num, 3)
        same = {k: bool(np.array_equal(mine[k], off[k]) if mine[k].dtype.kind != 'f' else np.allclose(mine[k], off[k], atol=1e-6)) for k in off}
        rel = {}
        for f in sorted((gen.DATA / task).glob('*.npz')):
            d = dict(np.load(f)); bad = tot = 0
            for i in range(len(d['n'])):
                lab = label(task, inputs_from_split(task, d, i)); n = int(d['n'][i])
                for k, q in lab.items():
                    t = d[k][i]; q = np.asarray(q)
                    if t.ndim == 0: bad += int(q != t); tot += 1
                    elif t.ndim == 1: bad += int((q[:n] != t[:n]).sum()); tot += n
                    else:
                        m = t[:n, :n] != -1; bad += int((q[:n, :n][m] != t[:n, :n][m]).sum()); tot += int(m.sum())
            rel[f.stem] = [bad, tot]
        rep[task] = dict(test_reproduced=same, relabel_wrong_of=rel); print(task, json.dumps(rep[task]), flush=True)
    return rep


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        tasks = sys.argv[2:] or list(HULL) + ['segments_intersect', 'find_maximum_subarray_kadane'] + list(GRAPH_X) + ['matrix_chain_order', 'optimal_bst']
        json.dump(verify(tasks), open(Path(__file__).parent / ('gen_extra_verify.json' if not sys.argv[2:] else
                                                                'gen_extra_verify_' + '_'.join(sys.argv[2:]) + '.json'), 'w'), indent=1)
