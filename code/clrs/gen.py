"""CLRS-30 inputs and labels at any size, following DeepMind's library (dm-clrs 2.0.3) without importing it.

Inputs: the library's samplers (clrs/_src/samplers.py), with the same np.random.RandomState calls in the same order,
so seed 3 at length 64 reproduces the official test split bit for bit (checked by `verify`). Labels: the library's
algorithms (clrs/_src/algorithms/*.py) rewritten without the probes, with the same loops and tie-breaks. Bigger sizes:
the same sampler at length 256 or 1,024 and another seed. Arrays use the packed format of the official split files
(witness_clrs/generate.py of github.com/NeoVand/cold-attention): graph tasks n, pos, A, adj, s, outputs; the others n, x_<input>, outputs.

python gen.py verify            reproduces every official test split of the tasks below and relabels every split file
"""
import sys, json
from pathlib import Path
import numpy as np

import os
DATA = Path(os.environ.get('CLRS_DATA', Path(__file__).resolve().parent / 'clrs30'))   # the official splits (README)
SORTS = ('insertion_sort', 'bubble_sort', 'heapsort', 'quicksort')
GRAPH = ('bfs', 'mst_kruskal', 'mst_prim', 'dfs', 'topological_sort', 'strongly_connected_components', 'bellman_ford',
         'dijkstra', 'dag_shortest_paths', 'floyd_warshall')


# ---------------- samplers (same RandomState calls, same order) ----------------
def er_graph(rng, n, p, directed=False, acyclic=False, weighted=False, low=0., high=1.):
    mat = rng.binomial(1, p, size=(n, n))
    if not directed:
        mat *= np.transpose(mat)
    elif acyclic:
        mat = np.triu(mat, k=1)
        perm = rng.permutation(n)
        mat = mat[perm, :][:, perm]
    if weighted:
        w = rng.uniform(low=low, high=high, size=(n, n))
        if not directed:
            w *= np.transpose(w)
            w = np.sqrt(w + 1e-3)
        mat = mat.astype(float) * w
    return mat


def community_graph(rng, n, k=4, p=0.5, eps=0.01, directed=True):
    mat = np.zeros((n, n)); los, his = [], []; lo = 0
    for i in range(k):
        hi = n if i == k - 1 else lo + n // k
        mat[lo:hi, lo:hi] = er_graph(rng, hi - lo, p, directed=directed)
        los.append(lo); his.append(hi); lo = hi
    toggle = er_graph(rng, n, eps, directed=directed)
    for i in range(k):
        for j in range(i):
            toggle[los[i]:his[i], los[j]:his[j]] *= 0
    mat = np.where(toggle > 0.0, (1.0 - (mat > 0.0)) * toggle, mat)
    perm = rng.permutation(n)
    return mat[perm, :][:, perm]


def sample_inputs(task, rng, n, p=None):
    """One input as the library's sampler draws it (default sampler arguments unless p is given)."""
    if task in SORTS or task in ('minimum', 'quickselect'):
        return [rng.uniform(0., 1., size=(n,))]
    if task == 'binary_search':
        arr = rng.uniform(0., 1., size=(n,)); arr.sort(); x = rng.uniform(0., 1.)
        return [x, arr]
    if task == 'activity_selector':
        a1 = rng.uniform(0., 1., size=(n,)); a2 = rng.uniform(0., 1., size=(n,))
        return [np.minimum(a1, a2), np.maximum(a1, a2)]
    if task == 'task_scheduling':
        d = rng.randint(0, high=n, size=(n,)) + 1; w = rng.uniform(0., 1., size=(n,))
        return [d, w]
    if task == 'lcs_length':
        n2 = n // 2; n1 = n - n2
        return [rng.randint(0, high=4, size=(n1,)), rng.randint(0, high=4, size=(n2,))]
    if task in ('naive_string_matcher', 'kmp_matcher'):     # MatcherSampler; n = haystack + needle
        ln = (1 if n < 5 else n // 5) if p is None else rng.randint(1, high=1 - p)   # p < 0 here: a random needle length
        needle = rng.randint(0, high=4, size=(ln,)); hay = rng.randint(0, high=4, size=(n - ln,))
        pos = rng.choice(n - ln - ln); hay[pos:pos + ln] = needle
        return [hay, needle]
    if task == 'bfs':
        g = er_graph(rng, n, rng.choice(p or (0.5,)))
        return [g, rng.choice(n)]
    if task == 'mst_kruskal':
        return [er_graph(rng, n, rng.choice(p or (0.2,)), weighted=True)]
    if task in ('mst_prim', 'bellman_ford', 'dijkstra'):       # BellmanFordSampler
        g = er_graph(rng, n, rng.choice(p or (0.5,)), weighted=True)
        return [g, rng.choice(n)]
    if task == 'dag_shortest_paths':
        g = er_graph(rng, n, rng.choice(p or (0.5,)), directed=True, acyclic=True, weighted=True)
        return [g, rng.choice(n)]
    if task == 'floyd_warshall':
        return [er_graph(rng, n, rng.choice(p or (0.5,)), weighted=True)]
    if task == 'dfs':
        return [er_graph(rng, n, rng.choice(p or (0.5,)), directed=True)]
    if task == 'topological_sort':
        return [er_graph(rng, n, rng.choice(p or (0.5,)), directed=True, acyclic=True)]
    if task == 'strongly_connected_components':
        return [community_graph(rng, n, p=rng.choice(p or (0.5,)))]
    raise ValueError(task)


# ---------------- labels (the library's loops without probes) ----------------
def label(task, inp):
    if task in SORTS:
        key = inp[0]; order = np.argsort(key, kind='stable'); pred = np.arange(len(key))
        pred[order[1:]] = order[:-1]
        return dict(pred=pred)
    if task == 'minimum':
        A = inp[0]; m = 0
        for i in range(1, len(A)):
            if A[m] > A[i]: m = i
        return dict(min=m)
    if task == 'binary_search':
        x, A = inp; low, high = 0, len(A) - 1
        while low < high:
            mid = (low + high) // 2
            if x <= A[mid]: high = mid
            else: low = mid + 1
        return dict(**{'return': high})
    if task == 'quickselect':
        A = inp[0].copy(); pos = np.arange(len(A)); p, r, i = 0, len(A) - 1, len(A) // 2
        while True:
            x = A[r]; ii = p - 1
            for j in range(p, r):
                if A[j] <= x:
                    ii += 1; A[ii], A[j] = A[j], A[ii]; pos[ii], pos[j] = pos[j], pos[ii]
            A[ii + 1], A[r] = A[r], A[ii + 1]; pos[ii + 1], pos[r] = pos[r], pos[ii + 1]
            q = ii + 1; k = q - p
            if i == k: return dict(median=int(pos[q]))
            elif i < k: r = q - 1
            else: p, i = q + 1, i - k - 1
    if task == 'activity_selector':
        s, f = inp; A = np.zeros(len(s), np.int64); ind = np.argsort(f); A[ind[0]] = 1; k = ind[0]
        for m in range(1, len(s)):
            if s[ind[m]] >= f[k]: A[ind[m]] = 1; k = ind[m]
        return dict(selected=A)
    if task == 'task_scheduling':
        d, w = inp; A = np.zeros(len(d), np.int64); ind = np.argsort(-w); A[ind[0]] = 1; t = 1
        for i in range(1, len(d)):
            if t < d[ind[i]]: A[ind[i]] = 1; t += 1
        return dict(selected=A)
    if task in ('naive_string_matcher', 'kmp_matcher'):     # the first place the needle occurs (both algorithms)
        T, P = inp
        for s_ in range(len(T) - len(P) + 1):
            if (T[s_:s_ + len(P)] == P).all():
                return dict(match=s_)
        return dict(match=len(T))
    if task == 'lcs_length':
        x, y = inp; n, m = len(x), len(y); c = np.zeros((n, m), np.int64); b = np.zeros((n, m), np.int64)
        for i in range(n):
            if x[i] == y[0]: c[i, 0], b[i, 0] = 1, 0
            elif i > 0 and c[i - 1, 0] == 1: c[i, 0], b[i, 0] = 1, 1
            else: c[i, 0], b[i, 0] = 0, 1
        for j in range(m):
            if x[0] == y[j]: c[0, j], b[0, j] = 1, 0
            elif j > 0 and c[0, j - 1] == 1: c[0, j], b[0, j] = 1, 2
            else: c[0, j], b[0, j] = 0, 1
        for i in range(1, n):            # the library iterates Jacobi sweeps to the fixed point; this is that fixed point
            for j in range(1, m):
                if x[i] == y[j]: c[i, j], b[i, j] = c[i - 1, j - 1] + 1, 0
                elif c[i - 1, j] >= c[i, j - 1]: c[i, j], b[i, j] = c[i - 1, j], 1
                else: c[i, j], b[i, j] = c[i, j - 1], 2
        full = np.full((n + m, n + m), -1, np.int64); full[:n, n:] = b
        return dict(b=full)
    if task == 'bfs':
        A, s = inp; n = len(A); reach = np.zeros(n, bool); reach[s] = True; pi = np.arange(n)
        while True:
            adj = (A > 0) & reach[:, None]                  # rows: reached at the start of the sweep
            new = adj.any(0) & ~reach
            if not new.any(): break
            first = np.argmax(adj, 0)                        # smallest reached i with A[i, j] > 0
            pi[new] = first[new]; reach |= new
        return dict(pi=pi)
    if task == 'mst_kruskal':
        A = inp[0]; n = len(A); pi = np.arange(n); in_mst = np.zeros((n, n), np.int64)
        lx, ly = np.nonzero(np.triu(A > 0, 1)); wts = A[lx, ly]
        def root(u):
            while pi[u] != u: u = pi[u]
            return u
        for ind in np.argsort(wts):
            u, v = lx[ind], ly[ind]; ru, rv = root(u), root(v)
            # (the library also compresses paths while it walks to the roots; that changes no root and no answer)
            if ru < rv: in_mst[u, v] = in_mst[v, u] = 1; pi[ru] = rv
            elif ru > rv: in_mst[u, v] = in_mst[v, u] = 1; pi[rv] = ru
        return dict(in_mst=in_mst)
    if task == 'mst_prim':
        A, s = inp; n = len(A); key = np.zeros(n); mark = np.zeros(n, bool); inq = np.zeros(n, bool); pi = np.arange(n)
        inq[s] = True
        for _ in range(n):
            u = np.argsort(key + (1.0 - inq) * 1e9)[0]
            if not inq[u]: break
            mark[u] = True; inq[u] = False
            upd = (A[u] != 0) & ~mark & (~inq | (A[u] < key))
            pi[upd] = u; key[upd] = A[u, upd]; inq[upd] = True
        return dict(pi=pi)
    if task == 'floyd_warshall':                             # the library's loops, vectorised over (i, j) for each k (exact:
        A = inp[0]; n = len(A); D = A.copy(); Pi = np.repeat(np.arange(n)[:, None], n, 1)   # row k and column k never
        msk = (A + np.eye(n)) != 0                                                            # change in round k)
        for k in range(n):
            cand = msk[:, k][:, None] & msk[k, :][None, :]; nd = D[:, k][:, None] + D[k, :][None, :]
            upd = cand & (~msk | (nd < D))
            D = np.where(upd, nd, D); Pi = np.where(upd, Pi[k, :][None, :], Pi); msk = msk | cand
        return dict(Pi=Pi)
    if task == 'bellman_ford':
        A, s = inp; n = len(A); d = np.zeros(n); pi = np.arange(n); msk = np.zeros(n, bool); msk[s] = True
        while True:
            prev_d, prev_msk = d.copy(), msk.copy()
            for u in np.nonzero(prev_msk)[0]:                 # u in index order, v vectorised (each v is independent)
                nb = A[u] != 0
                upd = nb & (~msk | (prev_d[u] + A[u] < d))
                d[upd] = prev_d[u] + A[u, upd]; pi[upd] = u; msk |= nb
            if (d == prev_d).all():
                break
        return dict(pi=pi)
    if task == 'dijkstra':
        A, s = inp; n = len(A); d = np.zeros(n); mark = np.zeros(n, bool); inq = np.zeros(n, bool); pi = np.arange(n)
        inq[s] = True
        for _ in range(n):
            u = np.argsort(d + (1.0 - inq) * 1e9)[0]
            if not inq[u]: break
            mark[u] = True; inq[u] = False
            upd = (A[u] != 0) & ~mark & (~inq | (d[u] + A[u] < d))
            pi[upd] = u; d[upd] = d[u] + A[u, upd]; inq[upd] = True
        return dict(pi=pi)
    if task == 'dag_shortest_paths':
        A, s = inp; n = len(A); nb = [np.nonzero(A[u] != 0)[0] for u in range(n)]
        color = np.zeros(n, np.int8); fin = []; st = [s]; color[s] = 1; ptr = np.zeros(n, np.int64)
        while st:                                             # the library's DFS from s, neighbours by index
            u = st[-1]; L = nb[u]; i = ptr[u]
            while i < len(L) and color[L[i]]: i += 1
            ptr[u] = i
            if i < len(L): color[L[i]] = 1; st.append(L[i])
            else: color[u] = 2; fin.append(u); st.pop()
        order = fin[::-1]; assert order[0] == s
        d = np.zeros(n); mark = np.zeros(n, bool); pi = np.arange(n)
        for i in order[:-1]:                                  # the library stops before the last node of the order
            mark[i] = True
            upd = (A[i] != 0) & (~mark | (d[i] + A[i] < d))
            d[upd] = d[i] + A[i, upd]; pi[upd] = i; mark |= (A[i] != 0)
        return dict(pi=pi)
    if task in ('dfs', 'topological_sort', 'strongly_connected_components'):
        return dfs_family(task, inp[0])
    raise ValueError(task)


def dfs_family(task, A):
    """dfs (pi), topological_sort (topo, topo_head) and Kosaraju's SCC (scc_id), the library's loops."""
    n = len(A); nbrs = [np.nonzero(A[u] != 0)[0] for u in range(n)]
    def visit(nb, order):
        """Iterative DFS over roots in `order`, neighbours by index; returns pi, finish order, tree roots."""
        color = np.zeros(n, np.int8); pi = np.arange(n); fin = []; root = np.arange(n); ptr = np.zeros(n, np.int64)
        for s in order:
            if color[s]: continue
            color[s] = 1; st = [s]; root[s] = s
            while st:
                u = st[-1]; L = nb[u]; i = ptr[u]
                while i < len(L) and color[L[i]]: i += 1
                ptr[u] = i
                if i < len(L):
                    v = L[i]; color[v] = 1; pi[v] = u; root[v] = s; st.append(v)
                else:
                    color[u] = 2; fin.append(u); st.pop()
        return pi, fin, root
    pi, fin, _ = visit(nbrs, range(n))
    if task == 'dfs':
        return dict(pi=pi)
    if task == 'topological_sort':
        topo = np.arange(n)
        for a, b in zip(fin[1:], fin[:-1]):
            topo[a] = b
        return dict(topo=topo, topo_head=fin[-1])
    nbt = [np.nonzero(A[:, u] != 0)[0] for u in range(n)]
    _, _, root = visit(nbt, fin[::-1])
    return dict(scc_id=root)


# ---------------- packing ----------------
def pack(task, inps, labs):
    N = len(inps)
    if task in GRAPH:
        n = len(inps[0][0]); out = dict(n=np.full(N, n, np.int16), pos=np.tile((np.arange(n) / n).astype(np.float32), (N, 1)),
                                        A=np.stack([i[0] for i in inps]).astype(np.float32))
        out['adj'] = ((out['A'] + np.eye(n, dtype=np.float32)) != 0).astype(np.uint8)
        if len(inps[0]) > 1: out['s'] = np.array([i[1] for i in inps], np.int16)
    elif task in ('naive_string_matcher', 'kmp_matcher', 'lcs_length'):
        a, b = inps[0]; n1, n2 = len(a), len(b); n = n1 + n2
        out = dict(n=np.full(N, n, np.int16), x_string=np.tile(np.r_[np.zeros(n1), np.ones(n2)].astype(np.uint8), (N, 1)),
                   x_pos=np.tile(np.r_[np.arange(n1) / n1, np.arange(n2) / n2], (N, 1)),
                   x_key=np.stack([np.eye(4, dtype=np.uint8)[np.r_[i[0], i[1]]] for i in inps]))
    else:
        n = len(inps[0][-1]); out = dict(n=np.full(N, n, np.int16), x_pos=np.tile(np.arange(n) / n, (N, 1)))
        names = {'binary_search': ['target', 'key'], 'activity_selector': ['s', 'f'], 'task_scheduling': ['d', 'w']}.get(task, ['key'])
        for j, nm in enumerate(names):
            v = np.array([i[j] for i in inps], np.float64); out['x_' + nm] = v
    for k in labs[0]:
        out[k] = np.array([l[k] for l in labs]).astype(np.int16)
    return out


def make(task, n, num, seed, p=None):
    rng = np.random.RandomState(seed); inps = [sample_inputs(task, rng, n, p) for _ in range(num)]
    return pack(task, inps, [label(task, i) for i in inps])


def inputs_from_split(task, d, i):
    """The input of sample i of an official split file, in the order the label functions take it."""
    n = int(d['n'][i])
    if task in GRAPH:
        A = d['A'][i, :n, :n].astype(np.float64)
        return [A, int(d['s'][i])] if 's' in d else [A]
    if task in ('lcs_length', 'naive_string_matcher', 'kmp_matcher'):
        st = d['x_string'][i, :n]; key = d['x_key'][i, :n].argmax(-1)
        return [key[st == 0], key[st == 1]]
    if task == 'binary_search':
        return [float(d['x_target'][i]), d['x_key'][i, :n]]
    if task == 'activity_selector':
        return [d['x_s'][i, :n], d['x_f'][i, :n]]
    if task == 'task_scheduling':
        return [d['x_d'][i, :n].astype(np.int64), d['x_w'][i, :n]]
    return [d['x_key'][i, :n]]


def verify(tasks):
    """(1) seed 3 at length 64 reproduces the official test inputs and labels; (2) the label functions give the stored
    labels on every official split file (train_L4..L16, val, test)."""
    rep = {}
    for task in tasks:
        off = dict(np.load(DATA / task / 'test.npz')); num = len(off['n'])
        mine = make(task, 80 if 'matcher' in task else 64, num, 3)      # the matchers' test is 5/4 of 64 items
        same = {k: bool(np.array_equal(mine[k], off[k]) if mine[k].dtype.kind != 'f' else np.allclose(mine[k], off[k], atol=1e-6)) for k in off}
        rel = {}
        for f in sorted((DATA / task).glob('*.npz')):
            d = dict(np.load(f)); outs = [k for k in label(task, inputs_from_split(task, d, 0))]
            bad = tot = 0
            for i in range(len(d['n'])):
                lab = label(task, inputs_from_split(task, d, i)); n = int(d['n'][i])
                for k in outs:
                    t = d[k][i]; q = np.asarray(lab[k])
                    if t.ndim == 0: bad += int(q != t); tot += 1
                    elif t.ndim == 1: bad += int((q[:n] != t[:n]).sum()); tot += n
                    else:
                        m = t[:n, :n] != -1; bad += int((q[:n, :n][m] != t[:n, :n][m]).sum()); tot += int(m.sum())
            rel[f.stem] = [bad, tot]
        rep[task] = dict(test_reproduced=same, relabel_wrong_of=rel)
        print(task, json.dumps(rep[task]), flush=True)
    return rep


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        tasks = sys.argv[2:] or list(SORTS) + ['minimum', 'binary_search', 'quickselect', 'activity_selector', 'task_scheduling',
                                              'lcs_length', 'bfs', 'mst_kruskal', 'mst_prim', 'dfs', 'topological_sort',
                                              'strongly_connected_components', 'bellman_ford', 'dijkstra', 'dag_shortest_paths',
                                              'floyd_warshall', 'naive_string_matcher', 'kmp_matcher']
        name = 'gen_verify.json' if not sys.argv[2:] else 'gen_verify_' + '_'.join(sys.argv[2:]) + '.json'
        json.dump(verify(tasks), open(Path(__file__).parent / name, 'w'), indent=1)
