"""The role-tied check, fitted at the answer (CRAM's recipe on our kind of state), and what is kept from it.

Pair types. A cell of a grid of side n = b*b has coordinates (r, c). Eight coordinate functions are compared:
    r, c, r//b, c//b, r%b, c%b, r-c, r+c.
The *type* of a pair of distinct cells is the 8-bit pattern of which of them agree. Types are disjoint and say nothing
about where a cell is or how big the board is, so a cost on a type means the same thing on every board size.

The check. E(x) = sum over unordered pairs (i, j) of w[type(i, j)] * [x_i == x_j]. Values are compared only by "same
value" (CRAM imposes the same digit symmetry by relabelling). Costs are kept >= 0: a check charges for violations
and never pays for agreement.

Fit (CRAM's recipe): pseudolikelihood at solved boards (each cell predicted from the other 80 cells of its solved board),
full batch, gradient steps from zero with an L1 soft-threshold after each step, projected to w >= 0. No search, no loop
through a solver. Wall-clock is measured from data in memory to the inference-ready check (weights, kept relations,
count conditions, stop threshold), CRAM's convention.

What is kept, generically:
1. Kept pair types: w > 0.
2. Kept relations: the maximal relations of the form "these coordinate functions agree" (single functions and pairs
   of functions; each is an equivalence relation, its classes are units) whose pairs all have kept types. They are a
   size-free description of the kept types.
3. Count conditions on each kept unit and each value v: the count of cells holding v compared with 0 and 1 only
   ("there is one" n_v >= 1, "at most one" n_v <= 1), kept when it held in every unit of every training answer.
   Constants are 0 and 1 only, so no condition can mention the size of a unit.
4. Stop threshold tau = highest energy of a training answer + 1e-3 (CRAM's rule).
"""
import itertools
import time

import numpy as np

FNAMES = ('r', 'c', 'r//b', 'c//b', 'r%b', 'c%b', 'r-c', 'r+c')
BASE_NAMES = FNAMES
EXTRA_NAMES = ('r%2', 'c%2', '(r+c)%2', 'r//2', 'c//2', '(r+c)%3', '(r-c)%3', '(2r+c)%n', '(r+2c)%n', 'rc%n',
               '(r//b+c//b)%b', '(r%b+c%b)%b')
EXTRA = False   # ablation A1: widen the vocabulary by the 12 distractor functions
VOCAB_B = 5     # board on which the vocabulary is built (3 = the training board, used with EXTRA)


def set_extra(on):
    """Ablation A1: switch the 12 distractor functions on or off (resets the cached vocabulary)."""
    global EXTRA, FNAMES, VOCAB_B, _VOCAB
    EXTRA = on
    FNAMES = BASE_NAMES + (EXTRA_NAMES if on else ())
    VOCAB_B = 3 if on else 5
    _VOCAB = None


def coord_funcs(b):
    n = b * b
    r, c = np.divmod(np.arange(n * n), n)
    f = [r, c, r // b, c // b, r % b, c % b, r - c, r + c]
    if EXTRA:
        f += [r % 2, c % 2, (r + c) % 2, r // 2, c // 2, (r + c) % 3, (r - c) % 3, (2 * r + c) % n, (r + 2 * c) % n,
              (r * c) % n, (r // b + c // b) % b, (r % b + c % b) % b]
    return np.stack(f, 1)  # (cells, functions)


def pair_types(b):
    """(cells, cells) int: the 8-bit agreement pattern of each pair; -1 on the diagonal."""
    f = coord_funcs(b)
    eq = (f[:, None, :] == f[None, :, :])
    t = (eq * (np.int64(1) << np.arange(f.shape[1], dtype=np.int64))).sum(-1)
    np.fill_diagonal(t, -1)
    return t


def fit(sol, b=3, steps=32, lr=None, l1=None, threads_note='', held_filter=False):
    """sol: (N, cells) values 1..V of solved boards. Returns the check dict and the fit time in seconds."""
    t0 = time.perf_counter()
    N, M = sol.shape
    V = b * b
    T = pair_types(b)
    types, inv = np.unique(T[T >= 0], return_inverse=True)
    tid = -np.ones_like(T)
    tid[T >= 0] = inv
    P = len(types)
    onehot = np.zeros((N, M, V), np.float32)
    onehot[np.arange(N)[:, None], np.arange(M)[None, :], sol - 1] = 1.0
    Y = onehot.transpose(1, 0, 2).reshape(M, N * V)  # (cells, N*V)
    same = (Y @ Y.T)  # (cells, cells): number of boards where x_i == x_j
    # pairs per type, for normalising the gradient per type (a preconditioner, as CRAM uses for N-Queens offsets)
    mask_t = np.zeros((P, M, M), np.float32)
    for k in range(P):
        mask_t[k] = (tid == k)
    npairs = mask_t.sum((1, 2))
    w = np.zeros(P, np.float32)
    lr = 4.0 if lr is None else lr
    l1 = 1e-3 if l1 is None else l1
    for _ in range(steps):
        W = np.zeros((M, M), np.float32)
        valid = tid >= 0
        W[valid] = w[tid[valid]]
        logits = -(W @ Y).reshape(M, N, V)  # cost of each value at each cell given the others
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(-1, keepdims=True)
        Pm = p.reshape(M, N * V)
        # d(mean NLL)/dW_ij = mean over boards of ([x_i == x_j] - p_i(x_j)); grouped by type
        G = (same - Pm @ Y.T) / N
        g = np.tensordot(mask_t, G, axes=([1, 2], [0, 1])) / npairs
        w = np.maximum(w - lr * g - lr * l1, 0.0)  # step, L1 soft-threshold, w >= 0
    if held_filter:
        # A1b: a cost is kept only on a type that no training answer violates (no two cells of that type hold the
        # same value on any training board): "keep what held on every training answer", as for the count conditions
        viol = np.zeros(P)
        np.add.at(viol, tid[tid >= 0], same[tid >= 0])
        w = np.where(viol > 0, 0.0, w).astype(np.float32)
    W = np.zeros((M, M), np.float32)
    W[tid >= 0] = w[tid[tid >= 0]]
    kept_types = types[w > 0]
    rels = kept_relations(set(int(t) for t in kept_types))
    conds = count_conditions(sol, b, rels)
    E = energy_batch(sol, W)
    tau = float(E.max()) + 1e-3
    dt = time.perf_counter() - t0
    return dict(b=b, types=types, w=w, kept_types=kept_types, relations=rels, conditions=conds, tau=tau,
                wmin=float(w[w > 0].min()) if (w > 0).any() else 0.0), dt


_VOCAB = None


def vocabulary():
    """The relation vocabulary, built once from the coordinate functions alone (no data): for each candidate relation
    (1 or 2 agreeing functions), the pair types it contains, its partition and its pairs on a large board (b = 5, so
    that no type is missing because a small board cannot realise it). Like building the model class, it is not part
    of a fit."""
    global _VOCAB
    if _VOCAB is None:
        Tbig = pair_types(VOCAB_B)
        realised = set(np.unique(Tbig[Tbig >= 0]).tolist())
        f5 = coord_funcs(VOCAB_B)
        nf = len(FNAMES)
        cands = [(i,) for i in range(nf)] + list(itertools.combinations(range(nf), 2))
        voc = []
        for fs in cands:
            need = frozenset(t for t in realised if all((t >> f) & 1 for f in fs))
            k = f5[:, list(fs)]
            _, first, cls = np.unique(k, axis=0, return_index=True, return_inverse=True)
            # canonical labels (classes numbered by first cell), so that equal partitions have equal keys
            rank = np.argsort(np.argsort(first))
            voc.append((fs, need, tuple(rank[cls.ravel()].tolist()), (k[:, None, :] == k[None, :, :]).all(-1)))
        _VOCAB = voc
    return _VOCAB


def kept_relations(kept):
    """Maximal 'these functions agree' relations whose pair types are all kept (every type the relation can contain,
    on boards of any size). One relation per distinct partition (r//b with r%b is the partition of r), fewest
    functions first; a relation finer than another kept one is dropped (its pairs are already covered)."""
    part = {}
    for fs, need, key, pairs in vocabulary():
        if need and need <= kept and key not in part:
            part[key] = (fs, pairs)
    uniq = list(part.values())
    return [fs for fs, pf in uniq if not any(o is not pf and (pf <= o).all() for _, o in uniq)]


def units(b, rels):
    """List of (relation index, array of cells) for every class of every kept relation on a board of block size b."""
    f = coord_funcs(b)
    out = []
    for k, fs in enumerate(rels):
        keys = f[:, list(fs)]
        _, cls = np.unique(keys, axis=0, return_inverse=True)
        for c in range(cls.max() + 1):
            out.append((k, np.where(cls.ravel() == c)[0]))
    return out


def count_conditions(sol, b, rels):
    """For each kept relation: (ge1, le1) for 'n_v >= 1' and 'n_v <= 1', tied over values (value symmetry), kept when
    they held in every unit of every training answer."""
    V = b * b
    U = units(b, rels)
    conds = []
    for k in range(len(rels)):
        ge, le = True, True
        for kk, cells in U:
            if kk != k:
                continue
            x = sol[:, cells]
            cnt = np.stack([(x == v).sum(1) for v in range(1, V + 1)], 1)
            ge &= bool((cnt >= 1).all())
            le &= bool((cnt <= 1).all())
        conds.append((ge, le))
    return conds


def energy_batch(sol, W):
    """Energy of each board: sum over unordered pairs of W_ij [x_i == x_j]."""
    eq = (sol[:, :, None] == sol[:, None, :]).astype(np.float32)
    return 0.5 * (eq * W[None]).sum((1, 2))


def weights_for(check, b):
    """The check's cost matrix on a board of block size b (any size): w of each pair's type, 0 for types not fitted."""
    T = pair_types(b)
    wt = dict(zip(check['types'].tolist(), check['w'].tolist()))
    W = np.zeros(T.shape, np.float32)
    for t, v in wt.items():
        W[T == t] = v
    return W


def describe(check):
    lines = []
    for t, v in zip(check['types'], check['w']):
        if v > 0:
            agree = [FNAMES[f] for f in range(len(FNAMES)) if (t >> f) & 1]
            lines.append(f'type agree={{{",".join(agree)}}} w={v:.4f}')
    return lines
