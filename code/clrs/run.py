"""Fit the cold check at the answer on CLRS-30 training splits (sizes 4 to 16), run it cold, score it.

python run.py <task> [--per-size K] [--big 256:32,1024:8] [--seed S]

Reports: fit wall clock (state building + features + gate + sparse logistic, one process), the rule, loops of the
cold run, the official 64-node test with the library's scorer, and our generator's 256- and 1,024-node tests.
"""
import argparse, json, os, time, sys
from pathlib import Path
import numpy as np

import cold
import complete
from tasks import state, score_split, OUTPUTS, FLOYDNET
import gen

HERE = Path(__file__).parent
DATA = gen.DATA
SIZES = (4, 7, 11, 13, 16)
KIND = {**{t: 'pointer' for t in gen.SORTS}, 'minimum': 'one', 'binary_search': 'one', 'quickselect': 'one',
        'activity_selector': 'mask', 'task_scheduling': 'mask', 'bfs': 'pointer_loop', 'mst_prim': 'pointer_loop',
        'mst_kruskal': 'emask', 'bellman_ford': 'pointer_loop', 'dijkstra': 'pointer_loop', 'dag_shortest_paths': 'pointer_loop',
        'floyd_warshall': 'pointer_loop', 'lcs_length': 'ecat_loop',
        'naive_string_matcher': 'mask_one', 'kmp_matcher': 'mask_one'}
MATCHERS = ('naive_string_matcher', 'kmp_matcher')   # the library trains them at the single length 20
DIRECTED = ('dag_shortest_paths', 'dfs', 'topological_sort', 'strongly_connected_components')
# Floyd-Warshall: Pi[i, j], the node before j on the shortest path from i, is n single-source answers; each source's
# row is one state (the source marked), checked by the same kind of rule as Bellman-Ford. The library writes "no path"
# as i itself; our "no pointer" is written so. Its diagonal (a self-loop beaten by a round trip) is predicted as i.
PER_SOURCE = {'floyd_warshall'}
# Completion per task, from the shared library (complete.py). 'depth': number of ancestors along the answer's pointer
# (roots that count: the source); 'bottleneck': closure of the edge weights in (min, max), an input relation.
# 'dist': the sum of the weights along the answer's pointer (a weighted depth), and with it the pair number
# S(u, v) = dist u + w(u, v) (hand-supplied addition of a node number and an edge number: class B+).
COMPLETION = {'bfs': ('depth',), 'mst_prim': ('bottleneck', 'depth'), 'mst_kruskal': ('bottleneck',),
              'bellman_ford': ('dist',), 'dijkstra': ('dist',), 'dag_shortest_paths': ('dist',), 'floyd_warshall': ('dist',)}
CONJ = {'bfs', 'mst_prim', 'mst_kruskal', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall'}   # legs may and an edge test with a comparison
TRI = {'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall'}   # pair numbers compared through k


def load(task, split):
    f = DATA / task / f'{split}.npz'
    if not f.exists():                                   # training splits made by make_train.py (the library's protocol)
        f = HERE / 'data' / task / f'{split}.npz'
    return dict(np.load(f))


def out_name(task):
    return OUTPUTS[task][0][0]


def source_states(task, d, i):
    """Floyd-Warshall: one state per source, with that source's row of Pi as a pointer (none = itself)."""
    n = int(d['n'][i]); out = []
    reach = complete.reach(d['A'][i, :n, :n] != 0)
    for src in range(n):
        st = state(task, d, i); st.bits['source'] = np.arange(n) == src
        y = np.asarray(d['Pi'][i, src, :n]).copy() if 'Pi' in d else None
        if y is not None:
            y = np.where(reach[src], y, np.arange(n)); st.src_label = int(y[src]); y[src] = src
        out.append((st, y))
    return out


def train_states(task, per_size):
    S, Y = [], []
    sizes = (20,) if task in MATCHERS else SIZES
    for L in sizes:
        d = load(task, f'train_L{L}')
        if task in MATCHERS:
            per_size = per_size * len(SIZES)                     # the same 100 inputs, all at length 20
        if task in PER_SOURCE:
            for i in range(min(max(1, per_size // 5), len(d['n']))):
                for st, y in source_states(task, d, i):
                    S.append(st); Y.append(y)
            continue
        for i in range(min(per_size, len(d['n']))):
            S.append(state(task, d, i)); Y.append(d[out_name(task)][i])
    return S, Y


def line_prev(st):
    """The input's line order: prev[i] = the item right before i in its own string (-1 if first)."""
    grp = st.bits.get('string', np.zeros(st.n, bool)).astype(int); pos = st.reg['pos']; prev = np.full(st.n, -1)
    for g in np.unique(grp):
        idx = np.nonzero(grp == g)[0]; idx = idx[np.argsort(pos[idx])]
        prev[idx[1:]] = idx[:-1]
    return prev


def lcs_complete(st, b):
    """LCS: c = the number of match arrows along the answer's arrows from each cell (pointer doubling over the cells;
    off the grid counts 0; a cell with no arrow, or whose arrows reach one, is "not yet"); then the shift: every cell
    reads Uc = c of the cell above and Lc = c of the cell to its left (off the grid: 0). Its own c is not shown."""
    n = st.n; prev = st.prev
    if not hasattr(st, 'grid'):                                     # the cells in line order: x items by rows, y by columns
        x = np.nonzero(~st.bits['string'])[0]; y = np.nonzero(st.bits['string'])[0]
        x = x[np.argsort(st.reg['pos'][x])]; y = y[np.argsort(st.reg['pos'][y])]; st.grid = (x, y)
    x, y = st.grid; n1, n2 = len(x), len(y); B = b[np.ix_(x, y)]
    N = n1 * n2; ROOT = N; I, J = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    di = np.where(B == 2, 0, 1); dj = np.where(B == 1, 0, 1)                  # 0: up-left, 1: up, 2: left
    pi_, pj = I - di, J - dj
    nxt = np.where((pi_ < 0) | (pj < 0), ROOT, pi_.clip(0) * n2 + pj.clip(0)).ravel()
    none = (B < 0).ravel(); f = np.arange(N); nxt = np.where(none, f, nxt)
    nxt = np.r_[nxt, ROOT]; w = np.r_[(B == 0).ravel().astype(float), 0.0]; ok = np.r_[~none, True]
    tot, p = w.copy(), nxt.copy()
    for _ in range(int(np.ceil(np.log2(N + 1))) + 1):
        tot = tot + np.where(p == ROOT, 0.0, tot[p]); ok = ok & ok[p]; p = np.where(p == ROOT, ROOT, p[p])
    c = np.where(ok[:N] & (p[:N] == ROOT), tot[:N], np.inf).reshape(n1, n2)
    U = np.vstack([np.zeros((1, n2)), c[:-1]]); L = np.hstack([np.zeros((n1, 1)), c[:, :-1]])
    Uc = np.full((n, n), np.nan); Lc = np.full((n, n), np.nan)
    Uc[np.ix_(x, y)] = U; Lc[np.ix_(x, y)] = L
    st.pnum['Uc'], st.pnum['Lc'] = Uc, Lc; st.pgroups = [['Uc', 'Lc']]; st._c.clear()
    return st


def complete_static(task, st):
    """Completions of input relations (computed once per input)."""
    if task in ('lcs_length',) + MATCHERS:
        st.prev = line_prev(st)
    if task in MATCHERS:                     # closure of an input relation along the two lines: the text from t matches
        n = st.n; pat = st.bits['string']; nxt = np.full(n, -1); has = st.prev >= 0   # the pattern from p to its end
        nxt[st.prev[has]] = np.nonzero(has)[0]
        same = st.rel['same_key'] & (~pat)[:, None] & pat[None, :]
        run_ = np.zeros((n, n), bool); last = pat & (nxt < 0)
        order = sorted(np.nonzero(pat)[0], key=lambda p: -st.reg['pos'][p])          # from the pattern's end backwards
        tn = np.where(nxt >= 0, nxt, 0); tok = nxt >= 0
        for p in order:
            if last[p]:
                run_[:, p] = same[:, p]
            else:
                run_[:, p] = same[:, p] & tok & run_[tn, nxt[p]]
        st.rel['run'] = run_; st.bits['first'] = st.prev < 0
        st._c.clear()
    if 'bottleneck' in COMPLETION.get(task, ()):
        st.pnum['B'] = complete.bottleneck(st.pnum['w']); st.pgroups = [['w', 'B']]
    return st


def complete_answer(task, st, pi):
    """Completions of the answer (recomputed from the current answer in the cold run)."""
    root_ok = st.bits['source'] if 'source' in st.bits else None
    for c in COMPLETION.get(task, ()):
        if c not in ('depth', 'dist'):
            continue
        if c == 'depth':
            st.reg['depth'] = complete.depth(np.asarray(pi), root_ok)
        else:
            st.reg['dist'] = complete.dist(np.asarray(pi), st.pnum['w'], root_ok)
            st.pnum['S'] = st.reg['dist'][:, None] + st.pnum['w']; st.pgroups = [['w', 'S']]
        if [c] not in st.groups:
            st.groups.append([c])
        st.bits['has_' + c] = np.isfinite(st.reg[c]); st.answer.add('has_' + c); st.hidden.add(c)
    st._c.clear()
    return st


def with_answer(task, st, y):
    """The state with the answer written in (the fit sees the answer at every cell but the target's own)."""
    k = KIND[task]
    complete_static(task, st)
    if k == 'mask':
        st.bits['sel'] = np.asarray(y[:st.n]) > 0; st.answer.add('sel'); st._c.clear()
    if k == 'mask_one':
        st.bits['sel'] = np.arange(st.n) == int(y); st.answer.add('sel'); st._c.clear()
    if k == 'pointer_loop':
        complete_answer(task, st, np.asarray(y[:st.n]))
    if k == 'ecat_loop':
        lcs_complete(st, np.asarray(y[:st.n, :st.n]))
    return st


def fit_task(task, per_size, lam):
    t0 = time.perf_counter()
    S, Y = train_states(task, per_size)
    S = [with_answer(task, s, y) for s, y in zip(S, Y)]
    tgt = {id(s): y for s, y in zip(S, Y)}
    k = KIND[task]; rules = {}; cj = task in CONJ; tr = task in TRI
    if k == 'pointer_loop':
        off_rows = lambda s: ~np.eye(s.n, dtype=bool)
        off_t = lambda s: (np.asarray(tgt[id(s)][:s.n])[None, :] == np.arange(s.n)[:, None])
        rules['off'], _ = cold.fit(S, off_t, 'pair', off_rows, lam=lam, conj=cj, tri=tr)
        rules['diag'], _ = cold.fit(S, lambda s: np.asarray(tgt[id(s)][:s.n]) == np.arange(s.n), 'item', lambda s: np.ones(s.n, bool), lam=lam, conj=cj)
        if task in PER_SOURCE:      # the source's own cell (Floyd-Warshall's diagonal): one more rule on the pairs (u, source)
            src_rows = lambda s: (~np.eye(s.n, dtype=bool)) & s.bits['source'][None, :]
            src_t = lambda s: (np.arange(s.n)[:, None] == s.src_label) & s.bits['source'][None, :]
            rules['srcptr'], _ = cold.fit(S, src_t, 'pair', src_rows, lam=lam, conj=cj, tri=tr)
    elif k == 'ecat_loop':                      # one rule per arrow class, on the cells
        rows = lambda s: (~s.bits['string'])[:, None] & s.bits['string'][None, :]
        for c in (0, 1, 2):
            rules[f'cat{c}'], _ = cold.fit(S, lambda s, c=c: np.asarray(tgt[id(s)][:s.n, :s.n]) == c, 'pair', rows, lam=lam)
    elif k == 'emask':
        rows = lambda s: ~np.eye(s.n, dtype=bool)
        rules['emask'], _ = cold.fit(S, lambda s: np.asarray(tgt[id(s)][:s.n, :s.n]) > 0, 'pair', rows, lam=lam, conj=cj)
    elif k == 'pointer':
        off_rows = lambda s: ~np.eye(s.n, dtype=bool)
        off_t = lambda s: (np.asarray(tgt[id(s)][:s.n])[None, :] == np.arange(s.n)[:, None])
        rules['off'], _ = cold.fit(S, off_t, 'pair', off_rows, lam=lam)
        rules['diag'], _ = cold.fit(S, lambda s: np.asarray(tgt[id(s)][:s.n]) == np.arange(s.n), 'item', lambda s: np.ones(s.n, bool), lam=lam)
    elif k == 'one':
        rules['one'], _ = cold.fit(S, lambda s: np.arange(s.n) == int(tgt[id(s)]), 'item', lambda s: np.ones(s.n, bool), lam=lam)
    elif k == 'mask_one':
        rules['mask'], rules['levels'] = fit_levels(S, lambda s: np.arange(s.n) == int(tgt[id(s)]), 'item', lambda s: np.ones(s.n, bool), lam)
    elif k == 'mask':
        rules['mask'], rules['levels'] = fit_levels(S, lambda s: np.asarray(tgt[id(s)][:s.n]) > 0, 'item', lambda s: np.ones(s.n, bool), lam)
    return rules, time.perf_counter() - t0, len(S)


def fit_levels(S, target, kind, rows, lam):
    """The answer is read only downward along one order (a level). Every register and direction is a candidate;
    the fit at the answer picks: a level whose rule is exact on the training answers, the sparsest such."""
    cands = []
    for r in sorted(S[0].reg):
        for op in ('lt', 'gt'):
            rule, _ = cold.fit(S, target, kind, rows, lam=lam, level=(r, op))
            cands.append(((r, op), rule))
    tried = {f'{r}_{op}': dict(train_wrong=ru.info['train_wrong_rule'], gate=ru.info['gate'], score=ru.info['score_features'])
             for (r, op), ru in cands}
    best = min(cands, key=lambda c: (c[1].info['train_wrong_rule'], c[1].info['gate'] + c[1].info['score_features']))
    best[1].info['level'] = best[0]
    return best[1], tried


def predict(task, rules, st, start=0, max_loops=None):
    """Run the fitted check cold from nothing. Returns prediction (unpadded) and loops."""
    k = KIND[task]
    complete_static(task, st)
    if k == 'pointer':
        return cold.read_pointer(st, rules['off'], rules['diag'])[0], 1
    if k == 'emask':
        m = rules['emask'].decide(st).copy(); np.fill_diagonal(m, False)
        return m.astype(int), 1
    if k == 'ecat_loop':                                            # from nothing: no arrow anywhere
        cell = (~st.bits['string'])[:, None] & st.bits['string'][None, :]
        b = np.full((st.n, st.n), -1)
        for loop in range(1, (max_loops or 2 * st.n + 8) + 1):
            lcs_complete(st, b)
            ok = np.stack([rules[f'cat{c}'].decide(st) for c in (0, 1, 2)])
            sc = np.stack([rules[f'cat{c}'].score(st) for c in (0, 1, 2)])
            new = np.where(ok.any(0), np.where(ok, sc, -np.inf).argmax(0), -1); new = np.where(cell, new, -1)
            if (new == b).all():
                return b, loop
            b = new
        return b, -loop
    if k == 'pointer_loop':                                         # from nothing: no pointer anywhere
        pi = np.full(st.n, -1)
        for loop in range(1, (max_loops or 2 * st.n + 8) + 1):
            complete_answer(task, st, pi)
            new = cold.read_pointer(st, rules['off'], rules['diag'], none=True)[0]
            if (new == pi).all():
                pi = np.where(pi < 0, np.arange(st.n), pi)               # no pointer = the library's "none" (itself)
                if 'srcptr' in rules:                                     # then the source's own cell, read once at rest
                    src = int(np.argmax(st.bits['source'])); r = rules['srcptr']
                    ok, sc = r.decide(st)[:, src].copy(), r.score(st)[:, src].copy(); ok[src] = False
                    if ok.any():
                        pi[src] = int(np.where(ok, sc, -np.inf).argmax())
                return pi, loop
            pi = new
        return np.where(pi < 0, np.arange(st.n), pi), -loop
    if k == 'one':
        return cold.read_one(st, rules['one'])[0], 1
    sel = np.full(st.n, bool(start)); st.answer.add('sel')
    one = lambda m: int(np.nonzero(m)[0][np.argmin(st.reg['pos'][m])]) if m.any() else int((~st.bits['string']).sum())
    for loop in range(1, (max_loops or 4 * st.n + 8) + 1):
        st.bits['sel'] = sel; st._c.clear()
        new = rules['mask'].decide(st)
        if (new == sel).all():
            return (one(sel) if k == 'mask_one' else sel.astype(int)), loop
        sel = new
    return (one(sel) if k == 'mask_one' else sel.astype(int)), -loop                                   # negative: did not come to rest


def evaluate(task, rules, d, start=0):
    name = out_name(task); N = len(d['n']); M = d[name].shape[1] if d[name].ndim > 1 else None
    pred = np.full(d[name].shape, -1, np.int64) if M else np.zeros(N, np.int64); loops = []
    t0 = time.perf_counter()
    for i in range(N):
        if task in PER_SOURCE:
            n = int(d['n'][i]); P = np.zeros((n, n), np.int64); ll = []
            for src, (st, _) in enumerate(source_states(task, d, i)):
                p, l = predict(task, rules, st, start); ll.append(l)
                P[src] = np.where(p == np.arange(n), src, p)          # no pointer: the library's "i itself"
            pred[i, :n, :n] = P; loops.append(max(ll, key=abs)); continue
        st = state(task, d, i); p, l = predict(task, rules, st, start); loops.append(l)
        if d[name].ndim == 3: pred[i, :st.n, :st.n] = p
        elif M: pred[i, :st.n] = p
        else: pred[i] = p
    return score_split(task, d, {name: pred}), loops, time.perf_counter() - t0


def describe(rule):
    g = [('' if v else 'not ') + str(f) for f, v in rule.gate]
    return dict(info=rule.info, exact_gate=rule.exact_gate, gate=g[:40], gate_size=len(g),
                score=[(str(f), round(float(w), 3)) for f, w in sorted(zip(rule.feats, rule.w), key=lambda x: -abs(x[1]))[:12]])


def go(task, per_size=20, big='256:32,1024:8', seed=101, lam=1e-3, start=0, log=print):
    rules, fit_s, ntrain = fit_task(task, per_size, lam)
    res = dict(task=task, kind=KIND[task], train_inputs=ntrain, fit_ms=round(1000 * fit_s, 1),
               rules={k: (describe(r) if isinstance(r, cold.Rule) else r) for k, r in rules.items()}, start=start)
    log(json.dumps(dict(task=task, fit_ms=res['fit_ms'], train_inputs=ntrain,
                        **{k: (dict(exact_gate=r.exact_gate, **r.info) if isinstance(r, cold.Rule) else r) for k, r in rules.items()}), default=str))
    sc, loops, ev = evaluate(task, rules, load(task, 'test'), start)
    res['test64'] = dict(score=sc, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2))
    log(f'official test 64: {res["test64"]}')
    if os.environ.get('HELDOUT_VAL') == '1':                         # the official val split (16 nodes), untouched
        sc, loops, ev = evaluate(task, rules, load(task, 'val'), start)
        res['val'] = dict(score=sc, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2))
        log(f'official val: {res["val"]}')
    for spec in big.split(','):
        if not spec:
            continue
        parts = spec.split(':'); n, num = int(parts[0]), int(parts[1])
        # n:num:c = a sparse stress test with about c neighbours per node: edge probability c / n for directed graphs,
        # sqrt(c / n) for undirected ones (the library keeps an undirected edge only if both of its draws are on)
        p = ((float(parts[2]) / n) if task in DIRECTED else np.sqrt(float(parts[2]) / n),) if len(parts) > 2 else None
        d = gen.make(task, n, num, seed + n + (7 if p else 0), p=p)
        sc, loops, ev = evaluate(task, rules, d, start)
        key = f'gen{n}' + (f'_sparse{parts[2]}' if p else '')
        res[key] = dict(score=sc, num=num, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2))
        log(f'generated {key}: {res[key]}')
    res['paper'] = dict(floydnet=FLOYDNET.get(task))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--per-size', type=int, default=20)
    ap.add_argument('--big', default='256:32,1024:8'); ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--lam', type=float, default=1e-3); ap.add_argument('--start', type=int, default=0)
    a = ap.parse_args()
    res = go(a.task, a.per_size, a.big, a.seed, a.lam, a.start, log=lambda m: print(m, flush=True))
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    json.dump(res, open(out / f'{a.task}.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
