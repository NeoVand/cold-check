"""The tasks added after the freeze, with the frozen library (cold.py, complete.py, tasks.py, run.py) used unchanged.
Each task adds one hand-supplied primitive, labelled, and then fits at the answer as before (own conditions kept, the
greedy conjunction, the score only for ties), with a 1-loop read (no answer feedback) unless said.

  graham_scan, jarvis_march   primitive: orientation sign of three points (a ceiling); one witness layer on it gives
                              the input relations noright(i, j) / noleft(i, j): no point strictly right / left of i -> j
  find_maximum_subarray_kadane primitive (a ceiling): prefix sums P, P' (exclusive), prefix minimum m, M = P - m;
                              end is fitted first, then start reads the end as an input bit
  bridges                     primitive, "closure of an input relation": C(u, v) = u reaches v in G without the edge
                              (u, v) (one closure per edge)

python extra.py <task> [--big 256:32,1024:8] [--seed S] [--val] [--out DIR]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np

import cold, run
from tasks import state, score_split, OUTPUTS
import gen_extra

HERE = Path(__file__).parent


# ------------------------------------------------------------------------------------------------- primitives --
def hull_relations(st):
    x, y = st.reg['x'], st.reg['y']; n = st.n
    left = np.zeros((n, n), bool); right = np.zeros((n, n), bool)
    for i in range(n):
        dx, dy = x - x[i], y - y[i]                                     # cross(p_j - p_i, p_k - p_i) for all j, k
        cr = dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :]
        left[i] = (cr > 0).any(1); right[i] = (cr < 0).any(1)
    nr, nl = ~right, ~left; np.fill_diagonal(nr, False); np.fill_diagonal(nl, False)
    st.rel['noright'], st.rel['noleft'] = nr, nl; st._c.clear()
    return st


def kadane_registers(st):
    key = st.reg['key']; order = np.argsort(st.reg['pos']); k = key[order]
    P = np.cumsum(k); Pp = P - k; m = np.minimum.accumulate(Pp); M = P - m
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for name, v in (('P', P), ('Pp', Pp), ('m', m), ('M', M)):
        st.reg[name] = v[inv]
    st.groups = [['pos'], ['key'], ['M'], ['Pp', 'm']]; st._c.clear()
    return st


def deletion_reach(st):
    E = st.rel['edge']; n = st.n; nb = [np.nonzero(E[u])[0] for u in range(n)]
    def reach(u, skip):
        seen = np.zeros(n, bool); seen[u] = True; stack = [u]
        while stack:
            a = stack.pop()
            for b in nb[a]:
                if (a, b) == skip or (b, a) == skip or seen[b]:
                    continue
                seen[b] = True; stack.append(b)
        return seen
    C = np.zeros((n, n), bool)
    base = [reach(u, None) for u in range(n)]
    for u in range(n):
        C[u] = base[u]
    for u, v in zip(*np.nonzero(np.triu(E, 1))):
        r = reach(u, (u, v)); C[u, v] = C[v, u] = r[v]
    np.fill_diagonal(C, False); st.rel['reach_without'] = C; st._c.clear()
    return st


def deletion_components(st):
    """Articulation points: sep(v, a) = some neighbour b of v is not reached from a in G - v (components of G - v
    for every v: one closure per item; then one witness layer)."""
    E = st.rel['edge']; n = st.n; nb = [np.nonzero(E[u])[0] for u in range(n)]; sep = np.zeros((n, n), bool)
    for v in range(n):
        lab = np.full(n, -1); c = 0
        for s0 in range(n):
            if s0 == v or lab[s0] >= 0: continue
            stack = [s0]; lab[s0] = c
            while stack:
                a = stack.pop()
                for b in nb[a]:
                    if b != v and lab[b] < 0: lab[b] = c; stack.append(b)
            c += 1
        nv = nb[v]
        if len(nv):
            comps = set(lab[nv]); 
            if len(comps) > 1:
                for a in range(n):
                    if a != v: sep[v, a] = any(lab[b] != lab[a] for b in nv)
    st.rel['sep'] = sep; st._c.clear()
    return st


def segment_relations(st):
    """Segments: left(i, k) / right(i, k) = k strictly left / right of the line from i through its partner (0-1, 2-3
    in the library's input order); other(i, k) = k is on the other segment."""
    x, y = st.reg['x'], st.reg['y']; n = st.n; part = np.array([1, 0, 3, 2])[:n]
    L = np.zeros((n, n), bool); Rr = np.zeros((n, n), bool)
    for i in range(n):
        j = part[i]; cr = (x[j] - x[i]) * (y - y[i]) - (y[j] - y[i]) * (x - x[i])
        L[i] = cr > 0; Rr[i] = cr < 0
    seg = np.arange(n) // 2; other = seg[:, None] != seg[None, :]
    st.rel['left'], st.rel['right'], st.rel['other'] = L, Rr, other; st._c.clear()
    return st


PRIM = {'articulation_points': deletion_components, 'segments_intersect': segment_relations,
        'graham_scan': hull_relations, 'jarvis_march': hull_relations, 'find_maximum_subarray_kadane': kadane_registers,
        'bridges': deletion_reach}


def load(task, split):
    return run.load(task, split)


def make_state(task, d, i):
    return PRIM[task](state(task, d, i))


# ------------------------------------------------------------------------------------------------ fit and read --
def fit_task(task, per_size=20):
    t0 = time.perf_counter(); S, Y = [], []
    for L in run.SIZES:
        d = load(task, f'train_L{L}')
        for i in range(min(per_size, len(d['n']))):
            S.append(make_state(task, d, i)); Y.append({k: d[k][i] for k, _ in OUTPUTS[task]})
    tg = {id(s): y for s, y in zip(S, Y)}; rules = {}
    allrows = lambda s: np.ones(s.n, bool)
    if task in ('graham_scan', 'jarvis_march'):
        rules['mask'], _ = cold.fit(S, lambda s: np.asarray(tg[id(s)]['in_hull'][:s.n]) > 0, 'item', allrows)
    elif task == 'find_maximum_subarray_kadane':
        rules['end'], _ = cold.fit(S, lambda s: np.arange(s.n) == int(tg[id(s)]['end']), 'item', allrows)
        for s in S:                                                   # start reads the end (the true one in training)
            s.bits['end'] = np.arange(s.n) == int(tg[id(s)]['end']); s._c.clear()
        rules['start'], _ = cold.fit(S, lambda s: np.arange(s.n) == int(tg[id(s)]['start']), 'item', allrows)
    elif task == 'articulation_points':
        rules['mask'], _ = cold.fit(S, lambda s: np.asarray(tg[id(s)]['is_cut'][:s.n]) > 0, 'item', allrows)
    elif task == 'segments_intersect':                                # every item carries the input's one answer
        rules['all'], _ = cold.fit(S, lambda s: np.full(s.n, int(tg[id(s)]['intersect']) > 0), 'item', allrows)
    elif task == 'bridges':
        rows = lambda s: ~np.eye(s.n, dtype=bool)
        rules['emask'], _ = cold.fit(S, lambda s: np.asarray(tg[id(s)]['is_bridge'][:s.n, :s.n]) > 0, 'pair', rows)
    return rules, time.perf_counter() - t0, len(S)


def predict(task, rules, st):
    if task in ('graham_scan', 'jarvis_march'):
        return {'in_hull': rules['mask'].decide(st).astype(int)}
    if task == 'find_maximum_subarray_kadane':
        e = cold.read_one(st, rules['end'])[0]
        st.bits['end'] = np.arange(st.n) == e; st._c.clear()
        return {'end': e, 'start': cold.read_one(st, rules['start'])[0]}
    if task == 'articulation_points':
        return {'is_cut': rules['mask'].decide(st).astype(int)}
    if task == 'segments_intersect':                                  # yes iff every item passes the kept conditions
        return {'intersect': int(rules['all'].passes(st).all())}
    if task == 'bridges':
        m = rules['emask'].decide(st).copy(); np.fill_diagonal(m, False)
        return {'is_bridge': m.astype(int)}


def evaluate(task, rules, d):
    preds = {k: (np.full(d[k].shape, -1, np.int64) if d[k].ndim > 1 else np.zeros(len(d['n']), np.int64)) for k, _ in OUTPUTS[task]}
    t0 = time.perf_counter()
    for i in range(len(d['n'])):
        st = make_state(task, d, i); p = predict(task, rules, st)
        for k, v in p.items():
            if preds[k].ndim == 3: preds[k][i, :st.n, :st.n] = v
            elif preds[k].ndim == 2: preds[k][i, :st.n] = v
            else: preds[k][i] = v
    return score_split(task, d, preds), time.perf_counter() - t0


def go(task, big='256:32,1024:8', seed=101, val=False, per_size=20, log=print):
    rules, fs, nt = fit_task(task, per_size)
    res = dict(task=task, fit_ms=round(1000 * fs, 1), train_inputs=nt,
               rules={k: run.describe(r) for k, r in rules.items()})
    log(json.dumps(dict(task=task, fit_ms=res['fit_ms'], **{k: dict(exact=r.exact_gate, **r.info) for k, r in rules.items()})))
    sc, ev = evaluate(task, rules, load(task, 'test')); res['test64'] = dict(score=sc, loops_max=1, not_at_rest=0, eval_s=round(ev, 2))
    log(f"official test: {res['test64']}")
    if val:
        sc, ev = evaluate(task, rules, load(task, 'val')); res['val'] = dict(score=sc, eval_s=round(ev, 2)); log(f"val: {res['val']}")
    for spec in big.split(','):
        if not spec:
            continue
        n, num = map(int, spec.split(':')[:2])
        d = gen_extra.make(task, n, num, seed + n)
        sc, ev = evaluate(task, rules, d); res[f'gen{n}'] = dict(score=sc, num=num, loops_max=1, not_at_rest=0, eval_s=round(ev, 2))
        log(f"generated {n}: {res[f'gen{n}']}")
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--big', default='256:8')
    ap.add_argument('--seed', type=int, default=101); ap.add_argument('--val', action='store_true'); ap.add_argument('--out', default='results')
    a = ap.parse_args()
    res = go(a.task, a.big, a.seed, a.val, log=lambda m: print(m, flush=True))
    (HERE / a.out).mkdir(exist_ok=True); json.dump(res, open(HERE / a.out / f'{a.task}.json', 'w'), indent=1, default=str)
