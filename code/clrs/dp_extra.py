"""Matrix chain order and optimal BST (registered before it ran): the split DP checked at the completed answer, run cold.
New file; cold.py is used unchanged. Hand-supplied (a ceiling): the products p_{i-1} p_k p_j and the sums w(a, b).

python dp_extra.py <task> [--big 128:4] [--seed S] [--val] [--out DIR]"""
import argparse, json, time
from pathlib import Path
import numpy as np
import cold, run
from tasks import score_split, OUTPUTS
import gen_extra

HERE = Path(__file__).parent
OUT = {'matrix_chain_order': 's', 'optimal_bst': 'root'}


def inputs(task, d, i):
    n = int(d['n'][i])
    if task == 'matrix_chain_order':
        return n, dict(p=d['x_p'][i, :n].astype(np.float64))
    return n, dict(p=d['x_p'][i, :n].astype(np.float64), q=d['x_q'][i, :n].astype(np.float64))


def cost_table(task, n, x, S):
    """C from the current split table S (-1 = no split yet), by the recursion along the splits: "not yet" (inf)
    until the split and both parts are known. Matrix chain: intervals 1 <= a <= b <= n - 1; BST: 0 <= a <= b <= n - 1."""
    C = np.full((n, n), np.inf)
    lo = 1 if task == 'matrix_chain_order' else 0
    if task == 'optimal_bst':
        p, q = x['p'], x['q']; W = np.zeros((n, n))
        for a in range(n):
            W[a, a] = q[a]
            for b in range(a + 1, n): W[a, b] = W[a, b - 1] + p[b - 1] + q[b]
    for a in range(lo, n):
        C[a, a] = 0.0 if task == 'matrix_chain_order' else x['q'][a]
    for l in range(1, n - lo):
        for a in range(lo, n - l):
            b = a + l; k = S[a, b]
            if k < 0: continue
            if task == 'matrix_chain_order':
                C[a, b] = C[a, k] + C[k + 1, b] + x['p'][a - 1] * x['p'][k] * x['p'][b]
            else:
                C[a, b] = C[a, k] + C[k + 1, b] + W[a, b]
    return C


def source_state(task, n, x, C, i):
    """State i: N(k, j) = cost of splitting (i .. j) at k (missing outside i <= k < j); valid(k, j) the same range."""
    k = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    valid = (k >= i) & (k < j) & (j > i)
    kk = np.clip(k + 1, 0, n - 1)
    if task == 'matrix_chain_order':
        Ncost = C[i, k] + C[kk, j] + x['p'][i - 1] * x['p'][k] * x['p'][j]
    else:
        Ncost = C[i, k] + C[kk, j]                       # w(i, j) is the same for every k
    N = np.where(valid, Ncost, np.nan)
    st = cold.State(n, reg={'pos': np.arange(n) / n}, groups=[['pos']], rel={'valid': valid},
                    pnum={'N': N}, pgroups=[['N']], bits={'source': np.arange(n) == i})
    return st


def sources(task, n):
    return range(1, n - 1) if task == 'matrix_chain_order' else range(0, n - 1)


def fit(task, per_size=4):
    t0 = time.perf_counter(); S_, T = [], {}
    for L in run.SIZES:
        d = run.load(task, f'train_L{L}')
        for i in range(min(per_size, len(d['n']))):
            n, x = inputs(task, d, i); Y = d[OUT[task]][i][:n, :n].astype(np.int64)
            C = cost_table(task, n, x, Y)
            for s in sources(task, n):
                st = source_state(task, n, x, C, s); S_.append(st); T[id(st)] = Y[s]
    rows = lambda s: s.rel['valid'] | (~np.eye(s.n, dtype=bool) & s.rel['valid'].any(0)[None, :])
    rows = lambda s: (~np.eye(s.n, dtype=bool)) & s.rel['valid'].any(0)[None, :]      # targets j with a split
    rule, _ = cold.fit(S_, lambda s: np.arange(s.n)[:, None] == T[id(s)][None, :], 'pair', rows, tri=True)
    return rule, time.perf_counter() - t0, len(S_)


def predict(task, rule, n, x, max_loops=None):
    S = np.full((n, n), -1, np.int64); loops = 0
    for loops in range(1, (max_loops or 2 * n + 8) + 1):
        C = cost_table(task, n, x, S); new = np.full((n, n), -1, np.int64)
        for i in sources(task, n):
            st = source_state(task, n, x, C, i); ok = rule.decide(st); sc = rule.score(st)
            has = st.rel['valid'].any(0)
            pick = np.where(ok, sc, -np.inf).argmax(0)
            new[i] = np.where(has & ok.any(0), pick, -1)
        if (new == S).all():
            break
        S = new
    out = np.where(S < 0, 0, S)
    return out, loops if (new == S).all() else -loops


def evaluate(task, rule, d):
    name = OUT[task]; pred = np.full(d[name].shape, -1, np.int64); loops = []
    t0 = time.perf_counter()
    for i in range(len(d['n'])):
        n, x = inputs(task, d, i); p, l = predict(task, rule, n, x); pred[i, :n, :n] = p; loops.append(l)
    return score_split(task, d, {name: pred}), loops, time.perf_counter() - t0


def go(task, big='128:4', seed=101, val=False, per_size=4, log=print):
    rule, fs, nt = fit(task, per_size)
    res = dict(task=task, fit_ms=round(1000 * fs, 1), train_states=nt, rules={'off': run.describe(rule)})
    log(json.dumps(dict(task=task, fit_ms=res['fit_ms'], exact=rule.exact_gate, **rule.info)))
    sc, loops, ev = evaluate(task, rule, run.load(task, 'test'))
    res['test64'] = dict(score=sc, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2)); log(f"official test: {res['test64']}")
    if val:
        sc, loops, ev = evaluate(task, rule, run.load(task, 'val')); res['val'] = dict(score=sc, loops_max=max(loops, key=abs)); log(f"val: {res['val']}")
    for spec in big.split(','):
        if not spec: continue
        n, num = map(int, spec.split(':')[:2])
        d = gen_extra.make(task, n + (1 if task == 'optimal_bst' else 0), num, seed + n)
        sc, loops, ev = evaluate(task, rule, d)
        res[f'gen{n}'] = dict(score=sc, num=num, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2)); log(f"generated {n}: {res[f'gen{n}']}")
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--big', default='128:2')
    ap.add_argument('--seed', type=int, default=101); ap.add_argument('--val', action='store_true'); ap.add_argument('--out', default='results')
    a = ap.parse_args(); res = go(a.task, a.big, a.seed, a.val, log=lambda m: print(m, flush=True))
    (HERE / a.out).mkdir(exist_ok=True); json.dump(res, open(HERE / a.out / f'{a.task}.json', 'w'), indent=1, default=str)
