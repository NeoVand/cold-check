"""DFS fitted at the prefixes of the answer's discovery order, run one node per loop (registered before it ran).
New file; cold.py unchanged. Also topological sort and SCC from the same DFS check (see their registrations).

python dfs2.py dfs [--big 256:8] [--seed S] [--val] [--out DIR]"""
import argparse, json, time
from pathlib import Path
import numpy as np
import cold, run
from tasks import state, score_split, OUTPUTS
import gen

HERE = Path(__file__).parent


def preorder(pi):
    """Discovery order of the answer forest (children and roots by index)."""
    n = len(pi); ch = [[] for _ in range(n)]; roots = [v for v in range(n) if pi[v] == v]
    for v in range(n):
        if pi[v] != v: ch[pi[v]].append(v)
    out = []
    for r in roots:
        st = [r]
        while st:
            u = st.pop(); out.append(u); st.extend(sorted(ch[u], reverse=True))
    return out


def complete(st, pi):
    """Partial answer pi (-1 = not placed): discovery rank d (inf if unplaced), placed, unplaced, open."""
    n = st.n; placed = pi >= 0; d = np.full(n, np.inf)
    if placed.any():
        sub = np.where(placed, pi, -1)
        # preorder of the placed forest
        ch = [[] for _ in range(n)]; roots = [v for v in range(n) if sub[v] == v]
        for v in range(n):
            if sub[v] >= 0 and sub[v] != v: ch[sub[v]].append(v)
        t = 0
        for r in roots:
            s_ = [r]
            while s_:
                u = s_.pop(); d[u] = t; t += 1; s_.extend(sorted(ch[u], reverse=True))
    E = st.rel['edge']
    st.reg['d'] = d
    if ['d'] not in st.groups: st.groups.append(['d'])
    st.bits['placed'] = placed; st.bits['unplaced'] = ~placed
    st.bits['open'] = placed & (E & (~placed)[None, :]).any(1)
    st.rel['all'] = ~np.eye(n, dtype=bool)
    st._c.clear(); return st


def train_states(task_data='dfs', per_size=20):
    S, T = [], {}
    for L in run.SIZES:
        d = run.load(task_data, f'train_L{L}')
        for i in range(min(per_size, len(d['n']))):
            n = int(d['n'][i]); pi = d['pi'][i][:n].astype(np.int64); order = preorder(pi)
            for t in range(n):
                part = np.full(n, -1); part[order[:t]] = pi[order[:t]]
                st = complete(state('dfs', d, i), part); v = order[t]
                S.append(st); T[id(st)] = (v, int(pi[v]))
    return S, T


def fit(per_size=20):
    t0 = time.perf_counter(); S, T = train_states('dfs', per_size)
    rows_off = lambda s: (~np.eye(s.n, dtype=bool)) & s.bits['unplaced'][None, :]
    off_t = lambda s: (np.arange(s.n)[None, :] == T[id(s)][0]) & (np.arange(s.n)[:, None] == T[id(s)][1]) & (T[id(s)][1] != T[id(s)][0])
    off, _ = cold.fit(S, off_t, 'pair', rows_off, conj=True)
    rows_d = lambda s: s.bits['unplaced']
    diag, _ = cold.fit(S, lambda s: (np.arange(s.n) == T[id(s)][0]) & (T[id(s)][1] == T[id(s)][0]), 'item', rows_d, conj=True)
    return dict(off=off, diag=diag), time.perf_counter() - t0, len(S)


def run_dfs(rules, A, pos, max_loops=None):
    """Cold from nothing: each loop, unplaced items take a pointer where the check passes; placed ones keep theirs."""
    n = len(A); E = A != 0; np.fill_diagonal(E, False)
    st = cold.State(n, reg={'pos': pos.astype(np.float64)}, groups=[['pos']], rel={'edge': E})
    pi = np.full(n, -1); loops = 0
    for loops in range(1, (max_loops or n + 3) + 1):
        complete(st, pi)
        new = cold.read_pointer(st, rules['off'], rules['diag'], none=True)[0]
        new = np.where(pi >= 0, pi, new)
        if (new == pi).all():
            break
        pi = new
    return pi, loops if (new == pi).all() else -loops


def evaluate(rules, d):
    pred = np.full(d['pi'].shape, -1, np.int64); loops = []; t0 = time.perf_counter()
    for i in range(len(d['n'])):
        n = int(d['n'][i]); p, l = run_dfs(rules, d['A'][i, :n, :n], d['pos'][i, :n]); loops.append(l)
        pred[i, :n] = np.where(p < 0, np.arange(n), p)
    return score_split('dfs', d, {'pi': pred}), loops, time.perf_counter() - t0


def go(big='256:8', seed=101, val=False, per_size=20, log=print):
    rules, fs, nt = fit(per_size)
    res = dict(task='dfs', fit_ms=round(1000 * fs, 1), train_states=nt, rules={k: run.describe(r) for k, r in rules.items()})
    log(json.dumps(dict(fit_ms=res['fit_ms'], **{k: dict(exact=r.exact_gate, **r.info) for k, r in rules.items()})))
    sc, loops, ev = evaluate(rules, run.load('dfs', 'test'))
    res['test64'] = dict(score=sc, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2)); log(f"official test: {res['test64']}")
    if val:
        sc, loops, ev = evaluate(rules, run.load('dfs', 'val')); res['val'] = dict(score=sc, loops_max=max(loops, key=abs)); log(f"val: {res['val']}")
    for spec in big.split(','):
        if not spec: continue
        n, num = map(int, spec.split(':')[:2]); dd = gen.make('dfs', n, num, seed + n)
        sc, loops, ev = evaluate(rules, dd); res[f'gen{n}'] = dict(score=sc, num=num, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops), eval_s=round(ev, 2))
        log(f"generated {n}: {res[f'gen{n}']}")
    return res, rules




# ------------------------------------------------------------------------- topological sort and SCC from the DFS check --
def finish_rank(pi):
    """Postorder rank of the forest (children and roots by index)."""
    n = len(pi); ch = [[] for _ in range(n)]; roots = [v for v in range(n) if pi[v] == v]
    for v in range(n):
        if pi[v] != v: ch[pi[v]].append(v)
    f = np.zeros(n); t = 0
    for r in roots:
        st = [(r, 0)]
        while st:
            u, i = st[-1]
            if i < len(ch[u]): st[-1] = (u, i + 1); st.append((ch[u][i], 0))
            else: f[u] = t; t += 1; st.pop()
    return f


def topo_states(per_size=20):
    """Topological sort's own training data, completed: f = place in the finish order, from the answer's chain."""
    S, T = [], {}
    for L in run.SIZES:
        d = run.load('topological_sort', f'train_L{L}')
        for i in range(min(per_size, len(d['n']))):
            n = int(d['n'][i]); topo = d['topo'][i][:n].astype(np.int64); head = int(d['topo_head'][i])
            f = np.zeros(n); u = head; k = n - 1
            while True:                                    # walk the chain back from the last finished
                f[u] = k; k -= 1
                if topo[u] == u: break
                u = topo[u]
            st = cold.State(n, reg={'pos': np.arange(n) / n, 'f': f}, groups=[['pos'], ['f']])
            S.append(st); T[id(st)] = (topo, head)
    return S, T


def fit_topo(per_size=20):
    t0 = time.perf_counter(); S, T = topo_states(per_size); R = {}
    R['off'], _ = cold.fit(S, lambda s: np.arange(s.n)[:, None] == T[id(s)][0][None, :], 'pair', lambda s: ~np.eye(s.n, dtype=bool))
    R['diag'], _ = cold.fit(S, lambda s: T[id(s)][0] == np.arange(s.n), 'item', lambda s: np.ones(s.n, bool))
    R['head'], _ = cold.fit(S, lambda s: np.arange(s.n) == T[id(s)][1], 'item', lambda s: np.ones(s.n, bool))
    return R, time.perf_counter() - t0


def eval_topo(dfs_rules, R, d):
    preds = dict(topo=np.full(d['topo'].shape, -1, np.int64), topo_head=np.zeros(len(d['n']), np.int64)); loops = []
    for i in range(len(d['n'])):
        n = int(d['n'][i]); p, l = run_dfs(dfs_rules, d['A'][i, :n, :n], d['pos'][i, :n]); loops.append(l)
        p = np.where(p < 0, np.arange(n), p); f = finish_rank(p)
        st = cold.State(n, reg={'pos': np.arange(n) / n, 'f': f}, groups=[['pos'], ['f']])
        preds['topo'][i, :n] = cold.read_pointer(st, R['off'], R['diag'])[0]; preds['topo_head'][i] = cold.read_one(st, R['head'])[0]
    return score_split('topological_sort', d, preds), loops


def eval_scc(dfs_rules, d):
    pred = np.full(d['scc_id'].shape, -1, np.int64); loops = []
    for i in range(len(d['n'])):
        n = int(d['n'][i]); A = d['A'][i, :n, :n]
        p1, l1 = run_dfs(dfs_rules, A, np.arange(n) / n); p1 = np.where(p1 < 0, np.arange(n), p1); f = finish_rank(p1)
        order = np.argsort(-f); pos2 = np.empty(n); pos2[order] = np.arange(n) / n      # item order by decreasing f
        p2, l2 = run_dfs(dfs_rules, A.T, pos2); p2 = np.where(p2 < 0, np.arange(n), p2)
        root = p2.copy()
        for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1): root = root[root]        # the root of each tree (closure)
        pred[i, :n] = root; loops.append(max(abs(l1), abs(l2)) * (1 if l1 > 0 and l2 > 0 else -1))
    return score_split('strongly_connected_components', d, {'scc_id': pred}), loops


def go_topo_scc(big='256:8', seed=101, val=False, per_size=20, log=print):
    dfs_rules, fs, _ = fit(per_size); R, ft = fit_topo(per_size); out = {}
    for task in ('topological_sort', 'strongly_connected_components'):
        ev = (lambda d: eval_topo(dfs_rules, R, d)) if task == 'topological_sort' else (lambda d: eval_scc(dfs_rules, d))
        res = dict(task=task, fit_ms=round(1000 * (fs + (ft if task == 'topological_sort' else 0)), 1),
                   rules=({k: run.describe(r) for k, r in R.items()} if task == 'topological_sort' else {}))
        sc, loops = ev(run.load(task, 'test')); res['test64'] = dict(score=sc, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops))
        log(f"{task} official test: {res['test64']}")
        if val:
            sc, loops = ev(run.load(task, 'val')); res['val'] = dict(score=sc, loops_max=max(loops, key=abs)); log(f"{task} val: {res['val']}")
        for spec in big.split(','):
            if not spec: continue
            n, num = map(int, spec.split(':')[:2]); dd = gen.make(task, n, num, seed + n)
            sc, loops = ev(dd); res[f'gen{n}'] = dict(score=sc, num=num, loops_max=max(loops, key=abs), not_at_rest=sum(l < 0 for l in loops))
            log(f"{task} generated {n}: {res[f'gen{n}']}")
        out[task] = res
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--big', default='')
    ap.add_argument('--seed', type=int, default=101); ap.add_argument('--val', action='store_true'); ap.add_argument('--out', default='results')
    a = ap.parse_args(); (HERE / a.out).mkdir(exist_ok=True)
    if a.task == 'dfs':
        res, _ = go(a.big, a.seed, a.val, log=lambda m: print(m, flush=True))
        json.dump(res, open(HERE / a.out / 'dfs.json', 'w'), indent=1, default=str)
    else:
        for t, res in go_topo_scc(a.big, a.seed, a.val, log=lambda m: print(m, flush=True)).items():
            json.dump(res, open(HERE / a.out / f'{t}.json', 'w'), indent=1, default=str)
