"""DFS fitted at the completed answer (registered before it ran): (a) own times visible, read at the answer; (b) the
same rule run cold from nothing; (c) own times read only as "below me", read at the answer. Uses cold.py unchanged."""
import json, sys, time
import numpy as np
import cold, run
from tasks import state

def forest_times(pi):
    """Discovery and finish times of the answer forest on one clock (children and roots in index order); inf for
    items with no pointer (pi < 0)."""
    n = len(pi); placed = pi >= 0; ch = [[] for _ in range(n)]; roots = []
    for v in range(n):
        if not placed[v]: continue
        if pi[v] == v: roots.append(v)
        else: ch[pi[v]].append(v)
    d = np.full(n, np.inf); f = np.full(n, np.inf); t = 0
    for r in roots:
        st = [(r, 0)]; d[r] = t; t += 1
        while st:
            u, i = st[-1]
            if i < len(ch[u]):
                st[-1] = (u, i + 1); c = ch[u][i]
                if np.isinf(d[c]): d[c] = t; t += 1; st.append((c, 0))
            else:
                f[u] = t; t += 1; st.pop()
    return d, f

def complete(st, pi, hide):
    d, f = forest_times(np.asarray(pi)); st.reg['d'], st.reg['f'] = d, f
    if ['d', 'f'] not in st.groups: st.groups.append(['d', 'f'])
    st.bits['placed'] = np.isfinite(d); st.answer.add('placed')
    if hide: st.hidden |= {'d', 'f'}
    st._c.clear(); return st

def fit(hide, per_size=20):
    S, Y = run.train_states('dfs', per_size)
    S = [complete(s, np.asarray(y[:s.n]), hide) for s, y in zip(S, Y)]
    tgt = {id(s): np.asarray(y[:s.n]) for s, y in zip(S, Y)}
    off, _ = cold.fit(S, lambda s: tgt[id(s)][None, :] == np.arange(s.n)[:, None], 'pair', lambda s: ~np.eye(s.n, dtype=bool), conj=True)
    diag, _ = cold.fit(S, lambda s: tgt[id(s)] == np.arange(s.n), 'item', lambda s: np.ones(s.n, bool), conj=True)
    return off, diag

def at_answer(off, diag, d, hide):
    ok = tot = 0
    for i in range(len(d['n'])):
        st = state('dfs', d, i); y = d['pi'][i][:st.n]; complete(st, y, hide)
        p = cold.read_pointer(st, off, diag)[0]; ok += (p == y).sum(); tot += st.n
    return ok / tot

def cold_run(off, diag, d, hide, max_loops=None):
    ok = tot = 0; loops = []
    for i in range(len(d['n'])):
        st = state('dfs', d, i); n = st.n; y = d['pi'][i][:n]; pi = np.full(n, -1)
        for loop in range(1, (max_loops or 2 * n + 8) + 1):
            complete(st, pi, hide); new = cold.read_pointer(st, off, diag, none=True)[0]
            if (new == pi).all(): break
            pi = new
        loops.append(loop if (new == pi).all() else -loop)
        p = np.where(pi < 0, np.arange(n), pi); ok += (p == y).sum(); tot += n
    return ok / tot, loops

if __name__ == '__main__':
    d = run.load('dfs', 'test'); out = {}
    for hide in (False, True):
        t0 = time.perf_counter(); off, diag = fit(hide); ft = time.perf_counter() - t0
        a = at_answer(off, diag, d, hide); c, loops = cold_run(off, diag, d, hide)
        out['hidden' if hide else 'visible'] = dict(fit_ms=round(1000 * ft), at_answer=a, cold=c, loops_max=max(loops, key=abs),
                                                    not_at_rest=sum(l < 0 for l in loops), off=dict(exact=off.exact_gate, **off.info),
                                                    diag=dict(exact=diag.exact_gate, **diag.info))
        print('hidden' if hide else 'visible', out['hidden' if hide else 'visible'], flush=True)
    json.dump(out, open('dfs_test.json', 'w'), indent=1, default=str)
