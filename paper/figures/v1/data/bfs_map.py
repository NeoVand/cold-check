"""Per-node data for the breadth-first-search figures: the recorded experiment, re-run with every node logged.

Executes code/walkthrough/completed_answer_bfs.py
up to its evaluation loop (definitions, the 30 training graphs from seed 1, the fitted 'before' rule), fits the
'order' completion's rule exactly as the script does, draws the test graphs in the script's order, and runs the rule
from 'not yet' with the kept conditions (kept) and as fitted, recording for every node the loop at which it was
settled and the parent it was given. The run is the script's `run` with one logging line (as in bfs_run.py).
Asserted: per-loop counts equal data/bfs_run.json (the same run logged loop by loop, itself asserted against the record); the kept run's parents
equal the label on every node.
Also: the same two runs on training graph 0 (12 nodes), every loop logged.
    python data/bfs_map.py
"""
import json
from pathlib import Path

import numpy as np

import os
CODE = Path(os.environ.get('COLDCHECK_DATA', Path(__file__).resolve().parents[4] / 'code'))
SRC = CODE / 'walkthrough' / 'completed_answer_bfs.py'
REF = json.loads((Path(__file__).resolve().parent / 'bfs_run.json').read_text())
OUT = Path(__file__).resolve().parent / 'bfs_map.json'
text = SRC.read_text()
G = {'__name__': 'bfs_script'}
exec(compile(text[:text.index('\nout = {}')], str(SRC), 'exec'), G)
INF = G['INF']; features = G['features']; pair_bits = G['pair_bits']; completion = G['completion']


def run_logged(kind, w, wb, A, s, top, gate=None):
    log = []
    n = len(A); idx = np.arange(n); par = np.full(n, -1); par[s] = s
    reg = np.full(n, INF if top else 0); reg[s] = 0; pp = np.full(n, INF if top else 0); pp[s] = -1
    for loop in range(min(n + 2, 80)):
        F = features(A, [reg], idx); sc = F @ w; np.fill_diagonal(sc, -1e9)
        if gate is not None: sc = np.where(F[..., gate].all(-1), sc, -1e9)
        new_par = par.copy(); changed = False
        for v in range(n):
            if v == s or (top and par[v] >= 0): continue
            u = int(sc[:, v].argmax())
            if sc[u, v] > 0 and new_par[v] != u: new_par[v] = u; changed = True
            elif sc[u, v] <= 0 and not top and new_par[v] != -1: new_par[v] = -1; changed = True
        par = new_par
        if kind == 'depth':
            new = reg.copy()
            for v in range(n):
                if v != s and par[v] >= 0 and reg[par[v]] < INF: new[v] = reg[par[v]] + 1
        else:
            pp = np.where((par >= 0) & (reg[np.maximum(par, 0)] < INF), reg[np.maximum(par, 0)], INF if top else 0); pp[s] = -1
            has = pp < INF
            cb = pair_bits(pp) ; ci = pair_bits(idx)
            Fb = np.stack([cb[0], cb[1], cb[2], ci[0], cb[1] & ci[0], np.ones((n, n), bool)], -1).astype(np.float64)
            before = (Fb @ wb > 0) & has[:, None] & has[None, :]; np.fill_diagonal(before, False)
            new = np.where(has, before.sum(0), INF if top else 0); new[s] = 0
        log.append((par.copy(), new.copy()))
        if not changed and (new == reg).all(): break
        reg = new
    return par, loop, log


train, wb = G['train'], G['wb']
X, y = G['dataset']('order', train)
gate = [j for j in range(5) if X[y > 0][:, j].all()]
w = G['logreg'](X, y)
assert gate == REF['gate'] and np.allclose(w, REF['w'], atol=1e-6)
gs = {n: G['graphs'](n, reps) for n, reps in ((16, 100), (64, 100), (256, 20), (1024, 3))}


def per_node(A, s, par, g):
    n = len(A)
    p, k, log = run_logged('order', w, wb, A, s, True, g)
    first = np.full(n, -1)
    for i, (q, _) in enumerate(log):
        first[(q >= 0) & (first < 0)] = i + 1           # loops counted from 1
    first[s] = 0
    return dict(loops=k, settle_loop=first.tolist(), parent=p.tolist(),
                settled=[int(((q >= 0) & (np.arange(n) != s)).sum()) for q, _ in log],
                right=[int(((q == par) & (np.arange(n) != s)).sum()) for q, _ in log],
                place=[[int(x) if x < INF else -1 for x in r] for _, r in log] if n <= 16 else None)


A, s, par = gs[1024][0]
depth, place, _ = completion(par, s)
rec = dict(n=len(A), source=int(s), edges=[[int(a), int(b)] for a, b in zip(*np.nonzero(np.triu(A)))],
           label=par.tolist(), depth=depth.tolist(), place=place.tolist())
rec['kept'] = per_node(A, s, par, gate)
rec['as_fitted'] = per_node(A, s, par, None)
ref = REF['runs1024'][0]
for arm in ('kept', 'as_fitted'):
    assert rec[arm]['settled'] == ref[arm]['settled'] and rec[arm]['right'] == ref[arm]['right'], arm
assert rec['kept']['parent'] == rec['label']
A0, s0, par0 = train[0]
d0, p0, _ = completion(par0, s0)
t0 = dict(n=len(A0), source=int(s0), edges=[[int(a), int(b)] for a, b in zip(*np.nonzero(np.triu(A0)))],
          label=par0.tolist(), depth=d0.tolist(), place=p0.tolist(),
          kept=per_node(A0, s0, par0, gate), as_fitted=per_node(A0, s0, par0, None))
assert t0['edges'] == REF['train0']['edges'] and t0['label'] == REF['train0']['par']
# loops against levels on every test graph (kept), and parents right for both arms per size
lv = []
for n in (16, 64, 256):
    for A_, s_, par_ in gs[n]:
        p_, k_, _ = run_logged('order', w, wb, A_, s_, True, gate)
        lv.append([n, int(completion(par_, s_)[0].max()), int(k_), bool((p_ == par_).all())])
OUT.write_text(json.dumps(dict(source=dict(script='code/walkthrough/completed_answer_bfs.py', checked_against='data/bfs_run.json (runs1024[0], train0, w, gate)'),
                               gate=gate, w=[float(x) for x in w], graph1024=rec, train0=t0, loops_levels=lv)))
print('wrote', OUT, 'kept loops', rec['kept']['loops'], 'as fitted loops', rec['as_fitted']['loops'],
      'train0 as fitted', t0['as_fitted']['parent'], 'label', t0['label'])
