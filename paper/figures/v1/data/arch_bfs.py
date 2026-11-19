"""Data for the architecture figure: the CLRS-30 breadth-first-search check, fitted at the completed answer exactly as
code/clrs/run.py fits it (100 training inputs, 20 at each training size, lam 1e-3), then run cold from nothing on one
small input of the library's generator (data/glyphs/bfs.npz, the Fig. 5 glyph input: ten nodes), every loop logged.

Writes data/arch_bfs.json:
  fit        fit time (s, this machine), training inputs, the gate's conditions in words (from code/clrs/RULES.md)
  train      one training input (the first of 7 to 11 nodes, all reached, at least three levels deep): edges, source, the answer (parent pointers), its completion (depth)
  run        the test input: edges, source, the answer; per loop: pointers, depth register, candidates passing per node
  read       at rest, for every node v: the fitted score of every candidate u and whether it passes the check
  witness    at rest, the witness legs leg1(u, k) = [pos u >= pos k], leg2(k, v) = [edge k-v and depth k < depth v],
             and their count N(u, v) = sum_k leg1 leg2 (k not u, v): the check's witness test is N = 0
Asserted: the cold run returns the label; the witness count from the legs equals the rule's own witness feature.
    python data/arch_bfs.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get('COLDCHECK_DATA', HERE.parents[3] / 'code'))
sys.path.insert(0, str(CODE / 'clrs'))
import run  # noqa: E402
import complete  # noqa: E402
import cold  # noqa: E402

t0 = time.perf_counter()
rules, fit_s, ntrain = run.fit_task('bfs', 20, 1e-3)
off, diag = rules['off'], rules['diag']
rules_md = (CODE / 'clrs' / 'RULES.md').read_text()
sec = rules_md[rules_md.index('## bfs'):rules_md.index('## binary_search')]
conds = [l[2:].strip() for l in sec.split('**v points to itself')[0].splitlines() if l.startswith('- ')]
assert len(conds) == len(off.gate), (conds, off.gate)

# one training input with its completed answer
S, Y = run.train_states('bfs', 20)
def _ok(i):
    dd = complete.depth(np.asarray(Y[i][:S[i].n]), S[i].bits['source'])
    return 7 <= S[i].n <= 11 and np.isfinite(dd).all() and dd.max() >= 3
i7 = next(i for i in range(len(S)) if _ok(i))
s7, y7 = S[i7], np.asarray(Y[i7][:S[i7].n])
src7 = int(np.argmax(s7.bits['source']))
train = dict(n=s7.n, source=src7, edges=[[int(a), int(b)] for a, b in zip(*np.nonzero(np.triu(s7.rel['edge'] | s7.rel['edge'].T, 1)))],
             answer=y7.tolist(), depth=[None if not np.isfinite(x) else int(x) for x in complete.depth(y7, s7.bits['source'])])

# the test input, run cold loop by loop (run.predict's pointer loop, logged)
z = dict(np.load(HERE / 'glyphs' / 'bfs.npz'))
d = z                                                                  # batched: input 0, as the glyph
st = run.state('bfs', d, 0)
run.complete_static('bfs', st)
n = st.n
label = np.asarray(z['pi'][0][:n]).astype(int)
pi = np.full(n, -1)
log = []
for loop in range(1, 2 * n + 9):
    run.complete_answer('bfs', st, pi)
    new, npass = cold.read_pointer(st, off, diag, none=True)
    log.append(dict(pointers=new.tolist(), depth=[None if not np.isfinite(x) else int(x) for x in st.reg['depth']],
                    passing=npass.tolist()))
    if (new == pi).all():
        break
    pi = new
final = np.where(pi < 0, np.arange(n), pi)
assert (final == label).all(), (final, label)
run.complete_answer('bfs', st, final)
so, ok = off.score(st), off.decide(st)
pos, edge, depth = st.reg['pos'], st.rel['edge'], st.reg['depth']
leg1 = pos[:, None] >= pos[None, :]                                    # leg1[u, k]
leg2 = edge & (depth[:, None] < depth[None, :])                        # leg2[k, v]
N = np.zeros((n, n), int)
for u in range(n):
    for v in range(n):
        N[u, v] = sum(1 for k in range(n) if k not in (u, v) and leg1[u, k] and leg2[k, v])
wf = [f for f, _ in off.gate if f[0] == 'wit']
assert len(wf) == 1, off.gate
feat = np.asarray(st.feature(wf[0], 'pair'), bool)
off_diag = ~np.eye(n, dtype=bool)
assert ((N > 0) == feat)[off_diag].all()
out = dict(fit=dict(seconds=fit_s, train_inputs=ntrain, conditions=conds, gate=[[str(f), bool(v)] for f, v in off.gate]),
           train=train,
           run=dict(n=n, source=int(z['s'][0]), edges=[[int(a), int(b)] for a, b in zip(*np.nonzero(np.triu(edge | edge.T, 1)))],
                    pos=pos.tolist(), answer=label.tolist(), loops=log, loops_to_rest=len(log)),
           read=dict(score=np.where(off_diag, so, np.nan).tolist(), passes=(ok & off_diag).tolist()),
           witness=dict(leg1=leg1.tolist(), leg2=leg2.tolist(), count=N.tolist()))
(HERE / 'arch_bfs.json').write_text(json.dumps(out))
print('fit', round(fit_s * 1e3), 'ms on', ntrain, 'inputs; loops', len(log), 'gate', off.gate)
for c in conds:
    print('  ', c)
