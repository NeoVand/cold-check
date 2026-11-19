"""CLRS-30 inputs as cold states, and the per-task bookkeeping (which output, which completion, which start).

Everything a task adds is declared here and nowhere else:
  inputs     read from the split file by its type (item numbers, per-input numbers, masks, graphs), units grouped as the
             library's own spec says they are measured (tasks.SAME_UNITS of witness_clrs; task scheduling's deadlines in
             the units of positions, i.e. divided by n: a labelled, hand-supplied scaling, as in our paper)
  output     its kind (pointer, one, item mask, pair mask, pair category)
  completion one entry of the shared library in complete.py, or none
"""
import sys
import numpy as np

from witness_clrs.tasks import OUTPUTS, SAME_UNITS, FLOYDNET  # noqa: E402
from witness_clrs.scoring import score as library_score        # noqa: E402

from cold import State  # noqa: E402

ITEM_SKIP = ('x_pos',)
WEIGHTED = ('mst_kruskal', 'mst_prim', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall')


def item_state(task, d, i):
    n = int(d['n'][i]); reg, glob, bits, rel = {'pos': d['x_pos'][i, :n].astype(np.float64)}, {}, {}, {}
    for k in sorted(d):
        if not k.startswith('x_') or k in ITEM_SKIP:
            continue
        a = d[k]
        if a.ndim == 1:
            glob[k[2:]] = float(a[i])
        elif a.ndim == 2 and a.dtype.kind == 'f':
            v = a[i, :n].astype(np.float64)
            reg[k[2:]] = v / n if task == 'task_scheduling' and k == 'x_d' else v
        elif a.ndim == 2:
            bits[k[2:]] = a[i, :n] > 0
        else:                                                    # a category per item: "same category" is a pair relation
            c = a[i, :n].argmax(-1); rel['same_' + k[2:]] = c[:, None] == c[None, :]
    same = [g for g in SAME_UNITS.get(task, [])]
    groups = same + [[r] for r in list(reg) + list(glob) if not any(r in g for g in same)]
    return State(n, reg=reg, glob=glob, groups=groups, bits=bits, rel=rel)


def graph_state(task, d, i):
    n = int(d['n'][i]); A = d['A'][i, :n, :n].astype(np.float64)
    edge = A != 0; np.fill_diagonal(edge, False)
    reg = {'pos': d['pos'][i, :n].astype(np.float64)}
    bits = {}
    if 's' in d:
        bits['source'] = np.arange(n) == int(d['s'][i])
    rel = {'edge': edge}
    pnum, pgroups = {}, []
    if task in WEIGHTED:
        pnum['w'] = np.where(edge, A, np.nan); pgroups = [['w']]     # no edge: no number (every comparison with it is false)
    st = State(n, reg=reg, groups=[['pos']], bits=bits, rel=rel, pnum=pnum, pgroups=pgroups)
    if task == 'floyd_warshall':        # its labels read the diagonal of A (a self-loop's weight); nan = no self-loop
        dA = np.diag(A).copy(); st.reg['loop'] = np.where(dA != 0, dA, np.nan)
        st.groups.append(['loop']); st.xgroups = [('w', 'loop'), ('S', 'loop')]
    return st


def state(task, d, i):
    return graph_state(task, d, i) if 'adj' in d else item_state(task, d, i)


def score_split(task, d, preds):
    """The library's scorer on a split: preds {output name: padded array like the split's}."""
    return library_score(preds, d, OUTPUTS[task])
