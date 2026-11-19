"""Our copy of the library's scorer (no jax needed)."""
import numpy as np

from .tasks import MASKS


def score(pred, d, outs):
    """Our copy of the library's scorer (clrs.evaluate), for one prediction {name: [samples, ...]}: pointers and one-of
    outputs score exact index matches (mean over items), masks the F1 pooled over the split (entries marked -1 excluded),
    categories the accuracy over cells not marked -1; the mean over outputs. scripts/score_library.py runs clrs.evaluate
    itself and checks it agrees."""
    sc = []
    for name, kind in outs:
        t, q = d[name], np.asarray(pred[name])
        if name == 'unsat' and 's' in d:  # the constraint problems: the one answer, at the global item s; share of inputs right
            i = np.arange(len(t))
            sc.append(float(np.mean((q[i, d['s']] > .5) == (t[i, d['s']] > .5))))
        elif kind in MASKS or kind == 'gmask':
            q, ok = q > .5, t != -1
            tp, fp, fn = (q & (t > .5) & ok).sum(), (q & (t < .5) & ok).sum(), (~q & (t > .5) & ok).sum()
            pr = tp / (tp + fp) if tp + fp > 0 else 1.
            rc = tp / (tp + fn) if tp + fn > 0 else 1.
            sc.append(float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.)
        elif kind == 'node':
            sc.append(float(np.mean(q[:, :t.shape[1]] == t)))
        elif kind == 'ecat':
            ok = t >= 0
            sc.append(float((q == t)[ok].sum() / ok.sum()))
        else:
            sc.append(float(np.mean(q == t)))
    return float(np.mean(sc))
