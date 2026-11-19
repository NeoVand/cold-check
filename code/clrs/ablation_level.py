"""Level ablation on CLRS-30's own labels (registered before it ran): the fitted check with and without the kept
conditions, run cold from nothing. Uses the frozen run.py and cold.py unchanged; the variants are copies of the one
fitted rule with parts of its conjunction removed."""
import copy, json, os, time
from pathlib import Path
import numpy as np
import cold, run, gen
from tasks import state

HERE = Path(__file__).parent


def variant(rule, which):
    r = copy.copy(rule); ng = rule.info.get('greedy', 0)
    if which == 'a':                      # the sparse score alone
        r.gate, r.exact_gate = [], False
    elif which == 'c':                    # own kept conditions + score
        r.gate, r.exact_gate = rule.gate[:len(rule.gate) - ng], False
    return r


def score_cold(task, rules, d):
    name = run.out_name(task); ok = tot = 0; rest = 0; loops = []
    for i in range(len(d['n'])):
        st = state(task, d, i); p, l = run.predict(task, rules, st); y = d[name][i][:st.n]
        ok += int((np.asarray(p) == y).sum()); tot += st.n; loops.append(l); rest += l > 0
    return dict(right=ok / tot, at_rest=f'{rest}/{len(loops)}', loops_max=max(loops, key=abs))


def score_at_answer(task, rules, d):
    name = run.out_name(task); ok = tot = 0
    for i in range(len(d['n'])):
        st = state(task, d, i); run.complete_static(task, st); y = d[name][i][:st.n]
        run.complete_answer(task, st, y); p = cold.read_pointer(st, rules['off'], rules['diag'])[0]
        ok += int((p == y).sum()); tot += st.n
    return ok / tot


if __name__ == '__main__':
    out = {}
    for task in ('bfs', 'mst_prim', 'dijkstra', 'dag_shortest_paths'):
        rules, fs, _ = run.fit_task(task, 20, 1e-3)
        c = 4 / 1024 if task in run.DIRECTED else np.sqrt(4 / 1024)
        tests = {'official64': run.load(task, 'test'), 'fresh1024': gen.make(task, 1024, 8, 9000 + 1024),
                 'fresh1024_sparse': gen.make(task, 1024, 8, 9000 + 1024 + 7, p=(c,))}
        res = {}
        for v in ('a', 'c', 'b'):
            R = dict(rules, off=variant(rules['off'], v), diag=variant(rules['diag'], v))
            res[v] = {k: score_cold(task, R, d) for k, d in tests.items()}
            if v == 'a':
                res['a_at_answer'] = {k: score_at_answer(task, R, d) for k, d in tests.items()}
            print(task, v, res[v], flush=True)
        res['conditions'] = dict(off=[(str(f), val) for f, val in rules['off'].gate], diag=[(str(f), val) for f, val in rules['diag'].gate])
        out[task] = res
    # activity selection: the answer read without the downward level, against the final fit
    task = 'activity_selector'; S, Y = run.train_states(task, 20); S = [run.with_answer(task, s, y) for s, y in zip(S, Y)]
    tg = {id(s): y for s, y in zip(S, Y)}
    nolevel, _ = cold.fit(S, lambda s: np.asarray(tg[id(s)][:s.n]) > 0, 'item', lambda s: np.ones(s.n, bool))
    rules, _, _ = run.fit_task(task, 20, 1e-3)
    tests = {'official64': run.load(task, 'test'), 'fresh1024': gen.make(task, 1024, 8, 9000 + 1024)}
    res = {}
    os.environ['GATE_SCOPE'], os.environ['GREEDY'] = 'all', '0'          # the original fit: every condition kept, no greedy
    original, _ = cold.fit(S, lambda s: np.asarray(tg[id(s)][:s.n]) > 0, 'item', lambda s: np.ones(s.n, bool))
    os.environ['GATE_SCOPE'], os.environ['GREEDY'] = 'own', '1'
    for v, R in (('original_all_conditions_no_level', dict(rules, mask=original)), ('no_level', dict(rules, mask=nolevel)), ('level', rules)):
        r2 = {}
        for k, d in tests.items():
            sc, loops, _ = run.evaluate(task, R, d); r2[k] = dict(score=sc, at_rest=f'{sum(l > 0 for l in loops)}/{len(loops)}', loops_max=max(loops, key=abs))
        res[v] = r2; print(task, v, r2, flush=True)
    out[task] = res
    json.dump(out, open(HERE / 'final' / 'ablation_level.json', 'w'), indent=1, default=str)
