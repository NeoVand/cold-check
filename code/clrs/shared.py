"""A shared check (registered before it ran): one rule fitted on the pooled training answers of a family of tasks
with one output type, over the union of the per-task fits' conditions, on a union input schema. Completions and
primitives stay per task (the task's encoder); the check is one. Variant T adds the task identity as an item bit;
variant N has none. Frozen files are used unchanged."""
import json, sys, time
from pathlib import Path
import numpy as np
import cold, run, gen, extra, gen_extra
from tasks import state

HERE = Path(__file__).parent
POINTER = ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'bfs', 'mst_prim', 'bellman_ford', 'dijkstra', 'dag_shortest_paths']
MASK = ['activity_selector', 'task_scheduling', 'graham_scan', 'jarvis_march', 'articulation_points']
EXTRA = ('graham_scan', 'jarvis_march', 'articulation_points')


class Schema:
    def __init__(self):
        self.reg, self.rel, self.pnum, self.bits = set(), set(), set(), set()

    def add(self, st):
        self.reg |= set(st.reg); self.rel |= set(st.rel); self.pnum |= set(st.pnum); self.bits |= set(st.bits)


def unify(st, sch, task, with_task):
    n = st.n
    for r in sch.reg - set(st.reg): st.reg[r] = np.full(n, np.nan)
    for r in sch.rel - set(st.rel): st.rel[r] = np.zeros((n, n), bool)
    for r in sch.pnum - set(st.pnum): st.pnum[r] = np.full((n, n), np.nan)
    for b in sch.bits - set(st.bits): st.bits[b] = np.zeros(n, bool)
    if with_task:
        for t in TASKS_NOW: st.bits['task_' + t] = np.full(n, t == task)
    st._c.clear(); return st


TASKS_NOW = []


def per_task(task):
    """Training states with the answer (as each per-task fit sees them), targets, and the per-task fitted rules."""
    if task in EXTRA:
        rules, _, _ = extra.fit_task(task); S, Y = [], []
        for L in run.SIZES:
            d = run.load(task, f'train_L{L}')
            for i in range(20):
                S.append(extra.make_state(task, d, i)); Y.append(d['is_cut' if task == 'articulation_points' else 'in_hull'][i])
        return rules, S, Y
    rules, _, _ = run.fit_task(task, 20, 1e-3); S, Y = run.train_states(task, 20)
    S = [run.with_answer(task, s, y) for s, y in zip(S, Y)]
    return rules, S, Y


def fit_family(tasks, kind_names, with_task):
    global TASKS_NOW
    TASKS_NOW = tasks; t0 = time.perf_counter(); sch = Schema(); data = {}; feats = {k: set() for k in kind_names}
    for t in tasks:
        rules, S, Y = per_task(t); data[t] = (S, Y)
        for s in S: sch.add(s)
        for k in kind_names:
            if k in rules: feats[k] |= set(rules[k].used())
    for t in tasks:
        for s in data[t][0]: unify(s, sch, t, with_task)
    if with_task:
        for k in kind_names:
            for t in tasks:
                feats[k].add(('bitA', ('bit', 'task_' + t)))
                if k == 'off': feats[k].add(('bitB', ('bit', 'task_' + t)))
    allS = [s for t in tasks for s in data[t][0]]; Y = {id(s): y for t in tasks for s, y in zip(*data[t])}
    task_of = {id(s): t for t in tasks for s in data[t][0]}
    shared = {}
    for k in kind_names:
        F = sorted(feats[k], key=str)
        if k == 'off':
            Sk = [s for s in allS]
            shared[k], _ = cold.fit(Sk, lambda s: np.asarray(Y[id(s)][:s.n])[None, :] == np.arange(s.n)[:, None], 'pair',
                                    lambda s: ~np.eye(s.n, dtype=bool), feats=F)
        elif k == 'diag':
            shared[k], _ = cold.fit(allS, lambda s: np.asarray(Y[id(s)][:s.n]) == np.arange(s.n), 'item', lambda s: np.ones(s.n, bool), feats=F)
        else:
            shared[k], _ = cold.fit(allS, lambda s: np.asarray(Y[id(s)][:s.n]) > 0, 'item', lambda s: np.ones(s.n, bool), feats=F)
    return shared, sch, time.perf_counter() - t0


def predict(task, R, sch, st, with_task):
    k = run.KIND.get(task)
    if task in EXTRA:
        unify(st, sch, task, with_task); return R['mask'].decide(st).astype(int), 1
    run.complete_static(task, st)
    if k == 'pointer':
        unify(st, sch, task, with_task); return cold.read_pointer(st, R['off'], R['diag'])[0], 1
    if k == 'pointer_loop':
        pi = np.full(st.n, -1)
        for loop in range(1, 2 * st.n + 9):
            run.complete_answer(task, st, pi); unify(st, sch, task, with_task)
            new = cold.read_pointer(st, R['off'], R['diag'], none=True)[0]
            if (new == pi).all(): break
            pi = new
        return np.where(pi < 0, np.arange(st.n), pi), loop
    sel = np.zeros(st.n, bool); st.answer.add('sel')                   # mask with answer feedback
    for loop in range(1, 4 * st.n + 9):
        st.bits['sel'] = sel; unify(st, sch, task, with_task); new = R['mask'].decide(st)
        if (new == sel).all(): break
        sel = new
    return sel.astype(int), loop


def evaluate(task, R, sch, d, with_task):
    name = 'is_cut' if task == 'articulation_points' else ('in_hull' if task in EXTRA else run.out_name(task))
    ok = tot = 0
    for i in range(len(d['n'])):
        st = extra.make_state(task, d, i) if task in EXTRA else state(task, d, i)
        p, _ = predict(task, R, sch, st, with_task); y = d[name][i][:st.n]; ok += int((np.asarray(p) == y).sum()); tot += st.n
    return ok / tot


EXPERIMENTS = [
    ('A_pointer_all_task_bit', POINTER, ('off', 'diag'), True),
    ('A_pointer_all_no_task_bit', POINTER, ('off', 'diag'), False),
    ('B_sorts_no_task_bit', POINTER[:4], ('off', 'diag'), False),
    ('B_shortest_paths_no_task_bit', ['bellman_ford', 'dijkstra', 'dag_shortest_paths'], ('off', 'diag'), False),
    ('C_bfs_prim_paths_no_task_bit', ['bfs', 'mst_prim', 'bellman_ford', 'dijkstra', 'dag_shortest_paths'], ('off', 'diag'), False),
    ('D_mask_all_task_bit', MASK, ('mask',), True),
    ('D_mask_all_no_task_bit', MASK, ('mask',), False),
]

if __name__ == '__main__':
    out = {}
    for key, tasks, kinds, with_task in EXPERIMENTS:
        R, sch, ft = fit_family(tasks, kinds, with_task)
        res = dict(tasks=tasks, task_bit=with_task, fit_ms=round(1000 * ft),
                   rules={k: dict(exact=r.exact_gate, kept=[(str(f), v) for f, v in r.gate], n_score_features=len(r.feats), **r.info) for k, r in R.items()})
        for t in tasks:
            mk = gen_extra.make if t in EXTRA else gen.make
            res[t] = dict(official64=evaluate(t, R, sch, run.load(t, 'test'), with_task),
                          fresh1024=evaluate(t, R, sch, mk(t, 1024, 8, 9000 + 1024), with_task))
            print(key, t, res[t], flush=True)
        out[key] = res; print(key, 'fit_ms', res['fit_ms'], {k: (v['exact'], len(v['kept']), v['n_score_features']) for k, v in res['rules'].items()}, flush=True)
        json.dump(out, open(HERE / 'final' / 'shared.json', 'w'), indent=1, default=str)
