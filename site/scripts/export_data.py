"""Export the fitted rules, test instances and the Python outputs the page and its parity tests need.

Reads (never writes) code/{bench,released,walkthrough} of this repository (run code/fetch.py first); writes
site/static/data/*.json.
Run (from anywhere; one heavy CPU job at a time on the laptop):
    NUMBA_CACHE_DIR=<tmp> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    uv run --no-project --with numpy==2.2.6 --with numba==0.61.2 python export_data.py [sudoku|mazes|bfs|all]
Every number shown on the page comes from these files; each file carries a `source` block naming its inputs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2] / 'code'
BENCH = CODE / 'bench'
RELEASED = CODE / 'released'   # CRAM's released models, run by released/modal_cram_check.py
WALK = CODE / 'walkthrough'
# final/run_final.py saves its per-puzzle Sudoku-Extreme record to bench/data/; FINAL_RECORDS points elsewhere if needed
FINAL_RECORDS = Path(os.environ.get('FINAL_RECORDS', str(BENCH / 'data')))
OUT = Path(__file__).resolve().parent.parent / 'static' / 'data'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BENCH))


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(',', ':')))
    print('wrote', name, (OUT / name).stat().st_size, 'bytes', flush=True)


def ints(a):
    return [int(x) for x in np.asarray(a).ravel()]


# ------------------------------------------------------------------------------------------------------- Sudoku --
def final_sudoku_stats():
    """Every number the Sudoku card shows, from the final run (frozen engine, budget 10^7, unit pairs and restarts)."""
    F = BENCH / 'final'
    ext = json.loads((F / 'sudoku_extreme.json').read_text())['fits']
    loc = json.loads((F / 'sudoku_ladder_local.json').read_text())
    s25 = json.loads((F / 'sudoku_ladder_25.json').read_text())['summary']
    ext = [dict(f, correct=f['solved'] - f['wrong_stops']) for f in ext]
    return dict(extreme=[{k: f[k] for k in f if k != 'kept'} for f in ext], side4=loc['side4'], side9=loc['side9'],
                side16=loc['side16'], side25=s25)


BUDGET, RESTART = 10 ** 7, 10 ** 4   # the final run (bench/final/run_final.py): budget, Luby restart base, unit pairs on


def final_engine_ok():
    """The engine the page is checked against is the frozen final one (bench/final/ENGINE_SHA256.json)."""
    import hashlib
    want = json.loads((BENCH / 'final' / 'ENGINE_SHA256.json').read_text())
    for f in ('check.py', 'engine.py'):
        assert hashlib.sha256((BENCH / f).read_bytes()).hexdigest() == want[f], ('not the frozen final engine', f)


def solve_final(engine, E, dom0, seed):
    return engine.run2(E, dom0, seed=seed, budget=BUDGET, use_pairs=True, restart_base=RESTART)


def cold_fixed(engine, E, dom0):
    """Cells the first cold stage (count bounds + unit pairs) leaves with one value, clues excluded."""
    pa, pb, inb, bout, bout_n = E['pairs']
    dom = dom0.copy()
    nu = len(E['unit_size'])
    engine.propagate2(dom, E['unit_items'], E['unit_size'], E['item_units'], E['ge1'], E['le1'], E['V'],
                      np.zeros(nu, np.int64), np.ones(nu, np.bool_), np.zeros(2, np.int64), pa, pb, inb, True, bout, bout_n)
    single = lambda d: d != 0 and (d & (d - 1)) == 0
    return int(sum(single(int(d)) for d in dom) - sum(single(int(d)) for d in dom0))


def py_record(engine, E, dom0, dom, o, good):
    x = engine.decode(dom)
    return dict(status=int(o[0]), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]), cold=bool(o[4]), restarts=int(o[6]),
                cold_fixed=cold_fixed(engine, E, dom0), board=ints(x), correct=bool(good(x)))


def sudoku():
    import check
    import data_io
    import engine
    import run_sudoku_sizes as R

    final_engine_ok()
    check.vocabulary()
    sol9, _, _ = data_io.sudoku_split('train-s0')
    ck, _ = check.fit(sol9.astype(np.int64), steps=4, lr=8.0)  # the train-s0 fit of the final run (steps 4, lr 8)
    f0 = json.loads((BENCH / 'final' / 'sudoku_extreme.json').read_text())['fits'][0]
    assert f0['split'] == 'train-s0' and f0['steps'] == 4 and f0['lr'] == 8.0
    chk = dict(types=ints(ck['types']), w=[float(x) for x in ck['w']], relations=[list(fs) for fs in ck['relations']],
               relation_names=[[check.FNAMES[i] for i in fs] for fs in ck['relations']],
               conditions=[[bool(a), bool(b)] for a, b in ck['conditions']], tau=float(ck['tau']),
               kept=check.describe(ck))
    assert [[check.FNAMES[i] for i in fs] for fs in ck['relations']] == f0['relations']

    # test puzzles: 24 spread over the 412,786 (parity), plus named demo puzzles chosen from the final per-puzzle run
    ts, tc, tr = data_io.sudoku_split('test')
    rec = np.load(FINAL_RECORDS / 'final_sudoku_extreme_s0.npz')
    n = len(ts)
    spread = [int(k * (n // 24)) for k in range(24)]
    mv = rec['moves']
    order = np.argsort(mv, kind='stable')
    median_i = int(order[n // 2])
    hard_i = int(np.argmax(mv))
    # a puzzle a person can watch: the first test puzzle whose search takes between 150 and 400 moves
    watch_i = int(np.flatnonzero((mv >= 150) & (mv <= 400))[0])
    demo = dict(watch=watch_i, median=median_i, hardest=hard_i)
    E = engine.build(ck, 3)
    rows = []
    for i in sorted(set(spread) | set(demo.values())):
        puz = np.where(tc[i], ts[i], 0).astype(np.int64)
        dom0 = engine.masks(puz, 9)
        dom, o = solve_final(engine, E, dom0, i + 1000003 * 0)  # run_final.py's seed for split 0
        assert (int(o[1]), int(o[2]), int(o[6])) == (int(rec['moves'][i]), int(rec['ops'][i]), int(rec['restarts'][i])), ('not the final run', i)
        rows.append(dict(index=i, rating=int(tr[i]), puzzle=ints(puz), solution=ints(ts[i]), seed=i,
                         py=py_record(engine, E, dom0, dom, o, lambda x: (x == ts[i]).all())))
        print('extreme', i, int(o[1]), flush=True)

    # other sizes: the benchmark's own generator and seeds (run_sudoku_sizes.py, as final/run_final.py ladder_one),
    # solved with the 9 x 9 fit by the final engine
    sizes = {}
    for b, N in ((2, 20), (4, 20)):
        Eb = engine.build(ck, b)
        out = []
        for i in range(N):
            rng = np.random.default_rng([1000 + b, i])
            puz, solb = R.make_puzzle(Eb, b, rng, None)
            dom0 = engine.masks(puz, b * b)
            dom, o = solve_final(engine, Eb, dom0, 7 + i)
            out.append(dict(index=i, puzzle=ints(puz), solution=ints(solb), seed=7 + i,
                            py=py_record(engine, Eb, dom0, dom, o, lambda x: R.valid(x, b))))
            print('side', b * b, i, out[-1]['py']['moves'], flush=True)
        sizes[str(b * b)] = out
    # 25 x 25: puzzles of the 100 the final run solved (bench/sudoku25b.json keeps them; same generator and seeds),
    # each solved again here and checked against final/sudoku_ladder_25.json move for move
    E25 = engine.build(ck, 5)
    fin = {r['i']: r for r in json.loads((BENCH / 'final' / 'sudoku_ladder_25.json').read_text())['rows']}
    kept = {r['i']: r for r in json.loads((BENCH / 'sudoku25b.json').read_text())['rows']}
    s25 = []
    for i in (5, 0, 14, 11, 2, 7, 1):   # 1: one restart in the final run
        puz = np.array(kept[i]['puzzle'], np.int64)
        dom0 = engine.masks(puz, 25)
        dom, o = solve_final(engine, E25, dom0, 7 + i)
        assert (int(o[1]), int(o[2]), int(o[6])) == (fin[i]['moves'], fin[i]['ops'], fin[i]['restarts']), ('not the final 25 x 25 run', i)
        s25.append(dict(index=i, puzzle=ints(puz), seed=7 + i, py=py_record(engine, E25, dom0, dom, o, lambda x: R.valid(x, 5))))
        print('side 25', i, int(o[1]), flush=True)
    sizes['25'] = s25

    # every number the Sudoku card shows
    stats = final_sudoku_stats()
    dump('sudoku.json', dict(
        source=dict(check='bench/check.py fit(train-s0, steps=4, lr=8), as final/run_final.py',
                    test='Sudoku-Extreme test (412,786 puzzles), bench/data_io.sudoku_split("test")',
                    engine='bench/engine.py run2: unit pairs, Luby restarts base 10^4, budget 10^7 (final/run_final.py); '
                           'engine.py and check.py hashes asserted against final/ENGINE_SHA256.json',
                    runs='seed = index; moves, ops and restarts asserted equal to the final run (final_sudoku_extreme_s0.npz)',
                    sizes='bench/run_sudoku_sizes.py generator and seeds 7 + i; 25 x 25 puzzles from bench/sudoku25b.json, '
                          'asserted equal to final/sudoku_ladder_25.json',
                    stats='bench/final/sudoku_extreme.json, bench/final/sudoku_ladder_local.json, bench/final/sudoku_ladder_25.json (the final run)'),
        check=chk, demo=demo, extreme=rows, sizes=sizes, stats=stats))


# -------------------------------------------------------------------------------------------------------- Mazes --
def mazes():
    import data_io
    import mazes as M

    res = json.loads((BENCH / 'final' / 'mazes.json').read_text())   # the final run
    tk, y = data_io.maze_unique('train')
    ck, _ = M.fit_check(tk, y)
    th, yh = data_io.maze_hard('train')
    rule, _ = M.fit_rule(th, yh)
    names = ['open', 'nearer', 'level', 'farther', 'unreached']
    assert [names[f] for f in rule['required']] == res['hard_fit']['required']
    chk = dict(theta=[float(x) for x in ck['theta']], kept=[int(r) for r in ck['kept']],
               lo=np.asarray(ck['lo']).tolist(), hi=np.asarray(ck['hi']).tolist())
    rl = dict(w=[float(x) for x in rule['w']], required=[int(f) for f in rule['required']],
              forbidden=[int(f) for f in rule['forbidden']])
    enc = lambda t: ''.join('#.SG'[int(v)] for v in np.asarray(t).ravel())
    bits = lambda b: ''.join('1' if v else '0' for v in np.asarray(b).ravel())

    # Maze-Hard test: 24 for parity, including 106 (the one CRAM's released models stop wrong on)
    tt, yy = data_io.maze_hard('test')
    nb = M.neighbours(30, 30)
    idx = sorted(set([0, 106] + [int(k * 41) for k in range(1, 23)]))
    hard = []
    for i in idx:
        b, o, c = M.solve_hard(rl_np(rule), tt[i], nb=nb)
        hard.append(dict(index=i, tokens=enc(tt[i]), label=bits(yy[i]),
                         py=dict(board=bits(b) if b is not None else None, ops=o, cmp=c,
                                 exact=bool(b is not None and (b == yy[i]).all()))))
    cram106 = []
    for s in range(3):
        d = json.loads((RELEASED / f'run_released_hard_wrong_{s}.json').read_text())['result']
        cram106.append(dict(seed=s, stopped=d['stopped'], diff=[dict(cell=x['cell'], pred=x['pred']) for x in d['diff']]))

    # Maze-Unique test: 20 for parity of the cold derivation
    te, yt = data_io.maze_unique('test')
    uniq = []
    for i in range(20):
        b, o, c, u, good = M.cold_unique(ck, te[i])
        uniq.append(dict(index=i, tokens=enc(te[i]), label=bits(yt[i]),
                         py=dict(board=bits(b) if b is not None else None, ops=o, cmp=c, undecided=u, ok=bool(good))))

    # edge mazes: the generator and seeds of run_mazes.py / modal_cram_check.py job_pcd (h = 15: 29 x 29 without ring)
    edge = []
    pcd = json.loads((RELEASED / 'run_released_pcd_0_15.json').read_text())['result']['variants']['noring']['mazes']
    rng = np.random.default_rng(1000 + 15)
    for i in range(20):
        g = M.tree_maze(15, rng)
        op = np.argwhere(g)
        a, b2 = rng.choice(len(op), 2, replace=False)
        S, G = tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b2])
        g = g[1:-1, 1:-1]; S = (S[0] - 1, S[1] - 1); G = (G[0] - 1, G[1] - 1)
        t = M.tokens(g, S, G)
        lab = M.path_cells_tree(g, S, G)
        b, o, c, u, good = M.cold_unique(ck, t)
        cm = pcd[i]
        assert cm['path_len'] == int(lab.sum()) - 1, 'not the same maze as the CRAM record'
        edge.append(dict(index=i, tokens=enc(t), label=bits(lab), side=int(g.shape[0]),
                         py=dict(board=bits(b) if b is not None else None, ops=o, exact=bool(b is not None and (b == lab).all())),
                         cram=dict(solved=cm['solved'], stopped=cm['stopped'], n_diff=cm['n_diff'], n_diff_edge=cm['n_diff_edge'],
                                   edge_tips=cm['edge_tips'], diff_recorded=[dict(cell=x['cell'], pred=x['pred']) for x in cm['diff']])))

    # room-and-corridor mazes: built in the page (same builder); Python labels and our boards here for parity;
    # CRAM's released Maze-Hard model (seed 0, 1, 2) from run_released_pab_*.json
    pab = {s: {(r['extra'], r['sym'], r['n']): r for r in json.loads((RELEASED / f'run_released_pab_{s}.json').read_text())['result']}
           for s in range(3)}
    rooms = []
    for extra, ns in ((0, [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]), (2, [12, 16, 18, 20, 22, 24, 26])):
        for k in range(8):
            for n in ns:
                g, S, G = M.sym(*M.L_maze(n, n, extra), k)
                lab = M.label_rule(g, S, G)
                t = M.tokens(g, S, G)
                b, o, c = M.solve_hard(rl_np(rule), t)
                rows = {}
                for s in range(3):
                    r = pab[s][(extra, k, n)]
                    rows[str(s)] = dict(equals_label=r['own']['equals_label'], valid_path=r['own']['valid_path'],
                                        length=r['own']['length'], first=r['own']['returned_first'],
                                        stopped=r['own']['stopped'])
                rows_lab = pab[0][(extra, k, n)]
                rooms.append(dict(extra=extra, sym=k, n=n, shape=list(g.shape), S=list(S), G=list(G),
                                  shortest=rows_lab['shortest'], label_first=rows_lab['label_first'],
                                  corridor=rows_lab['corridor'], room=rows_lab['room'],
                                  label=bits(lab), py=dict(board=bits(b) if b is not None else None,
                                                           exact=bool(b is not None and (b == lab).all())),
                                  cram=rows))
    dump('mazes.json', dict(
        source=dict(check='bench/mazes.py fit_check on Maze-Unique train (1,000)', rule='bench/mazes.py fit_rule on Maze-Hard train (1,000)',
                    stats='bench/final/mazes.json', cram106='released/run_released_hard_wrong_{0,1,2}.json',
                    cram_edge='released/run_released_pcd_0_15.json (noring; the record keeps the first 60 differing cells)',
                    cram_rooms='released/run_released_pab_{0,1,2}.json (first step, length and validity; full paths not recorded)'),
        check=chk, rule=rl, stats=res, hard=hard, cram106=cram106, unique=uniq, edge=edge, rooms=rooms))


def rl_np(rule):
    return rule


# ---------------------------------------------------------------------------------------------------------- BFS --
def bfs():
    src = (WALK / 'completed_answer_bfs.py').read_text()
    head = src.split('\nout = {}')[0]  # everything up to the evaluation loop (which would write files)
    ns = {'__name__': 'bfs_export'}
    exec(compile(head, 'completed_answer_bfs.py', 'exec'), ns)
    train, wb, logreg, dataset, run, completion = ns['train'], ns['wb'], ns['logreg'], ns['dataset'], ns['run'], ns['completion']
    X, yv = dataset('order', train)
    gate = [j for j in range(5) if X[yv > 0][:, j].all()]
    t0 = time.perf_counter(); w = logreg(X, yv); fit_ms = (time.perf_counter() - t0) * 1e3
    assert gate == [0, 1], gate
    G = lambda A: [[int(u), int(v)] for u, v in zip(*np.nonzero(np.triu(A, 1)))]
    tr = [dict(n=int(len(A)), s=int(s), edges=G(A), par=ints(par)) for A, s, par in train]
    graphs = ns['graphs']
    par_cases = []
    for n, m in ((16, 12), (64, 8), (256, 3), (1024, 1)):
        for A, s, par in graphs(n, m):
            t0 = time.perf_counter()
            p, loop = run('order', w, wb, A, s, True, gate)
            sec = time.perf_counter() - t0
            par_cases.append(dict(n=n, s=int(s), edges=G(A), par=ints(par), py=dict(par=ints(p), loops=int(loop), sec=sec,
                                                                                    right=int((p == par).sum()) - 1)))
            print('bfs', n, par_cases[-1]['py']['right'], n - 1, loop, f'{sec:.1f}s', flush=True)
    rec = json.loads((WALK / 'completed_answer_bfs.json').read_text())
    dump('bfs.json', dict(
        source=dict(script='walkthrough/completed_answer_bfs.py (rng seed 1; executed up to the evaluation loop)',
                    record='walkthrough/completed_answer_bfs.json', fit_ms_note='fit_ms in record is the logged laptop time'),
        w=[float(x) for x in w], wb=[float(x) for x in wb], gate=gate, train=tr,
        X_shape=list(X.shape), parity=par_cases, record=rec))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('sudoku', 'all'):
        sudoku()
    if what in ('mazes', 'all'):
        mazes()
    if what in ('bfs', 'all'):
        bfs()
