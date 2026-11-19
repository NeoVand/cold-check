"""Export the fitted rules, test instances and the Python outputs the page and its parity tests need.

Reads (never writes) code/{bench,released,walkthrough} of this repository (run code/fetch.py first); writes
site/static/data/*.json.
Run (from anywhere; one heavy CPU job at a time on the laptop):
    NUMBA_CACHE_DIR=<tmp> PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    uv run --no-project --with numpy==2.2.6 --with numba==0.61.2 python export_data.py [sudoku|mazes|bfs|all]
Every number shown on the page comes from these files; each file carries a `source` block naming its inputs.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2] / 'code'
BENCH = CODE / 'bench'
RELEASED = CODE / 'released'   # CRAM's released models, run by released/modal_cram_check.py
WALK = CODE / 'walkthrough'
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


def side25_pairs(rows):
    """The 25 x 25 card numbers from bench/sudoku25b.json, arm 'pairs' (the engine with unit pairs)."""
    r = [x['pairs'] for x in rows]
    mv = sorted(x['moves'] for x in r)
    return dict(side=25, n=len(r), solved_valid=sum(x['good'] for x in r), equals_generated=sum(x['equals'] for x in r),
                budget_misses=sum(x['status'] == 0 for x in r), wrong_stops=sum(x['status'] == 1 and not x['good'] for x in r),
                moves_median=float(mv[len(mv) // 2]), moves_max=mv[-1], arm='pairs')


def sudoku():
    import check
    import data_io
    import engine
    import run_sudoku_sizes as R

    check.vocabulary()
    sol9, _, _ = data_io.sudoku_split('train-s0')
    ck, _ = check.fit(sol9.astype(np.int64), steps=4, lr=8.0)  # the train-s0 fit of run_sudoku.py (steps 4, lr 8)
    summary = json.loads((BENCH / 'sudoku_extreme.json').read_text())
    f0 = summary['fits'][0]
    assert f0['split'] == 'train-s0' and f0['steps'] == 4
    chk = dict(types=ints(ck['types']), w=[float(x) for x in ck['w']], relations=[list(fs) for fs in ck['relations']],
               relation_names=[[check.FNAMES[i] for i in fs] for fs in ck['relations']],
               conditions=[[bool(a), bool(b)] for a, b in ck['conditions']], tau=float(ck['tau']),
               kept=check.describe(ck))
    assert [[check.FNAMES[i] for i in fs] for fs in ck['relations']] == f0['relations']

    # test puzzles: 24 spread over the 412,786 (parity), plus named demo puzzles chosen from the recorded per-puzzle run
    ts, tc, tr = data_io.sudoku_split('test')
    rec = np.load(BENCH / 'data' / 'sudoku_extreme_s0.npz')
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
    engine.run(E, np.where(tc[0], ts[0], 0).astype(np.int64))  # compile
    rows = []
    for i in sorted(set(spread) | set(demo.values())):
        puz = np.where(tc[i], ts[i], 0).astype(np.int64)
        dom, o = engine.run(E, puz, seed=i + 1000003 * 0, budget=10 ** 6)  # run_sudoku.py's seed for split 0
        x = engine.decode(dom)
        assert int(o[1]) == int(rec['moves'][i]) and int(o[2]) == int(rec['ops'][i]), ('not the recorded run', i)
        rows.append(dict(index=i, rating=int(tr[i]), puzzle=ints(puz), solution=ints(ts[i]), seed=i,
                         py=dict(status=int(o[0]), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]), cold=bool(o[4]),
                                 maxdepth=int(o[5]), board=ints(x), correct=bool((x == ts[i]).all()))))

    # other sizes: the benchmark's own generator and seeds (run_sudoku_sizes.py), solved with the 9 x 9 fit
    sizes = {}
    for b, N in ((2, 20), (4, 20)):
        Eb = engine.build(ck, b)
        out = []
        for i in range(N):
            rng = np.random.default_rng([1000 + b, i])
            puz, solb = R.make_puzzle(Eb, b, rng, None)
            dom, o = R.solve_masks(Eb, engine.masks(puz, b * b), 7 + i, 10 ** 6)
            x = engine.decode(dom)
            out.append(dict(index=i, puzzle=ints(puz), solution=ints(solb), seed=7 + i,
                            py=dict(status=int(o[0]), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]), cold=bool(o[4]),
                                    maxdepth=int(o[5]), board=ints(x), correct=bool(R.valid(x, b)))))
            print('side', b * b, i, out[-1]['py']['moves'], flush=True)
        sizes[str(b * b)] = out
    # 25 x 25: the puzzles of the S6b rerun (sudoku25b.json) if it has finished; their original-engine runs are the
    # first arm and must match the original 100-puzzle run (sudoku_sizes.json side25_rows) move for move.
    p25 = BENCH / 'sudoku25b.json'
    s25 = []
    if p25.exists():
        d25 = json.loads(p25.read_text())
        orig = {r['i']: r for r in json.loads((BENCH / 'sudoku_sizes.json').read_text())['side25_rows']}
        for r in d25['rows']:
            if r['orig']['good'] and r['i'] in orig:
                assert r['orig']['moves'] == orig[r['i']]['moves']
            s25.append(dict(index=r['i'], puzzle=r['puzzle'], seed=7 + r['i'],
                            py=dict(orig=r['orig'], pairs=r['pairs'], pairs_restarts=r['pairs_restarts'])))
    else:
        # until the rerun lands: regenerate two of the original 100 (same generator and seed, so the same puzzles) and
        # check the solve against the recorded row move for move
        E25 = engine.build(ck, 5)
        orig = {r['i']: r for r in json.loads((BENCH / 'sudoku_sizes.json').read_text())['side25_rows']}
        for i in (5, 9):
            rng = np.random.default_rng([1005, i])
            puz, solb = R.make_puzzle(E25, 5, rng, 0.55)
            dom, o = R.solve_masks(E25, engine.masks(puz, 25), 7 + i, 10 ** 6)
            x = engine.decode(dom)
            assert int(o[1]) == orig[i]['moves'], ('not the recorded 25 x 25 puzzle', i)
            s25.append(dict(index=i, puzzle=ints(puz), solution=ints(solb), seed=7 + i,
                            py=dict(orig=dict(status=int(o[0]), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]),
                                              cold=bool(o[4]), board=ints(x), good=bool(R.valid(x, 5))))))
            print('side 25', i, int(o[1]), flush=True)
    sizes['25'] = s25

    # every number the Sudoku card shows
    stats = final_sudoku_stats()
    dump('sudoku.json', dict(
        source=dict(check='bench/check.py fit(train-s0, steps=4, lr=8), as run_sudoku.py',
                    test='Sudoku-Extreme test (412,786 puzzles), bench/data_io.sudoku_split("test")',
                    runs='bench/engine.py run(seed = index); moves and ops asserted equal to bench/data/sudoku_extreme_s0.npz',
                    sizes='bench/run_sudoku_sizes.py generator and seeds; 25 x 25 from bench/sudoku25b.json, or puzzles 5 and 9 regenerated until it lands',
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
