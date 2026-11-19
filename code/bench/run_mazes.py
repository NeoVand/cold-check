"""Maze-Unique and Maze-Hard in our setting. Writes mazes.json.

Maze-Unique: the check (mazes.fit_check, l1 = 1e-3 as for Sudoku) fitted on the 1,000 training mazes, the cold
derivation alone on the 1,000 test mazes; then the same check on tree mazes of 31, 61, 121 a side with a wall ring and
the same mazes without it (29, 59, 119; dead ends at the board's edge), the mazes of released/modal_cram_check.py job_pcd
(same generator, same seeds), on which CRAM's released model scored 600/600 and 0/600.
Maze-Hard: the path rule (mazes.fit_rule) fitted on the 1,000 training mazes, read cold on the 1,000 test mazes, on the
room-and-corridor mazes of job_pab (8 symmetries, rooms 3-14 with an equal corridor, 12-26 with a corridor two steps
shorter) and on the looped mazes of job_loops (sides 31, 61, 121, 100 each, same seeds). The labels of generated
mazes come from the benchmark's label rule (mazes.label_rule), never from the solver.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_io  # noqa: E402
import mazes as M  # noqa: E402


def timed_fit(f, *a):
    ts = []
    for _ in range(8):
        r, dt = f(*a)
        ts.append(dt)
    return r, ts[0] * 1e3, float(np.median(ts[1:])) * 1e3


def main(out_path=None):
    out = {}
    # ------------------------------------------------------------------ Maze-Unique
    tk, y = data_io.maze_unique('train')
    ck, cold_ms, warm_ms = timed_fit(M.fit_check, tk, y)
    out['unique_fit'] = dict(theta=ck['theta'].tolist(), kept=[M.RELNAMES[r] for r in ck['kept']],
                             ranges={f'{tn}/{an}/{M.RELNAMES[r]}': [int(ck['lo'][t, a, r]), int(ck['hi'][t, a, r])]
                                     for t, tn in enumerate(('wall', 'open', 'S', 'G')) for a, an in enumerate(('off', 'on'))
                                     for r in ck['kept'] if t > 0 and not (t >= 2 and a == 0)},
                             fit_ms_cold=cold_ms, fit_ms_warm_median7=warm_ms)
    te, yt = data_io.maze_unique('test')
    rows = []
    for t, l in zip(te, yt):
        b, o, c, u, good = M.cold_unique(ck, t)
        rows.append((b is not None and bool((b == l).all()) and M.passes(ck, t, b), u, o, c))
    out['unique_test'] = dict(n=len(rows), exact=int(sum(r[0] for r in rows)), undecided_mazes=int(sum(r[1] > 0 for r in rows)),
                              ops_mean=float(np.mean([r[2] for r in rows])), ops_plus_cmp_mean=float(np.mean([r[2] + r[3] for r in rows])))
    print('unique test', out['unique_test'], flush=True)
    gen = {}
    for h in (15, 30, 60):
        rng = np.random.default_rng(1000 + h)
        mz = []
        for i in range(200):
            g = M.tree_maze(h, rng)
            op = np.argwhere(g)
            a, b2 = rng.choice(len(op), 2, replace=False)
            mz.append((g, tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b2])))
        for var in ('ring', 'noring'):
            ok, edge_tips, mazes_with_tip, ops = 0, 0, 0, []
            for g, S, G in mz:
                if var == 'noring':
                    g = g[1:-1, 1:-1]; S = (S[0] - 1, S[1] - 1); G = (G[0] - 1, G[1] - 1)
                t = M.tokens(g, S, G)
                lab = M.path_cells_tree(g, S, G)
                b, o, c, u, good = M.cold_unique(ck, t)
                ok += int(b is not None and (b == lab).all() and M.passes(ck, t, b))
                ops.append(o)
                n = np.zeros(g.shape, int)
                n[1:] += g[:-1]; n[:-1] += g[1:]; n[:, 1:] += g[:, :-1]; n[:, :-1] += g[:, 1:]
                edge = np.zeros(g.shape, bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
                tips = int((g & (n == 1) & (t == 1) & edge).sum())
                edge_tips += tips; mazes_with_tip += tips > 0
            gen[f'{g.shape[0]}_{var}'] = dict(side=int(g.shape[0]), ring=var == 'ring', n=200, exact=ok,
                                              mazes_with_edge_dead_end=int(mazes_with_tip), edge_dead_ends=int(edge_tips),
                                              ops_mean=float(np.mean(ops)))
            print(var, g.shape, gen[f'{g.shape[0]}_{var}'], flush=True)
    out['unique_generated'] = gen
    # ------------------------------------------------------------------ Maze-Hard
    th, yh = data_io.maze_hard('train')
    rule, cold_ms, warm_ms = timed_fit(M.fit_rule, th, yh)
    names = ['open', 'nearer', 'level', 'farther', 'unreached'] + [f'dir:{d}' for d in M.DIRNAMES]
    out['hard_fit'] = dict(w=dict(zip(names, rule['w'].round(4).tolist())), required=[names[f] for f in rule['required']],
                           forbidden=[names[f] for f in rule['forbidden']], n_steps=rule['n_steps'],
                           direction_order=[M.DIRNAMES[j] for j in np.argsort(-rule['w'][5:])],
                           n_patterns=rule['n_patterns'],
                           fit_ms_cold=cold_ms, fit_ms_warm_median7=warm_ms)
    print('hard fit', out['hard_fit'], flush=True)
    tt, yy = data_io.maze_hard('test')
    nb = M.neighbours(30, 30)
    res = []
    for t, l in zip(tt, yy):
        b, o, c = M.solve_hard(rule, t, nb=nb)
        res.append((b is not None and bool((b == l).all()), o, c))
    wrong = [i for i, r in enumerate(res) if not r[0]]
    out['hard_test'] = dict(n=len(res), exact=int(sum(r[0] for r in res)), wrong=wrong, maze106_exact=bool(res[106][0]),
                            ops_mean=float(np.mean([r[1] for r in res])), ops_plus_cmp_mean=float(np.mean([r[1] + r[2] for r in res])))
    print('hard test', out['hard_test'], flush=True)
    # the same Maze-Hard rule on the Maze-Unique test set (one maze model for both)
    ok = 0
    for t, l in zip(te, yt):
        b, o, c = M.solve_hard(rule, t, nb=nb)
        ok += int(b is not None and (b == l).all())
    out['hard_rule_on_unique_test'] = dict(n=len(te), exact=ok)
    # rooms
    rooms = []
    for extra, ns in ((0, [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]), (2, [12, 16, 18, 20, 22, 24, 26])):
        for k in range(8):
            for n in ns:
                g, S, G = M.sym(*M.L_maze(n, n, extra), k)
                lab = M.label_rule(g, S, G)
                b, o, c = M.solve_hard(rule, M.tokens(g, S, G))
                rooms.append(dict(extra=extra, sym=k, n=n, exact=bool(b is not None and (b == lab).all())))
    out['hard_rooms'] = dict(n=len(rooms), exact=int(sum(r['exact'] for r in rooms)),
                             wrong=[r for r in rooms if not r['exact']])
    print('rooms', out['hard_rooms']['n'], out['hard_rooms']['exact'], flush=True)
    # looped mazes
    loops = {}
    for h in (15, 30, 60):
        rng = np.random.default_rng(5000 + h)
        ok, ops, tsec = 0, [], 0.0
        for i in range(100):
            g = M.tree_maze(h, rng)
            Hh = g.shape[0]
            cand = [(r, c) for r in range(1, Hh - 1) for c in range(1, Hh - 1)
                    if not g[r, c] and ((g[r - 1, c] and g[r + 1, c]) or (g[r, c - 1] and g[r, c + 1]))]
            pick = rng.choice(len(cand), int(round(0.05 * len(cand))), replace=False)
            for j in pick:
                g[cand[j]] = True
            op = np.argwhere(g)
            a, b2 = rng.choice(len(op), 2, replace=False)
            S, G = tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b2])
            lab = M.label_rule(g, S, G)
            t0 = time.perf_counter()
            b, o, c = M.solve_hard(rule, M.tokens(g, S, G))
            tsec += time.perf_counter() - t0
            ok += int(b is not None and (b == lab).all())
            ops.append(o)
        loops[str(2 * h + 1)] = dict(n=100, exact=ok, ops_mean=float(np.mean(ops)), sec_total=tsec)
        print('loops', 2 * h + 1, loops[str(2 * h + 1)], flush=True)
    out['hard_loops'] = loops
    Path(out_path or HERE / 'mazes.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
