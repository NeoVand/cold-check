"""The final, consistent run: one frozen engine, one budget, every benchmark.

One frozen engine set, checked against final/ENGINE_SHA256.json before anything runs:
- grids with units (Sudoku of every size): engine.solve2, cold stage with unit pairs + Luby restarts (base 10^4);
- N-Queens and colouring (per-value conditions): multi.solve_r, the same cold stage and the same Luby restarts;
- mazes: mazes.propagate_nbr (cold stage) and mazes.fit_rule / solve_hard (clock + path rule), no search needed.
Budget 10^7 moves for every search. Every fit time and operation count below comes from this run.

    python final/run_final.py sudoku | ladder | mazes | multi | proofs     (laptop)
    modal run final/modal_final.py                                         (25 x 25, the same 100 puzzles)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

FINAL = Path(__file__).resolve().parent
BENCH = FINAL.parent
sys.path.insert(0, str(BENCH))
import check  # noqa: E402
import data_io  # noqa: E402
import engine  # noqa: E402
import run_sudoku as RS  # noqa: E402
import run_sudoku_sizes as R  # noqa: E402

BUDGET = 10 ** 7
RESTART = 10 ** 4
FILES = ('check.py', 'engine.py', 'mazes.py', 'multi.py', 'data_io.py', 'run_sudoku.py', 'run_sudoku_sizes.py',
         'run_mazes.py', 'run_multi.py', 'proofs.py', 'final/run_final.py', 'final/modal_final.py')


def shas(root=BENCH):
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in FILES}


def verify(root=BENCH):
    want = json.loads((root / 'final' / 'ENGINE_SHA256.json').read_text())
    got = shas(root)
    assert got == want, {k: (got[k], want.get(k)) for k in got if got[k] != want.get(k)}


def solve_final(E, dom0, seed):
    return engine.run2(E, dom0, seed=seed, budget=BUDGET, use_pairs=True, restart_base=RESTART)


def sudoku():
    out = dict(full_all_hash_ok=RS.verify_full_all(), budget=BUDGET, restart_base=RESTART, fits=[])
    ts, tc, tr = data_io.sudoku_split('test')
    assert len(ts) == 412786
    puz = np.where(tc, ts, 0).astype(np.int64)
    for k in (0, 1, 2):
        sol, _, _ = data_io.sudoku_split(f'train-s{k}')
        sol = sol.astype(np.int64)
        steps, ref = RS.choose_steps(sol)
        times = []
        for _ in range(8):
            ck, dt = check.fit(sol, steps=steps, lr=32.0 / steps)
            times.append(dt)
        E = engine.build(ck, 3)
        solve_final(E, engine.masks(puz[0], 9), 0)  # compile
        n = len(puz)
        st = np.zeros(n, np.int8); ok = np.zeros(n, bool); mv = np.zeros(n, np.int64); ops = np.zeros(n, np.int64)
        cmp = np.zeros(n, np.int64); cold = np.zeros(n, bool); wall = np.zeros(n, np.float32); rs = np.zeros(n, np.int64)
        t0 = time.perf_counter()
        for i in range(n):
            t1 = time.perf_counter()
            dom, o = solve_final(E, engine.masks(puz[i], 9), i + 1000003 * k)
            wall[i] = time.perf_counter() - t1
            st[i] = o[0]; mv[i] = o[1]; ops[i] = o[2]; cmp[i] = o[3]; cold[i] = o[4]; rs[i] = o[6]
            ok[i] = o[0] == 1 and (engine.decode(dom) == ts[i]).all()
        np.savez_compressed(BENCH / 'data' / f'final_sudoku_extreme_s{k}.npz', status=st, correct=ok, moves=mv, ops=ops,
                            cmp=cmp, cold=cold, wall=wall, restarts=rs)
        f = dict(split=f'train-s{k}', steps=steps, lr=32.0 / steps, fit_ms_cold=times[0] * 1e3,
                 fit_ms_warm_median7=float(np.median(times[1:])) * 1e3, n_kept_types=int((ck['w'] > 0).sum()),
                 relations=[[check.FNAMES[i] for i in fs] for fs in ck['relations']], tau=ck['tau'], wmin=ck['wmin'],
                 n_test=n, solved=int(ok.sum()), wrong_stops=int(((st == 1) & ~ok).sum()), budget_misses=int((st == 0).sum()),
                 no_solution=int((st == -1).sum()), cold_only=int(cold.sum()), cold_only_frac=float(cold.mean()),
                 moves_mean=float(mv.mean()), moves_median=float(np.median(mv)), moves_max=int(mv.max()),
                 restarts_total=int(rs.sum()), ops_mean=float(ops.mean()), ops_median=float(np.median(ops)),
                 ops_max=int(ops.max()), ops_plus_cmp_mean=float((ops + cmp).mean()), ops_plus_cmp_max=int((ops + cmp).max()),
                 wall_ms_mean=float(wall.mean() * 1e3), wall_ms_max=float(wall.max() * 1e3),
                 wall_total_s=time.perf_counter() - t0)
        out['fits'].append(f)
        print(json.dumps(f), flush=True)
    (FINAL / 'sudoku_extreme.json').write_text(json.dumps(out, indent=1))


def ladder_one(b, i, tb, ck):
    """Puzzle i of block size b, generated exactly as run_sudoku_sizes.one (same seed, same generator), solved by the
    final engine."""
    E = engine.build(ck, b)
    rng = np.random.default_rng([1000 + b, i])
    t0 = time.perf_counter()
    puz, sol = R.make_puzzle(E, b, rng, tb)
    gen = time.perf_counter() - t0
    t1 = time.perf_counter()
    dom, o = solve_final(E, engine.masks(puz, b * b), 7 + i)
    w = time.perf_counter() - t1
    x = engine.decode(dom)
    good = bool(o[0] == 1 and R.valid(x, b) and (x[puz > 0] == puz[puz > 0]).all())
    return dict(i=i, good=good, equals=bool((x == sol).all()), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]),
                cold=bool(o[4]), status=int(o[0]), restarts=int(o[6]), wall=w, gen=gen, blank=float((puz == 0).mean()))


def ladder():
    ck = R.the_check()
    res = dict(budget=BUDGET, restart_base=RESTART)
    for b, N in ((2, 1000), (3, 200), (4, 1000)):
        rows = [ladder_one(b, i, None, ck) for i in range(N)]
        s = R.summarise(b, rows)
        s['restarts_total'] = int(sum(r['restarts'] for r in rows))
        s['wrong'] = int(sum(r['status'] == 1 and not r['good'] for r in rows))
        res[f'side{b * b}'] = s
        print(json.dumps(s), flush=True)
    (FINAL / 'sudoku_ladder_local.json').write_text(json.dumps(res, indent=1))


def mazes():
    import run_mazes
    run_mazes.main(FINAL / 'mazes.json')


def multi():
    import run_multi
    run_multi.main(FINAL / 'multi.json', budget=BUDGET, restart_base=RESTART)


def proofs():
    import proofs as P
    print(json.dumps(P.main(FINAL / 'proofs.json'), default=str)[:2000])


if __name__ == '__main__':
    verify()
    for part in sys.argv[1:]:
        globals()[part]()
