"""Sudoku-Extreme on CRAM's protocol: one fit per training split (train-s0, s1, s2), each scored once on the 412,786
test puzzles. Writes sudoku_extreme.json (summary) and data/sudoku_extreme_s<k>.npz (per puzzle).

Fit setting rule, fixed before testing and using training data only: steps from {1, 2, 4, 8, 16, 32} with lr = 32 /
steps, the fewest steps whose kept pair types equal those of the 32-step fit on the same split.
Usage: python run_sudoku.py [n_test]   (n_test limits the test puzzles, for a smoke run)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
import data_io  # noqa: E402
import engine  # noqa: E402


def verify_full_all():
    """The test puzzles are CRAM's: rebuild full-all (dev10k, FRM test, rest in official order) and check its hash."""
    idx = json.loads(data_io.SPLITS.read_text())
    s, c, g = data_io._read_csv(data_io.DATA / 'sudoku-extreme' / 'test.csv')
    dev = np.array(idx['files']['frm/dev10k/dev.npz']['rows'])
    frm = np.array(idx['files']['frm/split/test.npz']['rows'])
    rest = np.setdiff1d(np.arange(len(s)), np.concatenate([dev, frm]))
    every = np.concatenate([dev, frm, rest])
    ok = data_io._sha(s[every], c[every], g[every]) == idx['full_all_content_sha256']
    return bool(ok)


def choose_steps(sol):
    ref, _ = check.fit(sol, steps=32, lr=1.0)
    for st in (1, 2, 4, 8, 16, 32):
        ck, _ = check.fit(sol, steps=st, lr=32.0 / st)
        if set(ck['kept_types'].tolist()) == set(ref['kept_types'].tolist()):
            return st, ref
    return 32, ref


def main():
    ntest = int(sys.argv[1]) if len(sys.argv) > 1 else None
    check.vocabulary()
    out = dict(full_all_hash_ok=verify_full_all(), fits=[])
    ts, tc, tr = data_io.sudoku_split('test')
    assert len(ts) == 412786, len(ts)
    if ntest:
        ts, tc, tr = ts[:ntest], tc[:ntest], tr[:ntest]
    puz = np.where(tc, ts, 0).astype(np.int64)
    for k in (0, 1, 2):
        sol, _, _ = data_io.sudoku_split(f'train-s{k}')
        sol = sol.astype(np.int64)
        steps, ref = choose_steps(sol)
        times = []
        for rep in range(8):
            ck, dt = check.fit(sol, steps=steps, lr=32.0 / steps)
            times.append(dt)
        E = engine.build(ck, 3)
        engine.run(E, puz[0])  # compile
        n = len(puz)
        st = np.zeros(n, np.int8); ok = np.zeros(n, bool); mv = np.zeros(n, np.int64); ops = np.zeros(n, np.int64)
        cmp = np.zeros(n, np.int64); cold = np.zeros(n, bool); wall = np.zeros(n, np.float32)
        t0 = time.perf_counter()
        for i in range(n):
            t1 = time.perf_counter()
            dom, o = engine.run(E, puz[i], seed=i + 1000003 * k, budget=10 ** 6)
            wall[i] = time.perf_counter() - t1
            st[i] = o[0]; mv[i] = o[1]; ops[i] = o[2]; cmp[i] = o[3]; cold[i] = o[4]
            ok[i] = o[0] == 1 and (engine.decode(dom) == ts[i]).all()
        tot = time.perf_counter() - t0
        np.savez_compressed(HERE / 'data' / f'sudoku_extreme_s{k}.npz', status=st, correct=ok, moves=mv, ops=ops, cmp=cmp,
                            cold=cold, wall=wall, rating=tr)
        f = dict(split=f'train-s{k}', steps=steps, lr=32.0 / steps,
                 fit_ms_cold=times[0] * 1e3, fit_ms_warm_median7=float(np.median(times[1:])) * 1e3,
                 kept=check.describe(ck), n_kept_types=int((ck['w'] > 0).sum()), n_types=int(len(ck['types'])),
                 relations=[[check.FNAMES[i] for i in fs] for fs in ck['relations']],
                 conditions=ck['conditions'], tau=ck['tau'], wmin=ck['wmin'],
                 ref32_kept_types=int((ref['w'] > 0).sum()),
                 n_test=n, solved=int((st == 1).sum()), correct=int(ok.sum()), wrong_stops=int(((st == 1) & ~ok).sum()),
                 budget_misses=int((st == 0).sum()), no_solution=int((st == -1).sum()),
                 cold_only=int(cold.sum()), cold_only_frac=float(cold.mean()),
                 moves_mean=float(mv.mean()), moves_median=float(np.median(mv)), moves_max=int(mv.max()),
                 ops_mean=float(ops.mean()), ops_median=float(np.median(ops)), ops_max=int(ops.max()),
                 ops_plus_cmp_mean=float((ops + cmp).mean()), ops_plus_cmp_max=int((ops + cmp).max()),
                 wall_ms_mean=float(wall.mean() * 1e3), wall_ms_max=float(wall.max() * 1e3), wall_total_s=tot)
        out['fits'].append(f)
        print(json.dumps({k2: v for k2, v in f.items() if k2 != 'kept'}), flush=True)
    name = 'sudoku_extreme.json' if not ntest else 'sudoku_extreme_smoke.json'
    (HERE / name).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
