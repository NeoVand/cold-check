"""An extra that was not registered: the second engine's unit-pair rule (engine.py) on all 412,786 Sudoku-Extreme test
puzzles with the train-s0 fit, no restarts. Writes sudoku_extreme_pairs.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_io  # noqa: E402
import engine  # noqa: E402
import run_sudoku_sizes as R  # noqa: E402

ck = R.the_check()
E = engine.build(ck, 3)
ts, tc, tr = data_io.sudoku_split('test')
puz = np.where(tc, ts, 0).astype(np.int64)
n = len(puz)
ok = np.zeros(n, bool); cold = np.zeros(n, bool); mv = np.zeros(n, np.int64); ops = np.zeros(n, np.int64); cmp = np.zeros(n, np.int64)
engine.run2(E, engine.masks(puz[0], 9))
t0 = time.perf_counter()
for i in range(n):
    dom, o = engine.run2(E, engine.masks(puz[i], 9), seed=i, budget=10 ** 6, use_pairs=True, restart_base=0)
    ok[i] = o[0] == 1 and (engine.decode(dom) == ts[i]).all(); cold[i] = o[4]; mv[i] = o[1]; ops[i] = o[2]; cmp[i] = o[3]
tot = time.perf_counter() - t0
r = dict(n=n, solved=int(ok.sum()), cold_only=int(cold.sum()), cold_only_frac=float(cold.mean()), moves_mean=float(mv.mean()),
         moves_median=float(np.median(mv)), moves_max=int(mv.max()), ops_mean=float(ops.mean()), ops_plus_cmp_mean=float((ops + cmp).mean()),
         wall_ms_mean=tot / n * 1e3)
(HERE / 'sudoku_extreme_pairs.json').write_text(json.dumps(r, indent=1))
print(json.dumps(r))
