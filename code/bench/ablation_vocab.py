"""Registered test A1: the Sudoku fit with the vocabulary widened by 12 distractor functions. Writes ablation_vocab.json."""
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
import run_sudoku as RS  # noqa: E402
import run_sudoku_sizes as R  # noqa: E402

check.set_extra(True)
t0 = time.perf_counter(); check.vocabulary(); vocab_s = time.perf_counter() - t0
sol, _, _ = data_io.sudoku_split('train-s0')
sol = sol.astype(np.int64)
steps, ref = RS.choose_steps(sol)
T = check.pair_types(3)
realised = set(np.unique(T[T >= 0]).tolist())
rcb = {t for t in realised if (t & 1) or (t & 2) or ((t & 4) and (t & 8))}
ts, tc, _ = data_io.sudoku_split('test')
puz = np.where(tc, ts, 0).astype(np.int64)
res = dict(n_functions=len(check.FNAMES), n_types_realised_9x9=len(realised), n_rcb_types=len(rcb), steps=steps,
           vocabulary_build_s=vocab_s)
for name, hf in (('A1', False), ('A1b', True)):
    times = []
    for _ in range(8):
        ck, dt = check.fit(sol, steps=steps, lr=32.0 / steps, held_filter=hf)
        times.append(dt)
    kept = set(ck['kept_types'].tolist())
    out = dict(n_kept=len(kept), kept_equals_rcb=kept == rcb, distractor_only_kept=len(kept - rcb), rcb_dropped=len(rcb - kept),
               relations=[[check.FNAMES[i] for i in fs] for fs in ck['relations']], conditions=ck['conditions'],
               fit_ms_cold=times[0] * 1e3, fit_ms_warm_median7=float(np.median(times[1:])) * 1e3, wmin=ck['wmin'], tau=ck['tau'])
    E = engine.build(ck, 3)
    good = 0
    for i in range(10000):
        dom, o = engine.run(E, puz[i], seed=i)
        good += int(o[0] == 1 and (engine.decode(dom) == ts[i]).all())
    out['test_first_10000_solved'] = good
    g16 = 0
    for i in range(100):
        g16 += int(R.one(4, i, None, ck)['good'])
    out['side16_first_100_solved'] = g16
    res[name] = out
    print(name, json.dumps(out), flush=True)
(HERE / 'ablation_vocab.json').write_text(json.dumps(res, indent=1))
