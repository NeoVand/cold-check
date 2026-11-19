"""Modal app cold-check-bench: the 25 x 25 Sudoku runs, CPU jobs too slow for a laptop.
Only our own code (this folder's .py files) and the train-s0 split cache (data/sudoku_train-s0.npz, written by
data_io.sudoku_split after fetch.py) go into the image; nothing third-party runs.

    modal run modal_bench.py::sudoku25 --n 100      # the registered run  -> sudoku_sizes.json (side25)
    modal run modal_bench.py::sudoku25b --n 100     # the registered retest with unit pairs  -> sudoku25b.json
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App('cold-check-bench')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.2.6', 'numba==0.61.2')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'NUMBA_CACHE_DIR': '/tmp/numba'})
         .add_local_file(HERE / 'data' / 'sudoku_train-s0.npz', '/root/bench/data/sudoku_train-s0.npz', copy=True)
         .add_local_dir(HERE, '/root/bench', ignore=['data', 'data/**', '__pycache__', '*.json', '*.log', '*.md', '*.npz']))


@app.function(image=image, cpu=1.0, memory=1024, timeout=3600)
def one25(i: int, blank: float = 0.55):
    import sys
    sys.path.insert(0, '/root/bench')
    import run_sudoku_sizes as R
    r = R.one(5, i, blank)
    r['i'] = i
    return r


@app.local_entrypoint()
def sudoku25(n: int = 100, blank: float = 0.55):
    import numpy as np
    import sys
    sys.path.insert(0, str(HERE))
    rows = list(one25.map(range(n), kwargs={'blank': blank}, order_outputs=True, return_exceptions=True))
    bad = [r for r in rows if not isinstance(r, dict)]
    rows = [r for r in rows if isinstance(r, dict)]
    import run_sudoku_sizes as R
    summ = R.summarise(5, rows)
    summ['errors'] = len(bad)
    summ['cpu_s_total'] = float(sum(r['gen'] + r['wall'] for r in rows))
    p = HERE / 'sudoku_sizes.json'
    res = json.loads(p.read_text()) if p.exists() else {}
    res['side25'] = summ
    res['side25_rows'] = rows
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(summ))


@app.function(image=image, cpu=1.0, memory=1024, timeout=3600)
def one25b(i: int, blank: float = 0.55):
    """S6b: the same puzzle as one25 (same seed), solved three ways: original engine, unit pairs, pairs + restarts."""
    import sys, time
    sys.path.insert(0, '/root/bench')
    import numpy as np
    import engine
    import run_sudoku_sizes as R
    ck = R.the_check()
    E = engine.build(ck, 5)
    rng = np.random.default_rng([1005, i])
    puz, sol = R.make_puzzle(E, 5, rng, blank)
    res = dict(i=i, puzzle=puz.tolist(), blank=float((puz == 0).mean()))
    for name, up, rb in (('orig', None, None), ('pairs', True, 0), ('pairs_restarts', True, 10 ** 4)):
        t0 = time.perf_counter()
        if up is None:
            dom, o = R.solve_masks(E, engine.masks(puz, 25), 7 + i, 10 ** 6)
        else:
            dom, o = engine.run2(E, engine.masks(puz, 25), seed=7 + i, budget=10 ** 6, use_pairs=up, restart_base=rb)
        w = time.perf_counter() - t0
        x = engine.decode(dom)
        good = bool(o[0] == 1 and R.valid(x, 5) and (x[puz > 0] == puz[puz > 0]).all())
        res[name] = dict(good=good, equals=bool((x == sol).all()), status=int(o[0]), moves=int(o[1]), ops=int(o[2]),
                         cmp=int(o[3]), cold=bool(o[4]), wall=w, restarts=int(o[6]) if up else 0)
    return res


@app.local_entrypoint()
def sudoku25b(n: int = 100, blank: float = 0.55):
    rows = list(one25b.map(range(n), kwargs={'blank': blank}, order_outputs=True, return_exceptions=True))
    bad = [repr(r) for r in rows if not isinstance(r, dict)]
    rows = [r for r in rows if isinstance(r, dict)]
    (HERE / 'sudoku25b.json').write_text(json.dumps(dict(errors=bad, rows=rows)))
    for name in ('orig', 'pairs', 'pairs_restarts'):
        g = [r[name]['good'] for r in rows]
        mv = sorted(r[name]['moves'] for r in rows)
        print(name, 'solved', sum(g), 'of', len(rows), 'median moves', mv[len(mv) // 2], 'max', mv[-1])
