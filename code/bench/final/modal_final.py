"""Modal app cold-check-bench-final: the 25 x 25 rung of the final run (the same 100 puzzles as before), final engine,
budget 10^7. Only our own .py files and the train-s0 cache go into the image.

    cd bench && modal run final/modal_final.py
"""
import json
from pathlib import Path

import modal

FINAL = Path(__file__).resolve().parent
BENCH = FINAL.parent
app = modal.App('cold-check-bench-final')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.2.6', 'numba==0.61.2')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'NUMBA_CACHE_DIR': '/tmp/numba'})
         .add_local_file(BENCH / 'data' / 'sudoku_train-s0.npz', '/root/bench/data/sudoku_train-s0.npz', copy=True)
         .add_local_file(FINAL / 'ENGINE_SHA256.json', '/root/bench/final/ENGINE_SHA256.json', copy=True)
         .add_local_dir(BENCH, '/root/bench', ignore=['data', 'data/**', '__pycache__', '*.json', '*.log', '*.md', '*.npz',
                                                       'final/*.json']))


@app.function(image=image, cpu=1.0, memory=2048, timeout=6 * 3600)
def one(i: int):
    import sys
    sys.path.insert(0, '/root/bench'); sys.path.insert(0, '/root/bench/final')
    import json as _j
    from pathlib import Path as _P
    import run_final as F
    F.verify(_P('/root/bench'))
    import run_sudoku_sizes as R
    return F.ladder_one(5, i, 0.55, R.the_check())


@app.local_entrypoint()
def main(n: int = 100):
    import sys
    sys.path.insert(0, str(BENCH))
    rows = list(one.map(range(n), order_outputs=True, return_exceptions=True))
    bad = [repr(r) for r in rows if not isinstance(r, dict)]
    rows = [r for r in rows if isinstance(r, dict)]
    import run_sudoku_sizes as R
    s = R.summarise(5, rows) if rows else {}
    s['errors'] = bad
    s['wrong'] = int(sum(r['status'] == 1 and not r['good'] for r in rows))
    s['restarts_total'] = int(sum(r['restarts'] for r in rows))
    s['budget'] = 10 ** 7
    s['cpu'] = 'Modal CPU container, 1 core'
    (FINAL / 'sudoku_ladder_25.json').write_text(json.dumps(dict(summary=s, rows=rows), indent=1))
    print(json.dumps(s))
