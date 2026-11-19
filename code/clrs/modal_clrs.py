"""Modal app cold-check-clrs: the final CLRS-30 runs of the fit-at-the-answer setting on CPU (one task per
container, one BLAS thread, so fit times are one core). Only our code, the library's scorer copy (witness_clrs
tasks.py and scoring.py) and the official split files go into the image.

    modal run modal_clrs.py --tasks bfs,mst_prim --big 256:32,1024:8
Results land in results/<task>.json (local entrypoint writes them).
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / 'clrs30'   # the official splits
app = modal.App('cold-check-clrs')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.3.4', 'scipy>=1.14,<1.17')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'CLRS_DATA': '/root/data',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_dir(DATA, '/root/data', copy=True)
         .add_local_dir(HERE, '/root/clrs', ignore=['clrs30', 'clrs30/**', 'results*', 'results*/**', 'sweep', 'sweep/**', 'logs', 'logs/**', '__pycache__', '*.md', '*.json', '*.sh']))


@app.function(image=image, cpu=2.0, memory=8192, timeout=6 * 3600)
def run_task(task: str, big: str, per_size: int = 20, seed: int = 101, val: bool = False):
    import sys, os
    os.environ['HELDOUT_VAL'] = '1' if val else '0'
    sys.path.insert(0, '/root/clrs'); os.chdir('/root/clrs')
    import run
    lines = []
    res = run.go(task, per_size, big, seed, log=lambda m: (lines.append(m), print(m, flush=True)))
    res['log'] = lines; res['where'] = 'modal cpu=2, one BLAS thread'
    return res


@app.local_entrypoint()
def main(tasks: str, big: str = '256:32,1024:8', per_size: int = 20, seed: int = 101, val: bool = False, out: str = 'results'):
    out = HERE / out; out.mkdir(exist_ok=True)
    for res in run_task.map(tasks.split(','), kwargs=dict(big=big, per_size=per_size, seed=seed, val=val), return_exceptions=True):
        if isinstance(res, Exception):
            print('FAILED', repr(res)); continue
        json.dump(res, open(out / f"{res['task']}.json", 'w'), indent=1, default=str)
        print(res['task'], 'fit_ms', res['fit_ms'], 'test64', res['test64']['score'], 'val', res.get('val', {}).get('score'),
              {k: v['score'] for k, v in res.items() if k.startswith('gen')})
