"""Modal app cold-check-clrs-extra: the 8 tasks added after the freeze (extra.py, dp_extra.py, gen_extra.py), on
CPU, one BLAS thread. Same image recipe as modal_clrs.py.
    modal run modal_extra.py --tasks graham_scan --big 64:32 --seed 9000 --val --out results_extra_heldout"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / 'clrs30'   # the official splits
app = modal.App('cold-check-clrs-extra')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.3.4', 'scipy>=1.14,<1.17')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'CLRS_DATA': '/root/data',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_dir(DATA, '/root/data', copy=True)
         .add_local_dir(HERE, '/root/clrs', ignore=['clrs30', 'clrs30/**', 'results*', 'results*/**', 'logs', 'logs/**', 'sweep', 'sweep/**', '__pycache__', '*.md', '*.json', '*.sh']))
DP = ('matrix_chain_order', 'optimal_bst')


@app.function(image=image, cpu=2.0, memory=8192, timeout=6 * 3600)
def run_task(task: str, big: str, seed: int, val: bool):
    import sys, os
    sys.path.insert(0, '/root/clrs'); os.chdir('/root/clrs')
    import extra, dp_extra
    lines = []
    mod = dp_extra if task in DP else extra
    res = mod.go(task, big, seed, val, log=lambda m: (lines.append(m), print(m, flush=True)))
    res['log'] = lines; res['where'] = 'modal cpu=2, one BLAS thread'
    return res


@app.local_entrypoint()
def main(tasks: str, big: str, seed: int = 9000, val: bool = False, out: str = 'results_extra_heldout'):
    o = HERE / out; o.mkdir(exist_ok=True)
    for res in run_task.map(tasks.split(','), kwargs=dict(big=big, seed=seed, val=val), return_exceptions=True):
        if isinstance(res, Exception):
            print('FAILED', repr(res)); continue
        json.dump(res, open(o / f"{res['task']}.json", 'w'), indent=1, default=str)
        print(res['task'], 'fit_ms', res['fit_ms'], 'test', res['test64']['score'], 'val', res.get('val', {}).get('score'),
              {k: v['score'] for k, v in res.items() if k.startswith('gen')})
