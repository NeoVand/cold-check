"""Modal app cold-check-clrs-dfs: depth-first search, topological sort and strongly connected components (dfs2.py),
on CPU, one BLAS thread. Same image recipe as modal_clrs.py.
    modal run modal_dfs.py --tasks dfs,topo_scc --big 64:32,256:4 --seed 9000 --val --out results_extra_heldout"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / 'clrs30'   # the official splits
app = modal.App('cold-check-clrs-dfs')
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
    import dfs2
    lines = []
    log = lambda m: (lines.append(m), print(m, flush=True))
    if task == 'dfs':
        res, _ = dfs2.go(big, seed, val, log=log); out = [res]
    else:
        out = list(dfs2.go_topo_scc(big, seed, val, log=log).values())
    for r in out:
        r['log'] = lines; r['where'] = 'modal cpu=2, one BLAS thread'
    return out


@app.local_entrypoint()
def main(tasks: str, big: str, seed: int = 9000, val: bool = False, out: str = 'results_extra_heldout'):
    o = HERE / out; o.mkdir(exist_ok=True)
    for res in run_task.map(tasks.split(','), kwargs=dict(big=big, seed=seed, val=val), return_exceptions=True):
        if isinstance(res, Exception):
            print('FAILED', repr(res)); continue
        for r in res:
            json.dump(r, open(o / f"{r['task']}.json", 'w'), indent=1, default=str)
            print(r['task'], 'fit_ms', r['fit_ms'], 'test', r['test64']['score'], 'val', r.get('val', {}).get('score'),
                  {k: v['score'] for k, v in r.items() if k.startswith('gen')})
