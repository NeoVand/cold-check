"""Modal app cold-check-clrs-final: the final battery, every CLRS-30 task with frozen code (final_run.py), one task per
CPU container, one BLAS thread. Writes final/<task>.json.
    modal run modal_final.py                      # all 30 (the DFS family runs as one job, dfs_family)
    modal run modal_final.py --tasks bfs,dijkstra"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / 'clrs30'   # the official splits
app = modal.App('cold-check-clrs-final')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.3.4', 'scipy>=1.14,<1.17')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'CLRS_DATA': '/root/data',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_dir(DATA, '/root/data', copy=True)
         .add_local_dir(HERE, '/root/clrs', ignore=['clrs30', 'clrs30/**', 'results*', 'results*/**', 'final', 'final/**', 'logs', 'logs/**', 'sweep', 'sweep/**', '__pycache__', '*.md', '*.json', '*.sh']))


@app.function(image=image, cpu=2.0, memory=8192, timeout=8 * 3600)
def run_task(task: str):
    import sys, os
    sys.path.insert(0, '/root/clrs'); os.chdir('/root/clrs')
    import final_run
    lines = []
    out = final_run.go(task, log=lambda m: (lines.append(m), print(m, flush=True)))
    for r in out:
        r['log'] = lines; r['where'] = 'modal cpu=2, one BLAS thread'
    return out


@app.local_entrypoint()
def main(tasks: str = ''):
    import final_run
    o = HERE / 'final'; o.mkdir(exist_ok=True)
    todo = tasks.split(',') if tasks else list(final_run.SPEC)
    for res in run_task.map(todo, return_exceptions=True):
        if isinstance(res, Exception):
            print('FAILED', repr(res)); continue
        for r in res:
            json.dump(r, open(o / f"{r['task']}.json", 'w'), indent=1, default=str)
            print(r['task'], 'test', r['test64']['score'], 'val', r.get('val', {}).get('score'),
                  {k: v['score'] for k, v in r.items() if k.startswith('gen')}, flush=True)
