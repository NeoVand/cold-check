"""Modal app cold-check-clrs-script: run one of this folder's scripts on a CPU container (one BLAS thread) and bring
back the file it writes to final/.
    modal run modal_script.py --script ablation_level.py --out ablation_level.json
    modal run modal_script.py --script shared.py --out shared.json"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE / 'clrs30'   # the official splits
app = modal.App('cold-check-clrs-script')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.3.4', 'scipy>=1.14,<1.17')
         .env({'PYTHONDONTWRITEBYTECODE': '1', 'CLRS_DATA': '/root/data',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
         .add_local_dir(DATA, '/root/data', copy=True)
         .add_local_dir(HERE, '/root/clrs', ignore=['clrs30', 'clrs30/**', 'results*', 'results*/**', 'logs', 'logs/**', 'sweep', 'sweep/**', '__pycache__', '*.md', '*.json', '*.sh']))


@app.function(image=image, cpu=2.0, memory=16384, timeout=8 * 3600)
def run_script(script: str, out: str):
    import sys, os, subprocess
    os.chdir('/root/clrs'); os.makedirs('final', exist_ok=True)
    p = subprocess.run([sys.executable, script], capture_output=True, text=True)
    body = open(f'final/{out}').read() if os.path.exists(f'final/{out}') else None
    return dict(stdout=p.stdout[-20000:], stderr=p.stderr[-5000:], code=p.returncode, out=body)


@app.local_entrypoint()
def main(script: str, out: str):
    r = run_script.remote(script, out)
    print(r['stdout']); print(r['stderr'][-2000:], 'exit', r['code'])
    if r['out']:
        (HERE / 'final').mkdir(exist_ok=True); open(HERE / 'final' / out, 'w').write(r['out'])
