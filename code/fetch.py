"""Fetch the third-party code and data this release uses but does not redistribute.

    python fetch.py            # everything below
    python fetch.py cram       # CRAM's repository only (split index, released checkpoints, result files)
    python fetch.py mazes      # Maze-Hard and Maze-Unique
    python fetch.py sudoku     # Sudoku-Extreme (about 760 MB)

CRAM (Hoover, Blayney and Whittington, arXiv:2611.17309; Apache 2.0) is cloned at commit 62bbe83 into ext/CRAM. We use
its Sudoku-Extreme split index (scripts/sudoku_extreme_splits.json), its released checkpoints (released/) and its
result files (the comparisons of the paper and the page). None of its code runs in our solvers; released/ runs it on
Modal to test its released models.

Datasets (Hugging Face, at pinned revisions), into bench/data/:
  sapientinc/sudoku-extreme        58942f96 (the revision named in CRAM's split index)
  sapientinc/maze-30x30-hard-1k    549754de
  locuslab/EqR-data                bcb7d45c (maze-30x30-unique-1k only)
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE / 'ext' / 'CRAM'
DATA = HERE / 'bench' / 'data'
CRAM_URL = 'https://github.com/Prefrontal-AI/CRAM'
CRAM_COMMIT = '62bbe830c7d6f87f2ba18f30a28039258b986454'
MAZE_HARD = ('sapientinc/maze-30x30-hard-1k', '549754de3d67eebe57b6b22c7d226b9412d55b07')
EQR = ('locuslab/EqR-data', 'bcb7d45c157c06a054b715933557d266f5bd9d7f')
SUDOKU_REVISION = '58942f96'


def cram():
    if not (EXT / '.git').exists():
        EXT.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q'], cwd=EXT, check=True)
        subprocess.run(['git', 'remote', 'add', 'origin', CRAM_URL], cwd=EXT, check=True)
    subprocess.run(['git', 'fetch', '-q', '--depth', '1', 'origin', CRAM_COMMIT], cwd=EXT, check=True)
    subprocess.run(['git', 'checkout', '-q', CRAM_COMMIT], cwd=EXT, check=True)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=EXT, check=True, capture_output=True, text=True).stdout.strip()
    assert head == CRAM_COMMIT, head
    print('CRAM at', head, 'in', EXT)


def mazes():
    from huggingface_hub import hf_hub_download, snapshot_download
    for n in ('train.csv', 'test.csv'):
        hf_hub_download(MAZE_HARD[0], n, repo_type='dataset', revision=MAZE_HARD[1], local_dir=DATA / 'maze-hard')
    snapshot_download(EQR[0], repo_type='dataset', revision=EQR[1], allow_patterns='maze-30x30-unique-1k/*',
                      local_dir=DATA / 'eqr')
    print('mazes in', DATA)


def sudoku():
    from huggingface_hub import hf_hub_download
    idx = json.loads((EXT / 'scripts' / 'sudoku_extreme_splits.json').read_text())   # run `python fetch.py cram` first
    assert idx['revision'].startswith(SUDOKU_REVISION), idx['revision']
    for n in ('train.csv', 'test.csv'):
        hf_hub_download(idx['dataset'], n, repo_type='dataset', revision=idx['revision'], local_dir=DATA / 'sudoku-extreme')
    print('Sudoku-Extreme in', DATA / 'sudoku-extreme')


if __name__ == '__main__':
    what = sys.argv[1:] or ['cram', 'mazes', 'sudoku']
    for w in what:
        {'cram': cram, 'mazes': mazes, 'sudoku': sudoku}[w]()
