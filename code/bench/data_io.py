"""Data loaders for CRAM's three benchmarks, written from scratch (their code is not imported).

Sudoku-Extreme: official CSVs at 58942f96; splits by the row lists in CRAM's sudoku_extreme_splits.json, each checked
against its content SHA-256 (the same hash CRAM's prepare_data.py computes). Test = every test.csv row that is not a
development row (dev10k): 412,786 puzzles (FRM's 1,000 test rows included), in official order (CRAM's full-all rows
10,000 and up hold the same puzzles; order does not matter for a per-puzzle score).
Maze-Unique: EqR's npy files (1 wall, 2 open, 3 S, 4 G; label 5 on path). Maze-Hard: HRM's CSVs ('#', ' ', 'S', 'G';
'o' on path). Mazes are returned as tokens (0 wall, 1 open, 2 S, 3 G) and a path mask that includes S and G.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
SPLITS = HERE.parent / 'ext' / 'CRAM' / 'scripts' / 'sudoku_extreme_splits.json'


def _sha(sol, clue, rat):
    parts = (np.ascontiguousarray(sol, np.int64), np.ascontiguousarray(clue, np.uint8), np.ascontiguousarray(rat, np.int64))
    return hashlib.sha256(b''.join(p.tobytes() for p in parts)).hexdigest()


def _read_csv(path, rows=None):
    """(solutions (N,81) int8 1..9, clue mask (N,81) bool, ratings (N,)); rows: optional set of row indices to keep."""
    sol, clue, rat = [], [], []
    with open(path, newline='') as f:
        r = csv.reader(f)
        next(r)
        for k, (_, q, a, g) in enumerate(r):
            if rows is not None and k not in rows:
                continue
            sol.append(np.frombuffer(a.encode(), np.uint8) - 48)
            clue.append(np.frombuffer(q.encode(), np.uint8) != 46)
            rat.append(int(g))
    return np.stack(sol).astype(np.int8), np.stack(clue), np.array(rat)


def sudoku_split(name):
    """name: 'train-s0'..'train-s19', 'test' (CRAM's 412,786), 'dev10k', 'frmtest'. Returns (solutions, clues, ratings)."""
    cache = DATA / f'sudoku_{name}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['sol'], z['clue'], z['rat']
    idx = json.loads(SPLITS.read_text())
    if name.startswith('train-s'):
        spec = idx['files'][f'frm/split/{name}.npz']
        rows = spec['rows']
        s, c, g = _read_csv(DATA / 'sudoku-extreme' / 'train.csv', set(rows))
        # _read_csv returns file order; reorder to the split's order
        order = np.argsort(np.argsort(rows))
        s, c, g = s[order], c[order], g[order]
        assert _sha(s, c, g) == spec['content_sha256'], name
    else:
        s, c, g = _read_csv(DATA / 'sudoku-extreme' / 'test.csv')
        assert len(s) == idx['test_csv_rows']
        dev = np.array(idx['files']['frm/dev10k/dev.npz']['rows'])
        frm = np.array(idx['files']['frm/split/test.npz']['rows'])
        if name == 'test':
            keep = np.setdiff1d(np.arange(len(s)), dev)  # rows 10,000+ of CRAM's full-all: FRM's 1,000 + the rest
        elif name == 'dev10k':
            keep = dev
        else:
            keep = frm
        s, c, g = s[keep], c[keep], g[keep]
        if name == 'dev10k':
            assert _sha(s, c, g) == idx['files']['frm/dev10k/dev.npz']['content_sha256']
    np.savez(cache, sol=s, clue=c, rat=g)
    return s, c, g


def maze_unique(split):
    d = DATA / 'eqr' / 'maze-30x30-unique-1k' / split
    x = np.load(d / 'all__inputs.npy').reshape(-1, 30, 30).astype(np.int64)
    y = np.load(d / 'all__labels.npy').reshape(-1, 30, 30)
    tok = np.select([x == 1, x == 2, x == 3, x == 4], [0, 1, 2, 3])
    return tok.astype(np.int8), (y == 5) | (tok == 2) | (tok == 3)


def maze_hard(split):
    rows = list(csv.reader(open(DATA / 'maze-hard' / f'{split}.csv')))[1:]
    q = np.array([list(r[1]) for r in rows]).reshape(-1, 30, 30)
    a = np.array([list(r[2]) for r in rows]).reshape(-1, 30, 30)
    tok = np.select([q == '#', q == ' ', q == 'S', q == 'G'], [0, 1, 2, 3]).astype(np.int8)
    return tok, (a == 'o') | (a == 'S') | (a == 'G')
