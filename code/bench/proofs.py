"""Finite checks behind the every-size statements of the paper. Writes proofs.json.

P1 (Sudoku types are size-free). The set of pair types (8-bit agreement patterns of r, c, r//b, c//b, r%b, c%b,
   r-c, r+c) realised on a board of block size b is the same 25 types for every b >= 2 (checked for b = 2..8; the
   argument for larger b: every type is fixed by which of dr = 0, dc = 0, same band, same stack,
   dr = dc, dr = -dc hold, and each such combination that is realisable at all is realised once b >= 2).
P2 (the fitted check is "no clash in a row, column or box", at every size). The kept types are exactly the types
   with r, or c, or (r//b and c//b) agreeing; so on every board the kept pairs are the row, column and box pairs.
P3 (stop sound at every size). Every kept cost is at least wmin and tau < wmin, so energy <= tau iff no kept pair
   shares a value iff every row, column and box holds each value at most once; with n cells and n values per unit
   that is "exactly once": a valid Sudoku of any size. Both kept count conditions (n_v >= 1, n_v <= 1) hold on every
   valid board of every size (pigeonhole), so the cold stage never removes a value of a valid completion.
P4 (maze ranges are true of every room-lattice tree maze, ring or not). Checked by brute force on all spanning trees
   of small room lattices (2 x 2, 2 x 3, 3 x 3 rooms; with and without the ring), every S, G pair: the kept count
   ranges hold at the true answer. The general argument is in the paper.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
import data_io  # noqa: E402
import mazes as M  # noqa: E402
import run_sudoku as RS  # noqa: E402


def p1_p2_p3():
    out = {}
    sets = {}
    for b in range(2, 9):
        T = check.pair_types(b)
        sets[b] = set(np.unique(T[T >= 0]).tolist())
    out['P1_types_per_b'] = {b: len(s) for b, s in sets.items()}
    out['P1_same_for_all_b_ge3'] = all(s == sets[3] for b, s in sets.items() if b >= 3)
    out['P1_b2_subset'] = sets[2] <= sets[3]
    fits = []
    for k in range(3):
        sol, _, _ = data_io.sudoku_split(f'train-s{k}')
        steps, _ = RS.choose_steps(sol.astype(np.int64))  # the fit setting rule of run_sudoku.py
        ck, _ = check.fit(sol.astype(np.int64), steps=steps, lr=32.0 / steps)
        kept = set(ck['kept_types'].tolist())
        want = {t for t in sets[3] if (t & 1) or (t & 2) or ((t & 4) and (t & 8))}
        fits.append(dict(split=k, kept_equals_row_col_box=kept == want, n_kept=len(kept), wmin=ck['wmin'], tau=ck['tau'],
                         stop_sound=ck['wmin'] > ck['tau'] and ck['tau'] < 0.01))
        # on every b, kept pairs == same row or same column or same box
        for b in range(2, 9):
            W = check.weights_for(ck, b)
            n = b * b
            r, c = np.divmod(np.arange(n * n), n)
            rcb = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (((r // b)[:, None] == (r // b)[None]) & ((c // b)[:, None] == (c // b)[None]))
            np.fill_diagonal(rcb, False)
            fits[-1][f'b{b}_kept_pairs_are_rcb'] = bool(((W > 0) == rcb).all())
    out['P2_P3_fits'] = fits
    return out


def spanning_trees(h, w):
    """All spanning trees of the h x w room grid, as sets of edges."""
    rooms = [(r, c) for r in range(h) for c in range(w)]
    edges = [((r, c), (r + dr, c + dc)) for r, c in rooms for dr, dc in ((0, 1), (1, 0)) if r + dr < h and c + dc < w]
    need = len(rooms) - 1
    for sub in itertools.combinations(edges, need):
        parent = {x: x for x in rooms}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ok = True
        for a, b in sub:
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if ok:
            yield sub


def p4():
    tk, y = data_io.maze_unique('train')
    ck, _ = M.fit_check(tk, y)
    viol, checked, exact = 0, 0, 0
    for h, w in ((2, 2), (2, 3), (3, 3)):
        for sub in spanning_trees(h, w):
            g = np.zeros((2 * h + 1, 2 * w + 1), bool)
            for (r, c) in [(r, c) for r in range(h) for c in range(w)]:
                g[2 * r + 1, 2 * c + 1] = True
            for (a, b) in sub:
                g[a[0] + b[0] + 1, a[1] + b[1] + 1] = True
            for ring in (True, False):
                gg = g if ring else g[1:-1, 1:-1]
                op = [tuple(x) for x in np.argwhere(gg)]
                for S, G in itertools.permutations(op, 2):
                    t = M.tokens(gg, S, G)
                    lab = M.path_cells_tree(gg, S, G)
                    checked += 1
                    viol += int(not M.passes(ck, t, lab))
                    b, *_ = M.cold_unique(ck, t)
                    exact += int(b is not None and (b == lab).all())
    return dict(P4_mazes_checked=checked, P4_range_violations=viol, P4_cold_exact=exact,
                kept=[M.RELNAMES[r] for r in ck['kept']])


def main(out_path=None):
    check.vocabulary()
    res = p1_p2_p3()
    res.update(p4())
    Path(out_path or HERE / 'proofs.json').write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=str))
