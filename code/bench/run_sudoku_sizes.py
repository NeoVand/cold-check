"""The 9 x 9 fit, unchanged, on Sudoku of other sizes: 4 x 4 (b = 2), 16 x 16 (b = 4), 25 x 25 (b = 5).

Puzzle generator (ours, seeded): a random solved board is a pattern board ((b (r mod b) + r div b + c) mod n) with the
digits, the rows within each band, the bands, the columns within each stack and the stacks permuted at random, and a
random transpose. Clues are then removed in a random order, a removal kept only if the puzzle still has one solution
(tested by asking the solver for a solution that differs from the board at that cell; a test that runs out of budget
keeps the clue). Removal stops at a minimal puzzle (4 x 4, 16 x 16) or at the target blank fraction (25 x 25: 0.55).
Scoring is independent of the solver: a returned board counts when every row, column and box holds each digit once
and every clue is kept; we also report agreement with the generated board.
Writes sudoku_sizes.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
import data_io  # noqa: E402
import engine  # noqa: E402


def pattern_board(b, rng):
    n = b * b
    r = np.arange(n)[:, None]
    c = np.arange(n)[None, :]
    base = (b * (r % b) + r // b + c) % n
    digits = rng.permutation(n) + 1
    rows = np.concatenate([g * b + rng.permutation(b) for g in rng.permutation(b)])
    cols = np.concatenate([g * b + rng.permutation(b) for g in rng.permutation(b)])
    x = digits[base][rows][:, cols]
    if rng.random() < 0.5:
        x = x.T
    return x.ravel()


def valid(board, b):
    n = b * b
    x = board.reshape(n, n)
    want = set(range(1, n + 1))
    for i in range(n):
        if set(x[i].tolist()) != want or set(x[:, i].tolist()) != want:
            return False
    for br in range(b):
        for bc in range(b):
            if set(x[br * b:(br + 1) * b, bc * b:(bc + 1) * b].ravel().tolist()) != want:
                return False
    return True


def solve_masks(E, dom0, seed, budget):
    out = np.zeros(8, np.int64)
    dom = engine.solve(dom0, E['unit_items'], E['unit_size'], E['item_units'], E['ge1'], E['le1'], E['V'],
                       E['W_i'], E['W_j'], E['W_w'], E['tau'], seed, budget, out)
    return dom, out


def make_puzzle(E, b, rng, target_blank=None, budget=20000):
    n = b * b
    sol = pattern_board(b, rng)
    full = (1 << n) - 1
    puz = sol.copy()
    for cell in rng.permutation(n * n):
        if target_blank is not None and (puz == 0).mean() >= target_blank:
            break
        trial = puz.copy()
        trial[cell] = 0
        dom0 = engine.masks(trial, n)
        dom0[cell] = full & ~(1 << (sol[cell] - 1))  # any solution with another digit here?
        _, o = solve_masks(E, dom0, int(rng.integers(1 << 30)), budget)
        if o[0] == -1:  # proved: no other solution
            puz = trial
    return puz, sol


_CK = None


def the_check():
    global _CK
    if _CK is None:
        check.vocabulary()
        sol9, _, _ = data_io.sudoku_split('train-s0')
        _CK = check.fit(sol9.astype(np.int64), steps=4, lr=8.0)[0]  # the train-s0 fit of run_sudoku.py
    return _CK


def one(b, i, tb, ck=None):
    """Generate puzzle i of side b*b (seeded by (b, i)) and solve it with the 9 x 9 fit. Returns a dict."""
    ck = ck or the_check()
    E = engine.build(ck, b)
    rng = np.random.default_rng([1000 + b, i])
    t0 = time.perf_counter()
    puz, sol = make_puzzle(E, b, rng, tb)
    gen = time.perf_counter() - t0
    t1 = time.perf_counter()
    dom, o = solve_masks(E, engine.masks(puz, b * b), 7 + i, 10 ** 6)
    w = time.perf_counter() - t1
    x = engine.decode(dom)
    good = bool(o[0] == 1 and valid(x, b) and (x[puz > 0] == puz[puz > 0]).all())
    return dict(good=good, equals=bool((x == sol).all()), moves=int(o[1]), ops=int(o[2]), cmp=int(o[3]),
                cold=bool(o[4]), status=int(o[0]), wall=w, gen=gen, blank=float((puz == 0).mean()))


def summarise(b, rows):
    g = lambda k: np.array([r[k] for r in rows])
    return dict(side=b * b, n=len(rows), blank_frac_mean=float(g('blank').mean()), solved_valid=int(g('good').sum()),
                equals_generated=int(g('equals').sum()), budget_misses=int((g('status') == 0).sum()),
                cold_only=int(g('cold').sum()), moves_mean=float(g('moves').mean()), moves_median=float(np.median(g('moves'))),
                moves_max=int(g('moves').max()), ops_mean=float(g('ops').mean()),
                ops_plus_cmp_mean=float((g('ops') + g('cmp')).mean()), wall_ms_mean=float(g('wall').mean() * 1e3),
                wall_ms_max=float(g('wall').max() * 1e3), gen_s_mean=float(g('gen').mean()))


def main():
    """python run_sudoku_sizes.py SIDE_B:N[:BLANK] ...   e.g. 2:1000 3:200 4:1000 (25 x 25 runs on Modal)."""
    ck = the_check()
    out_path = HERE / 'sudoku_sizes_local.json'
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res['fit'] = check.describe(ck)
    res['relations'] = [[check.FNAMES[i] for i in fs] for fs in ck['relations']]
    for arg in sys.argv[1:]:
        parts = arg.split(':')
        b, N = int(parts[0]), int(parts[1])
        tb = float(parts[2]) if len(parts) > 2 else None
        rows = [one(b, i, tb, ck) for i in range(N)]
        summ = summarise(b, rows)
        res[f'side{b * b}'] = summ
        print(json.dumps(summ), flush=True)
        out_path.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
