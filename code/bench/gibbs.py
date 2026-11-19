"""Registered test A2: CRAM's focused Gibbs search as the warm stage, with and without our cold stage first.
One chain; a move picks a uniformly random open cell whose digit clashes (field > 0; any open cell if none clashes...
which cannot happen before the stop) and redraws its digit among its still-possible values with probability
proportional to exp(-field / T). T falls geometrically from 0.40 to 0.31 of the mean kept cost over 524,288 moves,
then restarts (CRAM's schedule). The stop is the fitted check: energy <= tau. Writes gibbs.json.
Operations (CRAM's convention): field accumulation, energy update, exp and normalisation count; comparisons free.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_io  # noqa: E402
import engine  # noqa: E402
import run_sudoku_sizes as R  # noqa: E402


@njit(cache=True)
def gibbs(dom, part, pw, V, tau, hbar, seed, budget, out):
    """dom (n,) masks (singletons are clamped); part (n, k) partners (-1 pad), pw (n, k) their costs.
    out: 0 solved, 1 moves, 2 ops."""
    np.random.seed(seed)
    n = dom.shape[0]
    x = np.zeros(n, np.int64)
    openc = np.zeros(n, np.int64)
    no = 0
    for i in range(n):
        d = dom[i]
        c = engine_popcount(d)
        # random value among the possible ones
        p = np.random.randint(c)
        k = 0
        for v in range(V):
            if (d >> v) & 1:
                if k == p:
                    x[i] = v
                    break
                k += 1
        if c > 1:
            openc[no] = i
            no += 1
    # conflict field of each cell's own value
    conf = np.zeros(n)
    E = 0.0
    ops = 0
    for i in range(n):
        for j in range(part.shape[1]):
            q = part[i, j]
            if q >= 0 and x[q] == x[i]:
                conf[i] += pw[i, j]
                ops += 1
        E += 0.5 * conf[i]
    out[0] = 0
    moves = 0
    T0 = 0.40 * hbar
    T1 = 0.31 * hbar
    K = 524288
    field = np.zeros(V)
    probs = np.zeros(V)
    cand = np.zeros(n, np.int64)
    while moves < budget:
        if E <= tau:
            out[0] = 1
            break
        nc = 0
        for t in range(no):
            if conf[openc[t]] > 0:
                cand[nc] = openc[t]
                nc += 1
        if nc == 0:
            break
        i = cand[np.random.randint(nc)]
        T = T0 * (T1 / T0) ** ((moves % K) / K)
        ops += 2
        for v in range(V):
            field[v] = 0.0
        for j in range(part.shape[1]):
            q = part[i, j]
            if q >= 0:
                field[x[q]] += pw[i, j]
                ops += 1
        tot = 0.0
        for v in range(V):
            if (dom[i] >> v) & 1:
                probs[v] = np.exp(-field[v] / T)
                tot += probs[v]
                ops += 3
            else:
                probs[v] = 0.0
        r = np.random.random() * tot
        ops += 1
        nv = 0
        acc = 0.0
        for v in range(V):
            acc += probs[v]
            ops += 1
            if probs[v] > 0 and acc >= r:
                nv = v
                break
        old = x[i]
        if nv != old:
            # update conflicts of partners and the energy
            for j in range(part.shape[1]):
                q = part[i, j]
                if q >= 0:
                    if x[q] == old:
                        conf[q] -= pw[i, j]
                        ops += 1
                    elif x[q] == nv:
                        conf[q] += pw[i, j]
                        ops += 1
            E += field[nv] - field[old]
            conf[i] = field[nv]
            x[i] = nv
            ops += 2
        moves += 1
    out[1] = moves
    out[2] = ops
    return x


@njit(cache=True)
def engine_popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def partners(E, M):
    lists = [[] for _ in range(M)]
    for a, b, w in zip(E['W_i'], E['W_j'], E['W_w']):
        lists[a].append((b, w)); lists[b].append((a, w))
    k = max(len(l) for l in lists)
    part = -np.ones((M, k), np.int64)
    pw = np.zeros((M, k))
    for i, l in enumerate(lists):
        for j, (b, w) in enumerate(l):
            part[i, j] = b; pw[i, j] = w
    return part, pw


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    budget = 2 ** 21
    ck = R.the_check()
    E = engine.build(ck, 3)
    part, pw = partners(E, 81)
    hbar = float(E['W_w'].mean())
    ts, tc, _ = data_io.sudoku_split('test')
    puz = np.where(tc, ts, 0).astype(np.int64)
    res = {}
    for arm in ('no_cold', 'cold'):
        ok, wrong, mv, ops, cold_solved, wall = 0, 0, [], [], 0, 0.0
        t0 = time.perf_counter()
        for i in range(n):
            dom = engine.masks(puz[i], 9)
            extra_ops = 0
            if arm == 'cold':
                stats = np.zeros(2, np.int64)
                inq = np.ones(len(E['unit_size']), np.bool_)
                q = np.zeros(len(E['unit_size']), np.int64)
                engine.propagate(dom, E['unit_items'], E['unit_size'], E['item_units'], E['ge1'], E['le1'], 9, q, inq, stats)
                extra_ops = int(stats[0])
                if all(int(d) & (int(d) - 1) == 0 for d in dom):
                    cold_solved += 1
            o = np.zeros(3, np.int64)
            x = gibbs(dom, part, pw, 9, E['tau'], hbar, i + 1, budget, o)
            if o[0] == 1:
                if ((x + 1) == ts[i]).all():
                    ok += 1
                else:
                    wrong += 1
            mv.append(int(o[1])); ops.append(int(o[2]) + extra_ops)
        wall = time.perf_counter() - t0
        mv = np.array(mv)
        res[arm] = dict(n=n, budget=budget, solved=ok, wrong_stops=wrong, cold_stage_solved=cold_solved,
                        moves_mean=float(mv.mean()), moves_median=float(np.median(mv)),
                        moves_median_solved=float(np.median(mv[mv < budget])) if (mv < budget).any() else None,
                        ops_mean=float(np.mean(ops)), wall_s=wall)
        print(arm, json.dumps(res[arm]), flush=True)
    (HERE / 'gibbs.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
