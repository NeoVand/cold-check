"""N-Queens and graph colouring at 10x the training size (registered tests Q1 and C1). Writes multi.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multi as Q  # noqa: E402


def timed(f, *a, **k):
    ts = []
    for _ in range(8):
        r, dt = f(*a, **k)
        ts.append(dt)
    return r, ts[0] * 1e3, float(np.median(ts[1:])) * 1e3


def main(out_path=None, budget=10 ** 6, restart_base=0):
    """restart_base > 0 uses the final engine (multi.solve_r, Luby restarts); 0 the first version (multi.solve)."""

    def run(dom, ui, us, iu, ge, le, V, seed, o):
        if restart_base > 0:
            o6 = np.zeros(6, np.int64)
            d = Q.solve_r(dom, ui, us, iu, ge, le, V, seed, budget, o6, restart_base)
            o[:5] = o6[:5]
            return d
        return Q.solve(dom, ui, us, iu, ge, le, V, seed, budget, o)

    out = {}
    ck, c_ms, w_ms = timed(Q.queens_check)
    out['queens_fit'] = dict(rels=[Q.QF[f] for f in ck['rels']], kept_types=[[int(t), int(v)] for t, v in ck['kept_types']],
                             conds=[[g.tolist(), l.tolist()] for g, l in ck['conds']], w=ck['w'][:, 1].round(4).tolist(),
                             bias=ck['bias'].tolist(), fit_ms_cold=c_ms, fit_ms_warm_median7=w_ms)
    print(out['queens_fit'], flush=True)
    qres = {}
    for N, settings in ((8, [('k5-7', None)]), (10, [('k7-9', None)]), (80, [('k9', 9), ('7/8', 'frac'), ('empty', 'all')]),
                        (100, [('k9', 9), ('7/8', 'frac'), ('empty', 'all')])):
        ui, us, iu, ge, le = Q.queens_engine(ck, N)
        for name, how in settings:
            ok, mv, ops, cold, wall = 0, [], [], 0, []
            for i in range(100):
                rng = np.random.default_rng([N, i, 7])
                cols = Q.min_conflicts(N, int(rng.integers(1 << 30)), 10 ** 4)
                while not (cols >= 0).all():  # min-conflicts can stall on small boards: retry with a new seed
                    cols = Q.min_conflicts(N, int(rng.integers(1 << 30)), 10 ** 4)
                if how is None:
                    k = int(rng.integers(5, 8)) if N == 8 else int(rng.integers(7, 10))
                elif how == 'frac':
                    k = N - N // 8
                elif how == 'all':
                    k = N
                else:
                    k = how
                keep = rng.permutation(N)[k:]
                dom = np.full(N * N, 3, np.int64)
                for r in keep:
                    dom[r * N:(r + 1) * N] = 1
                    dom[r * N + cols[r]] = 2
                o = np.zeros(5, np.int64)
                t0 = time.perf_counter()
                d = run(dom, ui, us, iu, ge, le, 2, i + 1, o)
                wall.append(time.perf_counter() - t0)
                b = (d == 2).astype(int)
                good = o[0] == 1 and Q.queens_valid(b, N) and all(b[r * N + cols[r]] == 1 for r in keep)
                ok += int(good); mv.append(int(o[1])); ops.append(int(o[2])); cold += int(o[4])
            qres[f'{N}_{name}'] = dict(N=N, input=name, n=100, solved=ok, cold_only=cold, moves_mean=float(np.mean(mv)),
                                       moves_max=int(max(mv)), ops_mean=float(np.mean(ops)), wall_ms_mean=float(np.mean(wall) * 1e3))
            print(qres[f'{N}_{name}'], flush=True)
    out['queens'] = qres
    rng = np.random.default_rng(10)
    train = [Q.planted_graph(10, 0.5, rng) for _ in range(1000)]
    cc, c_ms, w_ms = timed(Q.colour_check, train)
    out['colour_fit'] = dict(w=cc['w'].round(4).tolist(), kept=['non-edge', 'edge'][cc['kept'][0]] if len(cc['kept']) == 1 else cc['kept'],
                             conds=[[g.tolist(), l.tolist()] for g, l in cc['conds']], fit_ms_cold=c_ms, fit_ms_warm_median7=w_ms)
    print(out['colour_fit'], flush=True)
    cres = {}
    for n, d in ((10, 'train-like'), (100, 3.0), (100, 4.5), (100, 6.0)):
        ok, mv, cold, ops = 0, [], 0, []
        for i in range(100):
            g = np.random.default_rng([n, i, 11])
            p = 0.5 if d == 'train-like' else d / ((2 / 3) * (n - 1))
            A, col = Q.planted_graph(n, p, g)
            ui, us, iu, ge, le = Q.colour_engine(cc, A)
            o = np.zeros(5, np.int64)
            dom = run(np.full(n, 7, np.int64), ui, us, iu, ge, le, 3, i + 1, o)
            x = np.array([int(v).bit_length() - 1 for v in dom])
            i_, j_ = np.nonzero(np.triu(A, 1))
            good = o[0] == 1 and (dom > 0).all() and ((dom & (dom - 1)) == 0).all() and not (x[i_] == x[j_]).any()
            ok += int(good); mv.append(int(o[1])); cold += int(o[4]); ops.append(int(o[2]))
        cres[f'{n}_{d}'] = dict(n_vertices=n, avg_degree=d, n=100, solved=ok, cold_only=cold, moves_mean=float(np.mean(mv)),
                               moves_max=int(max(mv)), ops_mean=float(np.mean(ops)))
        print(cres[f'{n}_{d}'], flush=True)
    out['colour'] = cres
    out['budget'] = budget
    out['restart_base'] = restart_base
    Path(out_path or HERE / 'multi.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
