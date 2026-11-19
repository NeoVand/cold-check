"""Boards of CRAM's released maze models for the figures (nothing trained; their checkpoints, their code).

Re-runs, with the exact code of code/released/modal_cram_check.py (imported, not retyped), two cases whose
summaries are already recorded there but whose boards were not kept:
  edge   P-c/P-d at h = 15 without the wall ring (29 x 29 tree mazes, dead ends at the board edge): the same 200 mazes
         (rng 1000 + 15), the same batched zero-temperature Gibbs call; full predicted boards kept for mazes 0-3.
  room   P-b: the Maze-Hard model (seed 0) on room-and-corridor mazes, extra 2, symmetry 1, rooms 24 and 26 (recorded
         '+2p': a valid path two steps longer) and extra 0, symmetry 0, room 8 (recorded 'Sp': another shortest path).
Every summary number is asserted against the recorded run_released_*.json before anything is written.

    uvx modal run data/modal_released_boards.py          (from the v1 folder; writes data/released_boards.json)
"""
import json
import pathlib
import sys

import modal

import os
WORK = pathlib.Path(os.environ.get('COLDCHECK_DATA', pathlib.Path(__file__).resolve().parents[4] / 'code')) / 'released'
sys.path[:0] = [str(WORK), '/root/ours']
import modal_cram_check as M  # noqa: E402

app = modal.App('cold-check-figs')
image = M.image.add_local_file(WORK / 'modal_cram_check.py', '/root/ours/modal_cram_check.py')
OUT = pathlib.Path(__file__).resolve().parent / 'released_boards.json'


@app.function(image=image, cpu=8, memory=8192, timeout=1800)
def boards():
    import numpy as np
    import torch
    M._setup()
    from cram import maze_unique as MU
    from pointer_test import bfs_dist, greedy, DIRS, ORDER
    out = {'edge': {}, 'room': []}
    h = 15
    for seed in range(3):
        m = M.unique_model(seed)
        rng = np.random.default_rng(1000 + h)
        mazes = []
        for i in range(200):
            g = M.tree_maze(h, rng)
            op = np.argwhere(g)
            a, b = rng.choice(len(op), 2, replace=False)
            mazes.append((g, tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b])))
        X, Y, keep = [], [], []
        for g, S, G in mazes:
            g = g[1:-1, 1:-1]; S = (S[0] - 1, S[1] - 1); G = (G[0] - 1, G[1] - 1)
            t = M.tokens(g, S, G)
            lab = np.zeros(g.shape, np.int64)
            for c in M.path_cells(g, S, G):
                lab[c] = 1
            X.append(t); Y.append(lab); keep.append((g, S, G))
        x = torch.as_tensor(np.stack(X)); y = torch.as_tensor(np.stack(Y)); fr = x == MU.OPEN
        budget = max(20000, 3 * int(fr.flatten(1).sum(1).max()))
        gen = torch.Generator().manual_seed(0)
        r = MU.gibbs(m, x, y, fr, T=0.0, max_moves=budget, focus='random', gen=gen)
        pred = r['pred'].numpy(); yy = y.numpy(); frn = fr.numpy()
        rows = []
        for i in range(4):
            g, S, G = keep[i]
            rows.append(dict(grid=g.astype(int).tolist(), S=list(S), G=list(G), label=yy[i].tolist(), pred=pred[i].tolist(),
                             solved=bool(r['right'][i]), stopped=bool(r['stopped'][i]), moves=int(r['moves'][i]),
                             n_diff=int(((pred[i] != yy[i]) & frn[i]).sum())))
        out['edge'][seed] = dict(budget=budget, solved_all=int(np.asarray(r['right']).sum()), mazes=rows)
    m = M.hard_model(0)
    for extra, n, k in ((2, 24, 1), (2, 26, 1), (0, 8, 0)):
        g, S, G = M.sym(*M.L_maze(n, n, extra), k)
        dG = bfs_dist(g, G)
        lab = greedy(g, S, G, lambda kk, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -ORDER.index(kk)))
        board, used, done, ps, psi, s = M.run_hard(m, M.tokens(g, S, G)[None])
        valid, length = M.check_path(board[0], S, G)
        out['room'].append(dict(extra=extra, n=n, sym=k, grid=g.astype(int).tolist(), S=list(S), G=list(G),
                                label=[list(map(int, c)) for c in lab], board=board[0].astype(int).tolist(),
                                valid=bool(valid), length=length, shortest=int(dG[S]), stopped=bool(done[0])))
    return out


@app.local_entrypoint()
def main():
    res = boards.remote()
    # the recorded summaries, asserted
    for seed in range(3):
        rec = json.loads((WORK / f'run_released_pcd_{seed}_15.json').read_text())['result']['variants']['noring']
        assert res['edge'][seed]['budget'] == rec['budget'] if 'budget' in rec else True
        assert res['edge'][seed]['solved_all'] == sum(mz['solved'] for mz in rec['mazes'])
        for i, row in enumerate(res['edge'][seed]['mazes']):
            assert row['n_diff'] == rec['mazes'][i]['n_diff'], (seed, i, row['n_diff'], rec['mazes'][i]['n_diff'])
            assert row['solved'] == rec['mazes'][i]['solved'] and row['moves'] == rec['mazes'][i]['moves']
    pab = json.loads((WORK / 'run_released_pab_0.json').read_text())['result']
    for r in res['room']:
        recd = [x for x in pab if (x['extra'], x['n'], x['sym']) == (r['extra'], r['n'], r['sym'])][0]
        assert r['valid'] == recd['own']['valid_path'] and r['length'] == recd['own']['length'], (r['length'], recd['own'])
        assert r['shortest'] == recd['shortest']
    res['source'] = dict(code='code/released/modal_cram_check.py', checked_against=['code/released/run_released_pcd_{0,1,2}_15.json',
                         'code/released/run_released_pab_0.json'], note='summaries asserted equal to the recorded runs')
    OUT.write_text(json.dumps(res))
    print('wrote', OUT, [(r['n'], r['length'], r['shortest']) for r in res['room']])
