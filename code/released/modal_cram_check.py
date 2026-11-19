"""Modal app cold-check-released: run CRAM's released Maze-Hard and Maze-Unique checkpoints,
with their own code, on CPU containers. Nothing is trained. Their clone (ext/CRAM, from fetch.py) is copied into the
image (no .git); their prepare_data.py downloads the two public datasets at the pinned revisions during the image build.

    modal run released/modal_cram_check.py            # every job; JSON results land in released/

Jobs: data (dataset files, label-rule check, Maze-Unique test-set border counts), repro_hard / repro_unique (their test
scores), pab (P-a, P-b room-and-corridor mazes on the Maze-Hard model), pcd (P-c, P-d spanning-tree mazes on the
Maze-Unique model), loops (looped mazes on the Maze-Hard model, seed 0)."""
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CRAM = HERE.parent / 'ext' / 'CRAM'
app = modal.App('cold-check-released')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.11.0', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy>=1.26', 'einops>=0.8,<0.9', 'tyro>=1.0', 'huggingface-hub>=0.25')
         .env({'CRAM_DATA': '/root/data', 'PYTHONDONTWRITEBYTECODE': '1'})
         .add_local_dir(CRAM, '/root/CRAM', copy=True, ignore=['.git', '.git/**', 'data', 'third_party', '.venv'])
         .run_commands('cd /root/CRAM && python scripts/prepare_data.py maze-hard && python scripts/prepare_data.py maze-unique')
         .add_local_file(HERE / 'pointer_test.py', '/root/ours/pointer_test.py'))
NCPU = 8


# ----------------------------------------------------------------------------------------------- our maze builders --
def L_maze(h, w, extra):  # verbatim from pointer_test2.py
    import numpy as np
    off = 1 if extra else 0
    H, W = h + 2, w + 2 + off
    g = np.zeros((H, W), bool)
    g[0:h, 0:w] = True
    g[0, w] = True
    S = (0, w + 1)
    g[0:h + 2, w + 1] = True
    if extra:
        g[h, 0] = True; g[h + 1, 0] = True; G = (h + 1, 1); g[h + 1, 1:w + 2] = True
    else:
        g[h, 0] = True; G = (h + 1, 0); g[h + 1, 0:w + 2] = True
    g = np.pad(g, 1)
    return g, (S[0] + 1, S[1] + 1), (G[0] + 1, G[1] + 1)


def sym(g, S, G, k):  # verbatim from pointer_test2.py
    import numpy as np
    m = np.zeros(g.shape, np.int8); m[S] = 1; m[G] = 2
    g2, m2 = np.rot90(g, k % 4), np.rot90(m, k % 4)
    if k >= 4: g2, m2 = g2.T, m2.T
    S2 = tuple(int(x) for x in np.argwhere(m2 == 1)[0]); G2 = tuple(int(x) for x in np.argwhere(m2 == 2)[0])
    return np.ascontiguousarray(g2), S2, G2


def tree_maze(h, rng):
    """Uniform random spanning tree (Wilson's algorithm) of an h x h lattice of rooms on a (2h+1) x (2h+1) grid."""
    import numpy as np
    g = np.zeros((2 * h + 1, 2 * h + 1), bool)
    cells = [(r, c) for r in range(h) for c in range(h)]
    intree = np.zeros((h, h), bool)
    root = cells[rng.integers(len(cells))]
    intree[root] = True
    g[2 * root[0] + 1, 2 * root[1] + 1] = True
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for start in rng.permutation(len(cells)):
        u = cells[start]
        if intree[u]:
            continue
        nxt = {}
        cur = u
        while not intree[cur]:
            while True:
                d = steps[rng.integers(4)]
                v = (cur[0] + d[0], cur[1] + d[1])
                if 0 <= v[0] < h and 0 <= v[1] < h:
                    break
            nxt[cur] = v
            cur = v
        cur = u
        while not intree[cur]:
            v = nxt[cur]
            intree[cur] = True
            g[2 * cur[0] + 1, 2 * cur[1] + 1] = True
            g[cur[0] + v[0] + 1, cur[1] + v[1] + 1] = True
            cur = v
    return g


def path_cells(g, S, G):
    """The BFS path S -> G (unique in a tree)."""
    from collections import deque
    H, W = g.shape
    par = {S: None}
    q = deque([S])
    while q:
        u = q.popleft()
        if u == G:
            break
        for d in ((1, 0), (0, -1), (0, 1), (-1, 0)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < H and 0 <= v[1] < W and g[v] and v not in par:
                par[v] = u; q.append(v)
    p = [G]
    while p[-1] != S:
        p.append(par[p[-1]])
    return p[::-1]


def side_open(g):
    import numpy as np
    n = np.zeros(g.shape, int)
    n[1:] += g[:-1]; n[:-1] += g[1:]; n[:, 1:] += g[:, :-1]; n[:, :-1] += g[:, 1:]
    return n


def on_edge(r, c, H, W):
    return bool(r == 0 or c == 0 or r == H - 1 or c == W - 1)


def check_path(board, S, G):
    """(valid simple S-G path, length in steps or None)."""
    import numpy as np
    from collections import deque
    b = np.asarray(board, bool)
    if not (b[S] and b[G]):
        return False, None
    n = side_open(b)
    on = np.argwhere(b)
    for r, c in on:
        want = 1 if (r, c) in (S, G) else 2
        if n[r, c] != want:
            return False, None
    seen = {S}
    q = deque([S])
    while q:
        u = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < b.shape[0] and 0 <= v[1] < b.shape[1] and b[v] and v not in seen:
                seen.add(v); q.append(v)
    if len(seen) != len(on):
        return False, None
    return True, len(on) - 1


def tokens(g, S, G):
    import numpy as np
    t = g.astype(np.int64)
    t[S] = 2; t[G] = 3
    return t


# --------------------------------------------------------------------------------------------- their code, loaded --
def _setup():
    import torch
    torch.set_num_threads(NCPU)
    sys.path[:0] = ['/root/CRAM', '/root/CRAM/experiments', '/root/ours']


def hard_model(seed):
    import torch
    import maze_hard_latent as H
    rec = json.loads(Path(f'/root/CRAM/checkpoints/maze-hard/seed{seed}/final.json').read_text())
    m = H.build(H.Recipe(**rec['recipe']))
    m.load_state_dict(torch.load(f'/root/CRAM/checkpoints/maze-hard/seed{seed}/model.pt', map_location='cpu', weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def unique_model(seed):
    import torch
    import maze_unique as U
    cfg = U.Train(**json.loads(Path(f'/root/CRAM/checkpoints/maze-unique/seed{seed}/config.json').read_text()))
    m = U.make_model(cfg)
    m.load_state_dict(torch.load(f'/root/CRAM/checkpoints/maze-unique/seed{seed}/model.pt', map_location='cpu', weights_only=True))
    return m.eval()


def run_hard(model, toks, settle_max=4000, descend_steps=2048):
    """As maze_hard_latent.evaluate on one batch of equal-shape mazes: settle, couplings, damped descent (0.2)."""
    import torch
    from cram import maze_h3 as m3
    T = torch.as_tensor(toks)
    with torch.no_grad():
        psi, ps = model.settle(T, max_steps=settle_max)
        s = model.att_scores(T, psi)
        a = model.couplings(T, psi)
        nu, used, done = m3.descend(model, T, a, steps=descend_steps, eta=.2)
    return nu.numpy() > 0.5, used.numpy(), done.numpy(), ps.numpy(), psi.numpy(), s.numpy()


def pointer_path(s, S, G):
    """Follow the largest side attention score from S (aoffs = self, up, left, right, down). Returns (path, number of
    cells on it whose overall argmax is the self offset)."""
    import numpy as np
    offs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    H, W = s.shape[1:]
    p, selfmax = [S], 0
    while p[-1] != G and len(p) <= H * W:
        r, c = p[-1]
        sc = s[:, r, c]
        selfmax += int(np.argmax(sc) == 0)
        k = int(np.argmax(sc[1:]))
        p.append((r + offs[k][0], c + offs[k][1]))
    return p


# ------------------------------------------------------------------------------------------------------------ jobs --
def job_data():
    import hashlib
    import numpy as np
    from cram import maze_learned as ML
    from cram import maze_unique as MU
    from pointer_test import bfs_dist, greedy, ORDER, DIRS
    import itertools
    out = {'files': {}}
    for p in sorted(Path('/root/data').rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            out['files'][str(p.relative_to('/root/data'))] = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    import scripts.prepare_data as PD  # noqa
    out['revisions'] = {'maze_hard': PD.MAZE_HARD, 'maze_unique': PD.MAZE_UNIQUE}
    tok, y = ML.load('test')
    out['maze_hard_test_shape'] = list(tok.shape)

    def rule_greedy(g, S, G, order):
        dG = bfs_dist(g, G)
        return greedy(g, S, G, lambda kk, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -order.index(kk)))

    def rule_bfs_from_S(g, S, G, order):
        from collections import deque
        H, W = g.shape; par = {S: None}; q = deque([S])
        while q:
            u = q.popleft()
            for k in order:
                v = (u[0] + DIRS[k][0], u[1] + DIRS[k][1])
                if 0 <= v[0] < H and 0 <= v[1] < W and g[v] and v not in par:
                    par[v] = u; q.append(v)
        p = [G]
        while p[-1] != S:
            p.append(par[p[-1]])
        return p
    agree, rules = 0, {}
    orders = list(itertools.permutations(['down', 'left', 'right', 'up']))
    mism = []
    for i in range(200):
        g = tok[i] > 0
        S = tuple(int(v) for v in np.argwhere(tok[i] == 2)[0]); G = tuple(int(v) for v in np.argwhere(tok[i] == 3)[0])
        lab = set(map(tuple, np.argwhere(y[i]).tolist()))
        ours = set(rule_greedy(g, S, G, ORDER))
        ok = ours == lab
        agree += ok
        if not ok:
            mism.append(i)
        for o in orders:
            for name, f in (('greedy_from_S', rule_greedy), ('bfs_parent_from_S', rule_bfs_from_S)):
                key = f'{name}:{",".join(o)}'
                rules[key] = rules.get(key, 0) + (set(f(g, S, G, list(o))) == lab)
        dG = bfs_dist(g, G)
    out['label_rule'] = {'ours_down_left_right_up_agree_of_200': agree, 'mismatch_indices': mism,
                         'best_alternatives': sorted(rules.items(), key=lambda kv: -kv[1])[:6]}
    # ties: how many of the 200 have more than one shortest path (so the rule matters)
    # Maze-Unique test set: border counts
    d = MU.load('test')
    X = d['inputs'].reshape(-1, 30, 30)
    edge = np.zeros((30, 30), bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
    ne, nt, mz_e, mz_t = [], [], 0, 0
    for x in X:
        g = x > 0
        n = side_open(g)
        tips = g & (n == 1) & (x == 1)
        ne.append(int((g & edge).sum())); nt.append(int((tips & edge).sum()))
    out['maze_unique_test'] = {'open_cells_on_edge_total': int(sum(ne)), 'mazes_with_open_edge_cell': int(sum(v > 0 for v in ne)),
                               'edge_dead_end_tips_total': int(sum(nt)), 'mazes_with_edge_tip': int(sum(v > 0 for v in nt)),
                               'mean_open_cells': float((X > 0).sum((1, 2)).mean()),
                               'open_at_odd_odd_fraction': float(((X > 0)[:, 1::2, 1::2]).mean())}
    Xh = tok
    eh = [int(((x > 0) & edge).sum()) for x in Xh]
    out['maze_hard_test_open_edge_total'] = int(sum(eh))
    return out


def job_repro_hard(seed):
    import maze_hard_latent as H
    import torch
    from cram import maze_h3 as m3
    m = hard_model(seed)
    tok, y = m3.load('test')
    r = H.evaluate(m, tok, y, dev=torch.device('cpu'))
    exp = json.loads(Path('/root/CRAM/checkpoints/manifest.json').read_text())['maze-hard']['models'][seed]['expected']
    pm = r.pop('per_maze')
    import numpy as np
    ex, sp = np.array(pm['exact']), np.array(pm['stopped'])
    r['wrong_stop_indices'] = np.where(sp & ~ex).tolist() if False else [int(i) for i in np.where(sp & ~ex)[0]]
    return {'seed': seed, 'got': r, 'recorded': exp}


def job_repro_unique(seed):
    import maze_unique as U
    m = unique_model(seed)
    e = json.loads(Path('/root/CRAM/checkpoints/manifest.json').read_text())['maze-unique']['models'][seed]['expected']
    r = U.score(m, 'test', U.Search(**e['search']), device='cpu')
    pm = r.pop('per_maze')
    return {'seed': seed, 'got': r, 'recorded': e}


def job_pab(seed):
    import numpy as np
    from pointer_test import bfs_dist, greedy, count_paths, DIRS, ORDER
    m = hard_model(seed)
    rows = []
    for extra, ns in ((0, [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]), (2, [12, 16, 18, 20, 22, 24, 26])):
        for k in range(8):
            for n in ns:
                g, S, G = sym(*L_maze(n, n, extra), k)
                dG = bfs_dist(g, G)
                lab = greedy(g, S, G, lambda kk, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -ORDER.index(kk)))
                nb = [(kk, (S[0] + d[0], S[1] + d[1])) for kk, d in DIRS.items() if g[S[0] + d[0], S[1] + d[1]]]
                N = count_paths(g, dG)
                (ka, va), (kb, vb) = nb
                corr, room = ((ka, va), (kb, vb)) if N[va] == 1 and dG[va] <= dG[vb] else ((kb, vb), (ka, va))
                labset = np.zeros(g.shape, bool)
                for c in lab:
                    labset[c] = True
                row = dict(seed=seed, extra=extra, sym=k, n=n, grid=list(g.shape), corridor=corr[0], room=room[0],
                           shortest=int(dG[S]), label_first=[kk for kk, d in DIRS.items() if (S[0] + d[0], S[1] + d[1]) == lab[1]][0])
                variants = [('own', g, S, G, labset)]
                if g.shape[0] <= 30 and g.shape[1] <= 30:
                    g30 = np.zeros((30, 30), bool); g30[:g.shape[0], :g.shape[1]] = g
                    l30 = np.zeros((30, 30), bool); l30[:g.shape[0], :g.shape[1]] = labset
                    variants.append(('emb30', g30, S, G, l30))
                for name, gg, SS, GG, ll in variants:
                    board, used, done, ps, psi, s = run_hard(m, tokens(gg, SS, GG)[None])
                    b = board[0]
                    valid, length = check_path(b, SS, GG)
                    first = None
                    for kk, d in DIRS.items():
                        v = (SS[0] + d[0], SS[1] + d[1])
                        if 0 <= v[0] < b.shape[0] and 0 <= v[1] < b.shape[1] and b[v]:
                            first = (first + '+' + kk) if first else kk
                    ptr = pointer_path(s[0], SS, GG)
                    row[name] = dict(equals_label=bool((b == ll).all()), valid_path=valid, length=length, stopped=bool(done[0]),
                                     descent_steps=int(used[0]), settle_steps=int(ps[0]), returned_first=first,
                                     pointer_first=[kk for kk, d in DIRS.items() if (SS[0] + d[0], SS[1] + d[1]) == ptr[1]][0],
                                     pointer_equals_label=ptr == lab, pointer_len=len(ptr) - 1,
                                     pointer_reaches_G=ptr[-1] == GG)
                rows.append(row)
    return rows


def job_pcd(seed, h, n_mazes=200):
    import numpy as np
    import torch
    from cram import maze_unique as MU
    m = unique_model(seed)
    rng = np.random.default_rng(1000 + h)  # the same mazes for every seed
    mazes = []
    for i in range(n_mazes):
        g = tree_maze(h, rng)
        op = np.argwhere(g)
        a, b = rng.choice(len(op), 2, replace=False)
        mazes.append((g, tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b])))
    out = {'seed': seed, 'h': h, 'variants': {}}
    for var in ('ring', 'noring'):
        X, Y, meta = [], [], []
        for g, S, G in mazes:
            if var == 'noring':
                g = g[1:-1, 1:-1]; S = (S[0] - 1, S[1] - 1); G = (G[0] - 1, G[1] - 1)
            t = tokens(g, S, G)
            lab = np.zeros(g.shape, np.int64)
            for c in path_cells(g, S, G):
                lab[c] = 1
            Hh, Ww = g.shape
            n = side_open(g)
            tips = g & (n == 1) & (t == 1)
            edge = np.zeros(g.shape, bool); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
            X.append(t); Y.append(lab)
            meta.append(dict(open=int(g.sum()), edge_open=int((g & edge).sum()), edge_tips=int((tips & edge).sum()),
                             S_edge=on_edge(*S, Hh, Ww), G_edge=on_edge(*G, Hh, Ww), path_len=int(lab.sum()) - 1))
        x = torch.as_tensor(np.stack(X)); y = torch.as_tensor(np.stack(Y)); fr = x == MU.OPEN
        budget = max(20000, 3 * int(fr.flatten(1).sum(1).max()))
        gen = torch.Generator().manual_seed(0)
        t0 = time.time()
        r = MU.gibbs(m, x, y, fr, T=0.0, max_moves=budget, focus='random', gen=gen)
        secs = time.time() - t0
        pred = r['pred'].numpy(); yy = y.numpy(); frn = fr.numpy()
        rows = []
        for i in range(len(X)):
            diff = np.argwhere((pred[i] != yy[i]) & frn[i])
            Hh, Ww = pred[i].shape
            n = side_open(X[i] > 0)
            d = [dict(cell=[int(a), int(b)], pred=int(pred[i][a, b]), edge=on_edge(a, b, Hh, Ww),
                      open_side_nb=int(n[a, b])) for a, b in diff[:60]]
            rows.append(dict(meta[i], solved=bool(r['right'][i]), stopped=bool(r['stopped'][i]), moves=int(r['moves'][i]),
                             n_diff=int(len(diff)), n_diff_edge=int(sum(on_edge(a, b, Hh, Ww) for a, b in diff)), diff=d))
        out['variants'][var] = {'side': int(X[0].shape[0]), 'budget': budget, 'seconds': secs, 'mazes': rows}
    return out


def job_loops(h, n_mazes=100, seed=0, frac=0.05):
    import numpy as np
    from pointer_test import bfs_dist, greedy, ORDER
    m = hard_model(seed)
    rng = np.random.default_rng(5000 + h)
    X, labs, meta = [], [], []
    for i in range(n_mazes):
        g = tree_maze(h, rng)
        Hh = g.shape[0]
        cand = []
        for r in range(1, Hh - 1):
            for c in range(1, Hh - 1):
                if g[r, c]:
                    continue
                if (g[r - 1, c] and g[r + 1, c]) or (g[r, c - 1] and g[r, c + 1]):
                    cand.append((r, c))
        pick = rng.choice(len(cand), int(round(frac * len(cand))), replace=False)
        for j in pick:
            g[cand[j]] = True
        op = np.argwhere(g)
        a, b = rng.choice(len(op), 2, replace=False)
        S, G = tuple(int(v) for v in op[a]), tuple(int(v) for v in op[b])
        dG = bfs_dist(g, G)
        lab = greedy(g, S, G, lambda kk, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -ORDER.index(kk)))
        ll = np.zeros(g.shape, bool)
        for c in lab:
            ll[c] = True
        X.append(tokens(g, S, G)); labs.append(ll)
        meta.append(dict(S=S, G=G, shortest=int(dG[S]), opened=int(len(pick))))
    side = X[0].shape[0]
    settle_max = int(np.ceil(4000 * side / 30)); dsteps = int(np.ceil(2048 * side / 30 / 16) * 16)
    rows = []
    t0 = time.time()
    for lo in range(0, n_mazes, 25):
        board, used, done, ps, psi, s = run_hard(m, np.stack(X[lo:lo + 25]), settle_max, dsteps)
        for j in range(len(board)):
            i = lo + j
            S, G = meta[i]['S'], meta[i]['G']
            valid, length = check_path(board[j], S, G)
            exact = bool((board[j] == labs[i]).all() and done[j])
            kind = 'exact' if exact else ('valid_shortest_different' if valid and length == meta[i]['shortest'] else 'other')
            rows.append(dict(meta[i], stopped=bool(done[j]), valid=valid, length=length, kind=kind, equals_label=bool((board[j] == labs[i]).all()),
                             descent_steps=int(used[j]), settle_steps=int(ps[j]), settle_hit_cap=int(ps[j]) >= settle_max))
    return {'h': h, 'side': side, 'settle_max': settle_max, 'descend_steps': dsteps, 'seconds': time.time() - t0, 'mazes': rows}


def job_hard_wrong(seed, idx=106):
    """The one Maze-Hard test maze each seed stops wrong on: which cells differ from the label, and are they on the edge."""
    import numpy as np
    from cram import maze_h3 as m3
    m = hard_model(seed)
    tok, y = m3.load('test')
    board, used, done, ps, psi, s = run_hard(m, tok[idx:idx + 1])
    b, lab, t = board[0], y[idx], tok[idx]
    diff = np.argwhere(b != lab)
    n = side_open(t > 0)
    return {'seed': seed, 'idx': idx, 'stopped': bool(done[0]),
            'diff': [dict(cell=[int(r), int(c)], pred=int(b[r, c]), edge=on_edge(r, c, 30, 30), open_side_nb=int(n[r, c]))
                     for r, c in diff],
            'edge_open_cells': int(sum(on_edge(r, c, 30, 30) for r, c in np.argwhere(t > 0))),
            'edge_tips': int(sum(on_edge(r, c, 30, 30) for r, c in np.argwhere((t == 1) & (n == 1))))}


@app.function(image=image, cpu=NCPU, memory=8192, timeout=3600, max_containers=20, scaledown_window=2, retries=0)
def job(kind, args):
    _setup()
    t0 = time.time()
    f = {'data': job_data, 'repro_hard': job_repro_hard, 'repro_unique': job_repro_unique, 'pab': job_pab,
         'pcd': job_pcd, 'loops': job_loops, 'hard_wrong': job_hard_wrong}[kind]
    try:
        res = f(*args)
        res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))  # plain types only
        err = None
    except Exception as e:  # report, do not retry
        import traceback
        res, err = None, traceback.format_exc()[-4000:]
    return {'kind': kind, 'args': list(args), 'seconds': time.time() - t0, 'result': res, 'error': err}


@app.local_entrypoint()
def main(which: str = 'all'):
    jobs = []
    if which in ('all', 'stage1'):
        jobs += [('data', ())] + [('repro_hard', (s,)) for s in range(3)] + [('repro_unique', (s,)) for s in range(3)]
    if which in ('all', 'stage2'):
        jobs += [('pab', (s,)) for s in range(3)]
        jobs += [('pcd', (s, h)) for s in range(3) for h in (15, 30, 60)]
        jobs += [('loops', (h,)) for h in (15, 30, 60)]
    if which == 'pcd':
        jobs += [('pcd', (s, h)) for s in range(3) for h in (15, 30, 60)]
    if which == 'stage3':
        jobs += [('hard_wrong', (s,)) for s in range(3)]
    t0 = time.time()
    outs = list(job.starmap(jobs, return_exceptions=True))
    ledger = []
    for (kind, args), o in zip(jobs, outs):
        if isinstance(o, Exception):
            print(kind, args, 'EXCEPTION', repr(o)[:500]); ledger.append({'kind': kind, 'args': list(args), 'error': repr(o)[:500]}); continue
        name = HERE / f"run_released_{kind}{''.join('_' + str(a) for a in args)}.json"
        name.write_text(json.dumps(o, indent=1))
        ledger.append({'kind': kind, 'args': list(args), 'seconds': o['seconds'], 'error': o['error'] is not None})
        print(kind, args, f"{o['seconds']:.0f} s", 'ERROR' if o['error'] else 'ok', flush=True)
        if o['error']:
            print(o['error'])
    led = HERE / 'run_released_ledger.json'
    old = json.loads(led.read_text()) if led.exists() else []
    old.append({'which': which, 'wall_s': time.time() - t0, 'jobs': ledger,
                'container_s': sum(j.get('seconds', 0) for j in ledger), 'cpu': NCPU, 'memory_gib': 8})
    led.write_text(json.dumps(old, indent=1))
    print('container seconds this call:', sum(j.get('seconds', 0) for j in ledger))
