"""CRAM+z's hidden layer and pointer, from the paper's Eq. (13)-(14) and the released numbers (checkpoints/maze-hard):
psi_i = -log z_i settles by  psi_i <- log(4 + m) - log(sum_(j~i) e^(-psi_j) + [i = G])   (their soft Bellman backup),
pointer of cell i = argmax over its side neighbours d of  beta_d - psi_(i+d).
This is the hidden layer and its read only; the answer flow of Eq. (15) is NOT run here.
Compared with: the label rule (shortest path, ties by down, left, right, up) and the cold read (exact distance, then
the same direction order), which is the label rule by construction."""
import json, sys, math
from collections import deque
import numpy as np

DIRS = {'up': (-1, 0), 'left': (0, -1), 'right': (0, 1), 'down': (1, 0)}
ORDER = ['down', 'left', 'right', 'up']           # the label's neighbour order
def params(seed):
    d = json.load(open(f'ext/CRAM/checkpoints/maze-hard/seed{seed}/final.json'))['latent']
    beta = dict(zip(['self', 'up', 'left', 'right', 'down'], d['beta']))   # aoffs = [(0,0),(-1,0),(0,-1),(0,1),(1,0)]
    return d['mass'][0], beta

def settle(open_, G, m, iters=None):
    H, W = open_.shape
    psi = np.full((H, W), np.inf); step = math.log(4 + m)
    n = 0
    while True:
        nb = np.full((4, H, W), np.inf)
        nb[0, 1:, :] = psi[:-1, :]; nb[1, :-1, :] = psi[1:, :]; nb[2, :, 1:] = psi[:, :-1]; nb[3, :, :-1] = psi[:, 1:]
        mn = nb.min(0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            lse = np.where(np.isfinite(mn), -mn + np.log(np.exp(-(nb - mn)).sum(0)), -np.inf)   # log sum_j e^(-psi_j)
        src = np.full((H, W), -np.inf); src[G] = 0.0
        new = step - np.logaddexp(lse, src)
        new[~open_] = np.inf
        n += 1
        if np.array_equal(np.isfinite(new), np.isfinite(psi)) and np.nanmax(np.abs(np.where(np.isfinite(new), new - psi, 0))) < 1e-9: return new, n
        psi = new
        if n > 20000: raise RuntimeError('no rest')

def bfs_dist(open_, src):
    H, W = open_.shape; d = np.full((H, W), -1); d[src] = 0; q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and open_[rr, cc] and d[rr, cc] < 0: d[rr, cc] = d[r, c] + 1; q.append((rr, cc))
    return d

def label_bfs_from_S(open_, S, G):
    """BFS from S, neighbours in ORDER, parent = first discoverer; the path read back from G."""
    H, W = open_.shape; par = {S: None}; q = deque([S])
    while q:
        u = q.popleft()
        for k in ORDER:
            v = (u[0] + DIRS[k][0], u[1] + DIRS[k][1])
            if 0 <= v[0] < H and 0 <= v[1] < W and open_[v] and v not in par: par[v] = u; q.append(v)
    p = [G]
    while p[-1] != S: p.append(par[p[-1]])
    return p[::-1]

def greedy(open_, S, G, score):
    """From S, step to the side neighbour of best score(cell, dirname, neighbour) until G (cap on steps)."""
    H, W = open_.shape; p = [S]
    while p[-1] != G and len(p) <= H * W:
        r, c = p[-1]; best = None
        for k, (dr, dc) in DIRS.items():
            v = (r + dr, c + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and open_[v]:
                s = score(k, v)
                if best is None or s > best[0]: best = (s, v)
        p.append(best[1])
    return p

def count_paths(open_, dG):
    """Number of shortest paths from each cell to G (exact integers)."""
    H, W = open_.shape; N = {}
    cells = sorted(((dG[r, c], r, c) for r in range(H) for c in range(W) if dG[r, c] >= 0))
    for d, r, c in cells:
        if d == 0: N[(r, c)] = 1; continue
        N[(r, c)] = sum(N[(r + dr, c + dc)] for dr, dc in DIRS.values()
                        if 0 <= r + dr < H and 0 <= c + dc < W and dG[r + dr, c + dc] == d - 1)
    return N

def run(open_, S, G, m, beta):
    dG = bfs_dist(open_, G)
    lab = greedy(open_, S, G, lambda k, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -ORDER.index(k)))   # cold read: distance, then direction
    psi, n = settle(open_, G, m)
    ptr = greedy(open_, S, G, lambda k, v: beta[k] - psi[v])
    return dict(label=lab, pointer=ptr, same=lab == ptr, shortest=len(ptr) == len(lab), settle=n, dG=dG, psi=psi)

def room_maze(w, corridor='left'):
    """S above G with a wall bar between them. One way round is a 1-wide corridor, the other an open room w wide.
    Both ways round have the same length. corridor='left': the label's preferred side (left before right) is the corridor."""
    a = 2                                   # half-length of the bar
    H, Wd = 2 * 2 + 3, (a + 1) + 1 + (a + w) + 2
    g = np.zeros((H + 2, Wd + 2), bool)     # wall border all round
    cS = 1 + (a + 1)                        # column of S and G
    r0, r1, r2 = 1, 1 + 3, 1 + 6            # rows of S, of the bar, of G
    # corridor side (left): row r0 from cS to cS-a-1, down to r2, back to cS
    cL = cS - a - 1
    g[r0, cL:cS + 1] = True; g[r0:r2 + 1, cL] = True; g[r2, cL:cS + 1] = True
    # room side (right): everything from column cS+1 .. cS+a+w, rows r0..r2, except the bar's row within the bar
    g[r0:r2 + 1, cS + 1:cS + a + 1 + w] = True
    g[r1, cS + 1:cS + a + 1] = False        # the bar (right half); left half of the bar is wall already
    g[r0 + 1:r2, cS] = False                # nothing straight down
    # keep the cells beside the bar on the room side from offering a shorter way: the bar spans a columns, room way goes round its end
    S, G = (r0, cS), (r2, cS)
    if corridor == 'right': g = g[:, ::-1].copy(); S, G = (S[0], g.shape[1] - 1 - S[1]), (G[0], g.shape[1] - 1 - G[1])
    return g, S, G

if __name__ == '__main__':
    out = []
    for seed in (0, 1, 2):
        m, beta = params(seed)
        print(f'seed {seed}: step cost ln(4+m) = {math.log(4 + m):.2f}; direction scores', {k: round(v, 2) for k, v in beta.items()})
        print(f'   gaps: down-left {beta["down"] - beta["left"]:.2f}, left-right {beta["left"] - beta["right"]:.2f}, right-up {beta["right"] - beta["up"]:.2f}'
              f'  -> path-count ratios that outvote them: {math.exp(beta["down"] - beta["left"]):.0f}, {math.exp(beta["left"] - beta["right"]):.1f}, {math.exp(beta["right"] - beta["up"]):.0f}')
        for w in range(1, 11):
            g, S, G = room_maze(w)
            r = run(g, S, G, m, beta)
            N = count_paths(g, r['dG'])
            L, R = (S[0], S[1] - 1), (S[0], S[1] + 1)
            tie = r['dG'][L] == r['dG'][R]
            ratio = N[R] / N[L]
            margin = (beta['left'] - r['psi'][L]) - (beta['right'] - r['psi'][R])
            print(f'   room width {w:2d}: grid {g.shape}, tie at S {tie}, shortest paths via corridor {N[L]}, via room {N[R]}, ratio {ratio:8.1f}, '
                  f'ln ratio {math.log(ratio):5.2f}, pointer margin for the label side {margin:+.2f}, pointer path = label {r["same"]}, still a shortest path {r["shortest"]}')
            out.append(dict(seed=seed, w=w, tie=bool(tie), n_corridor=N[L], n_room=N[R], margin=float(margin), same=bool(r['same']), shortest=bool(r['shortest'])))
    json.dump(out, open('released/pointer_room.json', 'w'), indent=1)
