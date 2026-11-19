"""Room-and-corridor mazes for CRAM+z's hidden layer and pointer (released numbers; hidden layer and read only).
L layout: S top-right, G bottom-left; one way is a 1-wide corridor (down, then left), the other crosses an open h x w room
(left, then down). extra = 0: both ways equally long (a tie the label breaks by direction). extra = 2: the room way is two
steps longer, so the corridor is the only shortest way. The 8 symmetries of the square move the pair of directions."""
import json, math, sys
import numpy as np
sys.path.insert(0, 'released')
from pointer_test import params, settle, bfs_dist, greedy, count_paths, DIRS, ORDER

def L_maze(h, w, extra):
    off = 1 if extra else 0                       # one more column on the left for the jog
    H, W = h + 2, w + 2 + off
    g = np.zeros((H, W), bool)
    g[0:h, 0:w] = True                            # the room: rows 0..h-1, cols 0..w-1
    g[0, w] = True                                # door from S into the room (row 0)
    S = (0, w + 1)
    g[0:h + 2, w + 1] = True                      # corridor down the right edge
    if extra:  # exit door under the room's bottom-left cell, then one step right to G: the room way is 2 longer
        g[h, 0] = True; g[h + 1, 0] = True; G = (h + 1, 1); g[h + 1, 1:w + 2] = True
    else:
        g[h, 0] = True; G = (h + 1, 0); g[h + 1, 0:w + 2] = True
    g = np.pad(g, 1)                              # wall border all round
    return g, (S[0] + 1, S[1] + 1), (G[0] + 1, G[1] + 1)

def sym(g, S, G, k):
    """k in 0..7: rotate by 90 degrees k % 4 times, then transpose if k >= 4. Returns the transformed maze."""
    m = np.zeros(g.shape, np.int8); m[S] = 1; m[G] = 2
    g2, m2 = np.rot90(g, k % 4), np.rot90(m, k % 4)
    if k >= 4: g2, m2 = g2.T, m2.T
    S2 = tuple(int(x) for x in np.argwhere(m2 == 1)[0]); G2 = tuple(int(x) for x in np.argwhere(m2 == 2)[0])
    return np.ascontiguousarray(g2), S2, G2

def first_dir(a, b):
    d = (b[0] - a[0], b[1] - a[1]); return [k for k, v in DIRS.items() if v == d][0]

rows = []
for seed in (0, 1, 2):
    m, beta = params(seed); c = math.log(4 + m)
    for extra in (0, 2):
        for k in range(8):
            for n in ([3, 4, 5, 6, 7, 8, 9, 10, 12] if extra == 0 else [12, 16, 18, 20, 22, 24, 26]):
                g, S, G = sym(*L_maze(n, n, extra), k)
                dG = bfs_dist(g, G)
                lab = greedy(g, S, G, lambda kk, v: (-(dG[v] if dG[v] >= 0 else 10 ** 9), -ORDER.index(kk)))
                psi, _ = settle(g, G, m)
                ptr = greedy(g, S, G, lambda kk, v: beta[kk] - psi[v])
                # the two ways out of S: which is the corridor, which the room
                nb = [(kk, (S[0] + d[0], S[1] + d[1])) for kk, d in DIRS.items() if g[S[0] + d[0], S[1] + d[1]]]
                assert len(nb) == 2
                N = count_paths(g, dG)
                (ka, va), (kb, vb) = nb
                corr, room = ((ka, va), (kb, vb)) if N[va] == 1 and dG[va] <= dG[vb] else ((kb, vb), (ka, va))
                margin = (beta[corr[0]] - psi[corr[1]]) - (beta[room[0]] - psi[room[1]])   # > 0: pointer takes the corridor
                rows.append(dict(seed=seed, extra=extra, sym=k, n=n, grid=list(g.shape), corridor=corr[0], room=room[0],
                                 label_first=first_dir(lab[0], lab[1]), pointer_first=first_dir(ptr[0], ptr[1]),
                                 same=lab == ptr, pointer_len=len(ptr) - 1, shortest=int(dG[S]), margin=float(margin),
                                 ln_paths_room=float(math.log(math.comb(2 * n - 2, n - 1)))))
json.dump(rows, open('released/pointer_L.json', 'w'), indent=1)
print('seed extra corridor room | label goes | first room size where the pointer leaves the label | grid | pointer path length vs shortest | margin just before / after')
for seed in (0, 1, 2):
    for extra in (0, 2):
        seen = set()
        for k in range(8):
            rs = [r for r in rows if r['seed'] == seed and r['extra'] == extra and r['sym'] == k]
            key = (rs[0]['corridor'], rs[0]['room'])
            if key in seen: continue
            seen.add(key)
            flips = [r for r in rs if not r['same']]
            f = flips[0] if flips else None
            prev = [r for r in rs if r['same']]
            print(seed, extra, f'{key[0]:>5s} {key[1]:>5s} | label {rs[0]["label_first"]:>5s} |',
                  (f'room {f["n"]}x{f["n"]} grid {f["grid"]} | pointer {f["pointer_len"]} vs shortest {f["shortest"]} | margin {prev[-1]["margin"] if prev else float("nan"):+.2f} -> {f["margin"]:+.2f} (ln paths {f["ln_paths_room"]:.1f})'
                   if f else f'never up to room {rs[-1]["n"]} (margin {rs[-1]["margin"]:+.2f}, ln paths {rs[-1]["ln_paths_room"]:.1f})'))
