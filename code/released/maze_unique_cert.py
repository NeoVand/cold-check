"""Finite check on CRAM's released Maze-Unique weights (no torch, no unpickling: the raw float32 storages of the zip).
For every state of the 8 neighbours of a free open cell (on / off / absent beyond the border), is the cell's
lowest-energy state 'on' exactly when at least two of its four side neighbours are on?  Energy as cram/cram_maze.py:
E = sum_i sum_(d in half offsets) s_i^T T_d s_(i+d) - sum_(i free) b . s_i ; half offsets (0,1),(1,-1),(1,0),(1,1)."""
import sys, zipfile, itertools, json
import numpy as np

def load(path):
    z = zipfile.ZipFile(path)
    st = {n.rsplit('/', 1)[1]: np.frombuffer(z.read(n), '<f4') for n in z.namelist() if '/data/' in n and not n.endswith('.pkl')}
    sizes = sorted((len(v), k) for k, v in st.items())
    assert [s for s, _ in sizes] == [2, 16], sizes          # b (2,), T (4, 2, 2)
    return st[sizes[1][1]].reshape(4, 2, 2).astype(np.float64), st[sizes[0][1]].astype(np.float64)

HALF = [(0, 1), (1, -1), (1, 0), (1, 1)]
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SIDE = [(-1, 0), (0, -1), (0, 1), (1, 0)]
out = {}
for seed in (0, 1, 2):
    T, b = load(f'ext/CRAM/checkpoints/maze-unique/seed{seed}/model.pt')
    def gain(nb):  # E(cell on) - E(cell off); nb: dict offset -> 0 off, 1 on, None absent
        g = -(b[1] - b[0])
        for k, d in enumerate(HALF):
            s = nb[d]                       # cell is first index, neighbour at +d second
            if s is not None: g += T[k, 1, s] - T[k, 0, s]
            s = nb[(-d[0], -d[1])]          # neighbour at -d is first index, cell second
            if s is not None: g += T[k, s, 1] - T[k, s, 0]
        return g
    bad, margins, by_count = [], [], {}
    for states in itertools.product((0, 1, None), repeat=8):
        nb = dict(zip(NB, states))
        n_on = sum(1 for d in SIDE if nb[d] == 1)
        g = gain(nb)
        want_on = n_on >= 2
        if (g < 0) != want_on: bad.append((states, float(g)))
        margins.append(abs(g))
        lo, hi = by_count.get(n_on, (1e9, -1e9)); by_count[n_on] = (min(lo, g), max(hi, g))
    # units: mean coupling to a side neighbour = mean over the 4 sides of [gain(side on) - gain(side off)], others off
    base = dict.fromkeys(NB, 0)
    unit = -np.mean([gain({**base, d: 1}) - gain(base) for d in SIDE])
    out[seed] = dict(T=T.round(4).tolist(), b=b.round(4).tolist(), patterns=3 ** 8, violations=len(bad),
                     min_margin=float(min(margins)), unit=float(unit),
                     gain_range_by_on_sides={k: (round(v[0], 3), round(v[1], 3)) for k, v in sorted(by_count.items())},
                     first_violations=bad[:5])
    print('seed', seed, 'violations', len(bad), 'of', 3 ** 8, 'min |gain|', round(min(margins), 4), 'side unit', round(unit, 4))
    for k, v in sorted(by_count.items()): print('   on side neighbours =', k, ' gain(on - off) in [%.3f, %.3f]' % v)
    for s, g in bad[:5]: print('   violation', s, round(g, 4))
json.dump(out, open('released/maze_unique_cert.json', 'w'), indent=1)
