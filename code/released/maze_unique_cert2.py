"""Second pass: split the neighbourhood check by where the cell sits (interior, edge, corner), and by whether the
3x3 window could occur in a maze whose open cells form a tree (no 2x2 block of 'on' cells including the centre)."""
import zipfile, itertools, json
import numpy as np
def load(path):
    z = zipfile.ZipFile(path)
    st = {n.rsplit('/', 1)[1]: np.frombuffer(z.read(n), '<f4') for n in z.namelist() if '/data/' in n and not n.endswith('.pkl')}
    sizes = sorted((len(v), k) for k, v in st.items())
    return st[sizes[1][1]].reshape(4, 2, 2).astype(np.float64), st[sizes[0][1]].astype(np.float64)
HALF = [(0, 1), (1, -1), (1, 0), (1, 1)]
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SIDE = [(-1, 0), (0, -1), (0, 1), (1, 0)]
def absent_sets():
    rows = {'top': [d for d in NB if d[0] == -1], 'bottom': [d for d in NB if d[0] == 1],
            'left': [d for d in NB if d[1] == -1], 'right': [d for d in NB if d[1] == 1]}
    yield 'interior', set()
    for k, v in rows.items(): yield 'edge:' + k, set(v)
    for a in ('top', 'bottom'):
        for c in ('left', 'right'): yield 'corner:' + a + '-' + c, set(rows[a]) | set(rows[c])
for seed in (0, 1, 2):
    T, b = load(f'ext/CRAM/checkpoints/maze-unique/seed{seed}/model.pt')
    def gain(nb):
        g = -(b[1] - b[0])
        for k, d in enumerate(HALF):
            s = nb[d]
            if s is not None: g += T[k, 1, s] - T[k, 0, s]
            s = nb[(-d[0], -d[1])]
            if s is not None: g += T[k, s, 1] - T[k, s, 0]
        return g
    print('seed', seed, ' b =', b.round(3), ' gain of one neighbour (on minus off), by offset:')
    base = dict.fromkeys(NB, 0)
    print('    all 8 off: gain %.3f' % gain(base), '| per neighbour turned on:', {d: round(gain({**base, d: 1}) - gain(base), 3) for d in NB})
    print('    per neighbour absent instead of off:', {d: round(gain({**base, d: None}) - gain(base), 3) for d in NB})
    for name, ab in absent_sets():
        present = [d for d in NB if d not in ab]
        tot = bad = bad_tree = tot_tree = 0; worst = None; mm = 1e9
        for states in itertools.product((0, 1), repeat=len(present)):
            nb = dict.fromkeys(NB, None); nb.update(zip(present, states))
            n_on = sum(1 for d in SIDE if nb[d] == 1)
            g = gain(nb)
            # tree-compatible: no diagonal neighbour on together with both side neighbours that complete a 2x2 block with the centre
            tree = not any(nb[(dr, dc)] == 1 and nb[(dr, 0)] == 1 and nb[(0, dc)] == 1 for dr in (-1, 1) for dc in (-1, 1))
            tot += 1; tot_tree += tree
            ok = (g < 0) == (n_on >= 2)
            if not ok:
                bad += 1; bad_tree += tree
                if worst is None or abs(g) > abs(worst[1]): worst = ({d: nb[d] for d in present if nb[d]}, round(float(g), 3), n_on)
            else: mm = min(mm, abs(g))
        print(f'    {name:22s} patterns {tot:4d} violations {bad:4d} | tree-compatible {tot_tree:4d} violations {bad_tree:4d} | min margin where right {mm:.3f} | largest violation {worst}')
