"""Training splits the local data folder lacks, made by gen.py exactly as the library's protocol makes them
(witness_clrs/generate.py: sizes 4..16, p drawn from 0.1..0.9, seed 10^7 (task index + 1) + 1000 L + chunk; the first
`num` inputs of chunk 0). Checked first on a task whose split is stored (dfs)."""
import sys, numpy as np
from pathlib import Path
import gen
from witness_clrs.tasks import TASKS
P_TRAIN = tuple(0.1 + 0.1 * i for i in range(9))
OUT = Path(__file__).parent / 'data'

def make_train(task, L, num):
    return gen.make(task, L, num, 10 ** 7 * (TASKS.index(task) + 1) + 1000 * L, p=P_TRAIN)

if __name__ == '__main__':
    off = dict(np.load(gen.DATA / 'dfs' / 'train_L16.npz')); mine = make_train('dfs', 16, 50)
    same = all(np.array_equal(mine[k][:50], off[k][:50]) for k in mine)
    print('dfs train_L16 first 50 reproduced:', same); assert same
    for task in sys.argv[1:]:
        (OUT / task).mkdir(parents=True, exist_ok=True)
        for L in (4, 7, 11, 13, 16):
            d = make_train(task, L, 200)
            M = 16; pad = {}
            for k, v in d.items():        # pad to 16 as the stored splits are
                if v.ndim >= 2:
                    w = [(0, 0)] + [(0, M - L)] * (v.ndim - 1)
                    v = np.pad(v, w, constant_values=(-1 if k in ('pi',) else 0))
                pad[k] = v
            np.savez_compressed(OUT / task / f'train_L{L}.npz', **pad)
        print(task, 'written')
