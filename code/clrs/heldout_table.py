"""The held-out section's table from results_heldout/ (seed 9000 + n: val, fresh 64, 1,024, sparse 1,024; FW 64, 256) and
results_heldout2/ (seed 9000 + n: fresh 256, 4,096, 5,120). Frozen code."""
import json
from pathlib import Path
import table
H1, H2 = Path('results_heldout'), Path('results_heldout2')
def g(r, *ks):
    return next((f"{100 * r[k]['score']:.2f} ({r[k]['num']})" for k in ks if r and k in r), '–')
print('| Task | official val | fresh 64 | fresh 256 | fresh 1,024 | fresh sparse 1,024 | fresh 4,096 | official 64 (not clean) |')
print('|---|---|---|---|---|---|---|---|')
n = miss = 0
for t in table.ORDER:
    a = json.load(open(H1 / f'{t}.json')); b = json.load(open(H2 / f'{t}.json')) if (H2 / f'{t}.json').exists() else None
    cells = [f"{100 * a['val']['score']:.2f}", g(a, 'gen64', 'gen80'), g(b, 'gen256', 'gen320') if b else g(a, 'gen256'),
             g(a, 'gen1024', 'gen1280'), g(a, 'gen1024_sparse4'), g(b, 'gen4096', 'gen5120')]
    for c in cells:
        if c != '–':
            n += 1; miss += not c.startswith('100.00')
    print(f"| {t} | " + ' | '.join(cells) + f" | {100 * a['test64']['score']:.2f} |")
print(f'\n{n} held-out scores, {miss} below 100%.')
