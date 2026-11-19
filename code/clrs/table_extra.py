"""The 11 tasks' held-out table (results_extra_heldout/)."""
import json
from pathlib import Path
T = ['graham_scan', 'jarvis_march', 'segments_intersect', 'find_maximum_subarray_kadane', 'articulation_points', 'bridges',
     'matrix_chain_order', 'optimal_bst', 'dfs', 'topological_sort', 'strongly_connected_components']
M4 = json.load(open('fit_times_m4_extra.json'))
PAPER = {'graham_scan': (99.80, 99.63), 'jarvis_march': (99.92, 99.50), 'segments_intersect': (99.33, 99.90),
         'find_maximum_subarray_kadane': (87.86, 86.20), 'articulation_points': (100.00, 100.00), 'bridges': (100.00, 100.00),
         'matrix_chain_order': (99.80, 91.50), 'optimal_bst': (82.71, 82.87), 'dfs': (72.97, 100.00),
         'topological_sort': (99.99, 100.00), 'strongly_connected_components': (90.61, 90.07)}
PRIM = {'graham_scan': 'orientation sign; "no point right of i → j"', 'jarvis_march': 'orientation sign; "no point right of i → j"',
        'segments_intersect': 'orientation sign; read-out "every item passes"', 'find_maximum_subarray_kadane': 'prefix sums, prefix minimum, best sum ending at j',
        'articulation_points': 'components of G − v (closure of an input relation)', 'bridges': 'reach in G − e (closure of an input relation)',
        'matrix_chain_order': 'split costs (products, sums) on the cost table completed from the answer', 'optimal_bst': 'split costs (sums) on the cost table completed from the answer',
        'dfs': '"open" on prefixes of the answer\'s preorder (one node per loop)', 'topological_sort': 'DFS check + finish order read-out',
        'strongly_connected_components': 'DFS check twice (Kosaraju) + root closure'}
g = lambda r, *ks: next((f"{100 * r[k]['score']:.2f} ({r[k]['num']})" for k in ks if k in r), '–')
rows = []; n = miss = 0
for t in T:
    r = json.load(open(Path('results_extra_heldout') / f'{t}.json'))
    cells = [f"{100*r['test64']['score']:.2f}", f"{100*r['val']['score']:.2f}", g(r, 'gen64', 'gen4'), g(r, 'gen128', 'gen256'), g(r, 'gen1024'), g(r, 'gen4096')]
    for c in cells[1:]:
        if c != '–': n += 1; miss += not c.startswith('100.00')
    loops = max([abs(r['test64'].get('loops_max', 1))] + [abs(v.get('loops_max', 1)) for k, v in r.items() if k.startswith('gen') or k == 'val'])
    rows.append(f"| {t} | {PRIM[t]} | {M4[t]['best_ms']:.0f} | {loops} | " + ' | '.join(cells) + f" | {PAPER[t][0]:.2f} | {PAPER[t][1]:.2f} |")
hdr = ("| Task | Hand-supplied (a ceiling) | Fit, ms (one M4 core) | Loops (max) | official 64 | official val | fresh 64 (inputs) | fresh 128 / 256 | fresh 1,024 | fresh 4,096 | Paper: general | FloydNet |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|")
print(hdr); print("\n".join(rows)); print(f'\n{n} held-out scores, {miss} below 100%.')
