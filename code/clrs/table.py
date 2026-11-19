"""The per-task table from results/<task>.json, beside our paper's and FloydNet's numbers
(arXiv 2611.11937 v3, Supplementary Table 4: general block mean of 3 seeds, cold setting, shared block; FloydNet)."""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
PAPER = {  # task: (general, cold or None, shared, FloydNet reported)
    'bfs': (100.00, None, 100.00, 100.00), 'dfs': (72.97, 100.00, 79.62, 100.00), 'topological_sort': (99.99, 100.00, 99.67, 100.00),
    'articulation_points': (100.00, None, 100.00, 100.00), 'bridges': (100.00, None, 100.00, 100.00),
    'strongly_connected_components': (90.61, 90.02, 88.00, 90.07), 'bellman_ford': (99.67, 100.00, 99.51, 100.00),
    'dijkstra': (99.63, 100.00, 99.50, 97.67), 'dag_shortest_paths': (99.84, 100.00, 96.84, 100.00),
    'floyd_warshall': (99.71, 100.00, 99.22, 96.20), 'mst_kruskal': (99.49, 100.00, 99.28, 99.63),
    'mst_prim': (98.93, 99.98, 97.48, 99.67), 'insertion_sort': (100.00, 100.00, 100.00, 100.00),
    'bubble_sort': (100.00, 100.00, 99.98, 100.00), 'heapsort': (100.00, 100.00, 100.00, 100.00),
    'quicksort': (99.98, 100.00, 100.00, 100.00), 'minimum': (100.00, 100.00, 99.98, 100.00),
    'binary_search': (98.89, 100.00, 98.80, 94.47), 'quickselect': (99.77, 100.00, 99.82, 80.33),
    'activity_selector': (99.40, 99.94, 99.16, 95.30), 'task_scheduling': (89.00, 99.86, 99.07, 91.05),
    'find_maximum_subarray_kadane': (87.86, None, 87.55, 86.20), 'lcs_length': (95.66, None, 92.43, 95.70),
    'matrix_chain_order': (99.80, None, 99.64, 91.50), 'optimal_bst': (82.71, None, 82.44, 82.87),
    'kmp_matcher': (99.15, None, 99.63, 99.43), 'naive_string_matcher': (99.07, None, 99.69, 100.00),
    'graham_scan': (99.80, None, 99.56, 99.63), 'jarvis_march': (99.92, None, 99.52, 99.50),
    'segments_intersect': (99.33, None, 99.67, 99.90)}
COMPLETION = {'bfs': 'depth', 'mst_prim': 'bottleneck + depth', 'mst_kruskal': 'bottleneck', 'bellman_ford': 'dist (+ sum)',
              'dijkstra': 'dist (+ sum)', 'dag_shortest_paths': 'dist (+ sum)', 'floyd_warshall': 'dist (+ sum), per source',
              'activity_selector': 'none (level: finish time)', 'task_scheduling': 'none (level: weight; d / n)',
              'lcs_length': 'matches along the arrows + shift', 'naive_string_matcher': 'run closure (level: position)',
              'kmp_matcher': 'run closure (level: position)', 'binary_search': 'none (score decides)'}
ORDER = ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'binary_search', 'quickselect',
         'activity_selector', 'task_scheduling', 'bfs', 'mst_kruskal', 'mst_prim', 'bellman_ford', 'dijkstra',
         'dag_shortest_paths', 'floyd_warshall', 'lcs_length', 'naive_string_matcher', 'kmp_matcher']


def pct(x):
    return '–' if x is None else f'{100 * x:.2f}'


M4 = json.load(open(HERE / 'fit_times_m4.json')) if (HERE / 'fit_times_m4.json').exists() else {}


def row(task, r):
    g = PAPER.get(task, (None,) * 4)
    def big(*keys):
        for k in keys:
            if k in r:
                return f"{pct(r[k]['score'])} ({r[k]['num']})"
        return '–'
    loops = max([abs(r['test64']['loops_max'])] + [abs(r[k]['loops_max']) for k in r if k.startswith('gen')])
    rest = sum([r['test64']['not_at_rest']] + [r[k]['not_at_rest'] for k in r if k.startswith('gen')])
    fmt = lambda v: '–' if v is None else f'{v:.2f}'
    m4 = M4.get(task, {}).get('best_ms')
    return (f"| {task} | {COMPLETION.get(task, 'none')} | {m4 if m4 is not None else '–'} | {loops}{'' if not rest else f' ({rest} not at rest)'} | "
            f"{pct(r['test64']['score'])} | {big('gen256', 'gen320')} | {big('gen1024', 'gen1280')} | {big('gen1024_sparse4')} | "
            f"{big('gen4096', 'gen5120')} | {fmt(g[0])} | {fmt(g[1])} | {fmt(g[3])} |")


if __name__ == '__main__':
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results'
    print('| Task | Completion | Fit, ms (one M4 core) | Loops (max, any size) | 64 official | 256 (inputs) | 1,024 (inputs) | sparse 1,024 (inputs) | 4,096 (inputs) | Paper: general | Paper: cold | FloydNet |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for t in ORDER:
        f = src / f'{t}.json'
        if f.exists():
            print(row(t, json.load(open(f))))
