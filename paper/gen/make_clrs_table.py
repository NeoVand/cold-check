"""Build the CLRS-30 table from the final battery (no number typed by hand). Run: python3 paper/gen/make_clrs_table.py.
Source (under code/): clrs/final/summary.json, one entry per task: scores on the official test and every held-out set,
the most loops of any run, and the fit time on one laptop core. Our looped blocks' and FloydNet's numbers are from
arXiv 2611.11937 v3, Supplementary Table 4."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(os.environ.get('COLDCHECK_DATA', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'code')), 'clrs')
# task: (display name, tier, what is supplied by hand)
T = dict(
    insertion_sort=('Insertion sort', 1, '--'), bubble_sort=('Bubble sort', 1, '--'), heapsort=('Heapsort', 1, '--'),
    quicksort=('Quicksort', 1, '--'), minimum=('Minimum', 1, '--'), binary_search=('Binary search', 1, '--'),
    quickselect=('Quickselect', 1, '--'), activity_selector=('Activity selection', 1, '--'),
    task_scheduling=('Task scheduling', 1, 'deadline $\\div n$'), bfs=('Breadth-first search', 1, '--'),
    mst_kruskal=('Kruskal', 1, '--'), mst_prim=('Prim', 1, '--'), lcs_length=('LCS length', 1, 'index shift'),
    naive_string_matcher=('Naive string matcher', 1, '--'), kmp_matcher=('KMP matcher', 1, '--'),
    bellman_ford=('Bellman--Ford', 2, 'sum'), dijkstra=('Dijkstra', 2, 'sum'), dag_shortest_paths=('DAG shortest paths', 2, 'sum'),
    floyd_warshall=('Floyd--Warshall', 2, 'sum'), graham_scan=('Graham scan', 2, 'orientation'),
    jarvis_march=('Jarvis march', 2, 'orientation'), segments_intersect=('Segments intersect', 2, 'orientation'),
    matrix_chain_order=('Matrix chain order', 2, 'products, sums'), optimal_bst=('Optimal BST', 2, 'sums'),
    find_maximum_subarray_kadane=('Maximum subarray', 3, 'prefix sums'), articulation_points=('Articulation points', 3, 'deletion closure'),
    bridges=('Bridges', 3, 'deletion closure'), dfs=('Depth-first search', 3, '``open\'\''),
    topological_sort=('Topological sort', 3, '``open\'\', finish order'), strongly_connected_components=('Strong components', 3, 'two passes'))
# arXiv 2611.11937 v3, Supp. Table 4: general (3 seeds), cold (seeds; None = not run), FloydNet reported
P = dict(insertion_sort=(100.00, 100.00, 100.00), bubble_sort=(100.00, 100.00, 100.00), heapsort=(100.00, 100.00, 100.00),
         quicksort=(99.98, 100.00, 100.00), minimum=(100.00, 100.00, 100.00), binary_search=(98.89, 100.00, 94.47),
         quickselect=(99.77, 100.00, 80.33), activity_selector=(99.40, 99.94, 95.30), task_scheduling=(89.00, 99.86, 91.05),
         bfs=(100.00, None, 100.00), mst_kruskal=(99.49, 100.00, 99.63), mst_prim=(98.93, 99.98, 99.67),
         lcs_length=(95.66, None, 95.70), naive_string_matcher=(99.07, None, 100.00), kmp_matcher=(99.15, None, 99.43),
         bellman_ford=(99.67, 100.00, 100.00), dijkstra=(99.63, 100.00, 97.67), dag_shortest_paths=(99.84, 100.00, 100.00),
         floyd_warshall=(99.71, 100.00, 96.20), graham_scan=(99.80, None, 99.63), jarvis_march=(99.92, None, 99.50),
         segments_intersect=(99.33, None, 99.90), matrix_chain_order=(99.80, None, 91.50), optimal_bst=(82.71, None, 82.87),
         find_maximum_subarray_kadane=(87.86, None, 86.20), articulation_points=(100.00, None, 100.00), bridges=(100.00, None, 100.00),
         dfs=(72.97, 100.00, 100.00), topological_sort=(99.99, 100.00, 100.00), strongly_connected_components=(90.61, 90.02, 90.07))
S = {t: v for t, v in json.load(open(f'{C}/final/summary.json')).items() if not t.startswith('_')}


def held(t):
    """Largest held-out size, and whether every held-out set of this task is exact."""
    sizes, ok = [], True
    for k, v in S[t]['scores'].items():
        if k == 'test64':
            continue
        ok &= v == 1.0
        n = ''.join(ch for ch in k.split('_')[0] if ch.isdigit())
        if n: sizes.append(int(n))
    return (max(sizes) if sizes else None), ok
def f2(x): return '--' if x is None else f'{x:.2f}'
rows, tier_prev, s64, sb, sf = [], None, [], [], []
for t, (name, tier, sup) in T.items():
    d = S[t]
    fit = d['fit_ms_m4']
    fit_s = f'{fit:.0f} ms' if fit < 1000 else f'{fit / 1000:.1f} s'
    big, ok = held(t)
    g, c, fn = P[t]; best = max(v for v in (g, c) if v is not None)
    sc = 100 * d['scores']['test64']; s64.append(sc); sb.append(best); sf.append(fn)
    if tier != tier_prev:
        label = {1: 'Comparisons and counts', 2: 'One arithmetic primitive supplied',
                 3: 'Algorithmic structure supplied'}[tier]
        if tier_prev is not None: rows.append('\\addlinespace[2pt]')
        rows.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}} \\\\')
        tier_prev = tier
    rows.append(f"{name} & {sup} & {fit_s} & {d['loops_max']} & {sc:.2f} & "
                f"{'fixed size' if t == 'segments_intersect' else ((f'{big:,}' if ok else 'miss') if big else 'pending')} & {f2(best)} & {f2(fn)} \\\\")
avg = sum(s64) / len(s64)
rows.append('\\midrule')
rows.append(f'Average ({len(s64)} tasks) & & & & {avg:.2f} & & {sum(sb)/len(sb):.2f} & {sum(sf)/len(sf):.2f} \\\\')
tex = ('\\begin{tabular}{llrrrrrr}\\toprule\n'
       'Task & Supplied & Fit & Loops & Official 64 & Exact to & Ours, looped & FloydNet \\\\\\midrule\n'
       + '\n'.join(rows) + '\n\\bottomrule\\end{tabular}\n')
open(os.path.join(os.path.dirname(__file__), 'clrs_table.tex'), 'w').write(tex)
print(tex)
