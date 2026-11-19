"""The 30 algorithms of CLRS-30 and what the block reads out for each.

Graph tasks receive a weighted graph (edge weights A, adjacency adj, node positions pos and, where the task has one, a
source node s). The other 18 tasks receive per-item numbers, masks or categories (arrays, intervals, strings, points);
for them the pair state starts on the complete graph over the items.

An output is one of these kinds:
  node    a pointer from every item to an item (parent pointers, the sorted order's predecessor)
  one     one item chosen out of all (minimum, the median, the first match)
  edge    a pointer from every pair to a third item (Floyd-Warshall's middle node, a split point)
  nmask   a yes/no per item           emask   a yes/no per pair (symmetric)
  ecat    one of four classes per pair (LCS's arrows)       gmask   one yes/no per input
"""

GRAPH_TASKS = ['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall', 'mst_prim', 'topological_sort',
               'dfs', 'strongly_connected_components', 'articulation_points', 'bridges', 'mst_kruskal']
SEQ_TASKS = ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'quickselect', 'binary_search',
             'find_maximum_subarray_kadane', 'matrix_chain_order', 'optimal_bst', 'lcs_length', 'activity_selector',
             'task_scheduling', 'naive_string_matcher', 'kmp_matcher', 'segments_intersect', 'graham_scan', 'jarvis_march']
TASKS = GRAPH_TASKS + SEQ_TASKS  # this order fixes the training-data seeds (see generate.py)

OUTPUTS = {
    'bfs': [('pi', 'node')], 'bellman_ford': [('pi', 'node')], 'dijkstra': [('pi', 'node')],
    'dag_shortest_paths': [('pi', 'node')], 'mst_prim': [('pi', 'node')], 'floyd_warshall': [('Pi', 'edge')],
    'topological_sort': [('topo', 'node'), ('topo_head', 'one')], 'dfs': [('pi', 'node')],
    'strongly_connected_components': [('scc_id', 'node')], 'articulation_points': [('is_cut', 'nmask')],
    'bridges': [('is_bridge', 'emask')], 'mst_kruskal': [('in_mst', 'emask')],
    'insertion_sort': [('pred', 'node')], 'bubble_sort': [('pred', 'node')], 'heapsort': [('pred', 'node')],
    'quicksort': [('pred', 'node')], 'minimum': [('min', 'one')], 'quickselect': [('median', 'one')],
    'binary_search': [('return', 'one')], 'find_maximum_subarray_kadane': [('start', 'one'), ('end', 'one')],
    'matrix_chain_order': [('s', 'edge')], 'optimal_bst': [('root', 'edge')], 'lcs_length': [('b', 'ecat')],
    'activity_selector': [('selected', 'nmask')], 'task_scheduling': [('selected', 'nmask')],
    'naive_string_matcher': [('match', 'one')], 'kmp_matcher': [('match', 'one')],
    'segments_intersect': [('intersect', 'gmask')], 'graham_scan': [('in_hull', 'nmask')],
    'jarvis_march': [('in_hull', 'nmask')],
}
MASKS = ('nmask', 'emask')
ECAT = 4  # lcs_length's arrows: the library's three directions and its 'blank' class (never a target)

NAMES = {'activity_selector': 'Activity Selector', 'articulation_points': 'Articulation Points', 'bellman_ford': 'Bellman-Ford',
         'bfs': 'BFS', 'binary_search': 'Binary Search', 'bridges': 'Bridges', 'bubble_sort': 'Bubble Sort',
         'dag_shortest_paths': 'DAG Shortest Paths', 'dfs': 'DFS', 'dijkstra': 'Dijkstra',
         'find_maximum_subarray_kadane': 'Find Max. Subarray', 'floyd_warshall': 'Floyd-Warshall', 'graham_scan': 'Graham Scan',
         'heapsort': 'Heapsort', 'insertion_sort': 'Insertion Sort', 'jarvis_march': "Jarvis' March", 'kmp_matcher': 'KMP Matcher',
         'lcs_length': 'LCS Length', 'matrix_chain_order': 'Matrix Chain Order', 'minimum': 'Minimum', 'mst_kruskal': 'MST-Kruskal',
         'mst_prim': 'MST-Prim', 'naive_string_matcher': 'Naive String Match', 'optimal_bst': 'Optimal BST',
         'quickselect': 'Quickselect', 'quicksort': 'Quicksort', 'segments_intersect': 'Segments Intersect',
         'strongly_connected_components': 'SCC', 'task_scheduling': 'Task Scheduling', 'topological_sort': 'Topological Sort'}

# Input numbers measured in the same units share one register kind (the compare-first block compares them with each other).
SAME_UNITS = {'binary_search': [['key', 'target']], 'activity_selector': [['f', 's']], 'task_scheduling': [['pos', 'd']]}

# Arithmetic of the exact pair-number block, chosen by the recodings that leave the task's answer alone:
# 'plus' (path sums: node potentials and a common factor) or 'max' (bottlenecks: any increasing map).
PAIR_NUMBER_TASKS = {'bellman_ford': 'plus', 'dijkstra': 'plus', 'dag_shortest_paths': 'plus', 'mst_kruskal': 'max'}

# FloydNet (arXiv 2601.19094), Table 9: one model per task, best of runs with and without hints; mean 96.64.
FLOYDNET = {'activity_selector': 95.30, 'articulation_points': 100.00, 'bellman_ford': 100.00, 'bfs': 100.00,
            'binary_search': 94.47, 'bridges': 100.00, 'bubble_sort': 100.00, 'dag_shortest_paths': 100.00, 'dfs': 100.00,
            'dijkstra': 97.67, 'find_maximum_subarray_kadane': 86.20, 'floyd_warshall': 96.20, 'graham_scan': 99.63,
            'heapsort': 100.00, 'insertion_sort': 100.00, 'jarvis_march': 99.50, 'kmp_matcher': 99.43, 'lcs_length': 95.70,
            'matrix_chain_order': 91.50, 'minimum': 100.00, 'mst_kruskal': 99.63, 'mst_prim': 99.67,
            'naive_string_matcher': 100.00, 'optimal_bst': 82.87, 'quickselect': 80.33, 'quicksort': 100.00,
            'segments_intersect': 99.90, 'strongly_connected_components': 90.07, 'task_scheduling': 91.05,
            'topological_sort': 100.00}


def is_graph(task):
    return task in GRAPH_TASKS


# Constraint problems beyond CLRS-30 (two of the bounded-width classes): one answer per input, "unsatisfiable or not",
# stored at the input's global item s and read at that item's own pair.
#   two_sat        2-SAT on n variables: 2n literal items plus the global item; A holds the implications of the clauses,
#                  adj pairs each literal with its negation.
#   two_colouring  can n variables, related by "same" (A) and "different" (adj) constraints, take two colours?
CSP_TASKS = ['two_sat', 'two_colouring']
OUTPUTS |= {t: [('unsat', 'nmask')] for t in CSP_TASKS}
NAMES |= {'two_sat': '2-SAT', 'two_colouring': 'Two-colouring'}
#   parity         a registered failure (positions given): parity equations on a graph whose vertices have three edges,
#                  two hidden-label variables per edge; the twisted instance is unsolvable and no count tells it from the plain one.
OUTPUTS['parity'] = [('unsat', 'nmask')]
NAMES['parity'] = 'Parity (Tseitin)'
