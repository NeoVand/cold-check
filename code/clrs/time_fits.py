"""Fit wall clock on one laptop core (M4, one BLAS thread): completion + feature table + gate + greedy conjunction +
sparse logistic, from the training splits to the fitted rule. Best of 3 (the first includes numpy warm-up)."""
import json, time, sys
import run
TASKS = ['insertion_sort', 'bubble_sort', 'heapsort', 'quicksort', 'minimum', 'binary_search', 'quickselect',
         'activity_selector', 'task_scheduling', 'bfs', 'mst_kruskal', 'mst_prim', 'bellman_ford', 'dijkstra',
         'dag_shortest_paths', 'floyd_warshall', 'lcs_length', 'naive_string_matcher', 'kmp_matcher']
out = {}
for t in TASKS:
    ts = []
    for _ in range(1 if t == 'floyd_warshall' else 3):
        t0 = time.perf_counter(); run.fit_task(t, 20, 1e-3); ts.append(time.perf_counter() - t0)
    out[t] = dict(best_ms=round(1000 * min(ts), 1), runs_ms=[round(1000 * x, 1) for x in ts])
    print(t, out[t], flush=True)
json.dump(out, open('fit_times_m4.json', 'w'), indent=1)
