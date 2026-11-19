"""Fit wall clock, one M4 core, best of 3 (DFS family: 1 run), for the 11 tasks added after the freeze."""
import json, time
import extra, dp_extra, dfs2
out = {}
for t in ['graham_scan', 'jarvis_march', 'segments_intersect', 'find_maximum_subarray_kadane', 'articulation_points', 'bridges']:
    ts = [];
    for _ in range(3):
        t0 = time.perf_counter(); extra.fit_task(t); ts.append(time.perf_counter() - t0)
    out[t] = dict(best_ms=round(1000 * min(ts), 1)); print(t, out[t], flush=True)
for t in ['matrix_chain_order', 'optimal_bst']:
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); dp_extra.fit(t); ts.append(time.perf_counter() - t0)
    out[t] = dict(best_ms=round(1000 * min(ts), 1)); print(t, out[t], flush=True)
t0 = time.perf_counter(); dfs2.fit(); fd = time.perf_counter() - t0
t0 = time.perf_counter(); dfs2.fit_topo(); ft = time.perf_counter() - t0
out['dfs'] = dict(best_ms=round(1000 * fd, 1)); out['strongly_connected_components'] = dict(best_ms=round(1000 * fd, 1), note='the DFS fit only')
out['topological_sort'] = dict(best_ms=round(1000 * (fd + ft), 1), note='the DFS fit + its read-out fit')
print(out, flush=True); json.dump(out, open('fit_times_m4_extra.json', 'w'), indent=1)
