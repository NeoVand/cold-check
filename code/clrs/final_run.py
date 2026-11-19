"""The final battery: every CLRS-30 task, frozen code, one go (official test, val, the held-out seeds 9000 + n).
Dispatches to the frozen runners (run.py, extra.py, dp_extra.py, dfs2.py); no method code here.
    python final_run.py <task>      writes final/<task>.json (dfs, topological_sort, scc: the DFS family together)"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
SEED = 9000
G = '64:32,256:16,1024:16,1024:8:4'
SPEC = {**{t: '64:32,256:32,1024:32,4096:4' for t in ('insertion_sort', 'bubble_sort', 'heapsort', 'quicksort')},
        **{t: '64:2048,256:256,1024:256,4096:64' for t in ('minimum', 'binary_search', 'quickselect')},
        'activity_selector': '64:32,256:32,1024:32,4096:8', 'task_scheduling': '64:32,256:32,1024:32,4096:8',
        **{t: G for t in ('bfs', 'mst_kruskal', 'mst_prim', 'bellman_ford', 'dijkstra', 'dag_shortest_paths')},
        'floyd_warshall': '64:8,256:4', 'lcs_length': '64:32,256:32,1024:8',
        'naive_string_matcher': '80:2048,320:64,1280:64,5120:4', 'kmp_matcher': '80:2048,320:64,1280:64,5120:4',
        'graham_scan': '64:32,256:32,1024:8', 'jarvis_march': '64:32,256:32,1024:8', 'segments_intersect': '4:2048',
        'find_maximum_subarray_kadane': '64:1024,256:256,1024:64,4096:16', 'articulation_points': '64:32,256:16,1024:4',
        'bridges': '64:32,256:16', 'matrix_chain_order': '64:32,128:4', 'optimal_bst': '64:32,128:4',
        'dfs_family': '64:32,256:4'}
EXTRA = ('graham_scan', 'jarvis_march', 'segments_intersect', 'find_maximum_subarray_kadane', 'articulation_points', 'bridges')
DP = ('matrix_chain_order', 'optimal_bst')


def go(task, log=print):
    os.environ['HELDOUT_VAL'] = '1'
    if task == 'dfs_family':
        import dfs2
        r, _ = dfs2.go(SPEC[task], SEED, True, log=log); out = [r] + list(dfs2.go_topo_scc(SPEC[task], SEED, True, log=log).values())
    elif task in EXTRA:
        import extra; out = [extra.go(task, SPEC[task], SEED, True, log=log)]
    elif task in DP:
        import dp_extra; out = [dp_extra.go(task, SPEC[task], SEED, True, log=log)]
    else:
        import run; out = [run.go(task, 20, SPEC[task], SEED, log=log)]
    return out


if __name__ == '__main__':
    (HERE / 'final').mkdir(exist_ok=True)
    for r in go(sys.argv[1], log=lambda m: print(m, flush=True)):
        json.dump(r, open(HERE / 'final' / f"{r['task']}.json", 'w'), indent=1, default=str)
