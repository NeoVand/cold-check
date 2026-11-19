#!/usr/bin/env bash
# The CLRS-30 runs, from this folder (the paper's numbers: modal_final.py, see ../README.md). Run from code/clrs. Each section is independent; the Modal
# sections run one task per CPU container (one BLAS thread, so fit times are one core) and write results*/<task>.json.
# Local runs: python run.py <task> --big 256:32,1024:8 writes results/<task>.json the same way, on your machine.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p logs
section=${1:-help}

case $section in
final)      # Table 3, first 19 tasks: fit at the answer, official 64-node test, our generator at larger sizes -> results/
  M() { modal run modal_clrs.py "$@"; }
  M --tasks insertion_sort,bubble_sort,heapsort,quicksort --big 256:32,1024:32,4096:4 > logs/final_sorts.log 2>&1 &
  M --tasks minimum,binary_search,quickselect --big 256:256,1024:256,4096:64 > logs/final_one.log 2>&1 &
  M --tasks activity_selector,task_scheduling --big 256:32,1024:32,4096:8 > logs/final_greedy.log 2>&1 &
  M --tasks bfs,mst_kruskal,mst_prim --big 256:32,1024:16,1024:8:4 > logs/final_trees.log 2>&1 &
  M --tasks bellman_ford,dijkstra,dag_shortest_paths --big 256:32,1024:8,1024:8:4 > logs/final_paths.log 2>&1 &
  M --tasks floyd_warshall --big 256:2 > logs/final_fw.log 2>&1 &
  M --tasks lcs_length --big 256:32,1024:4 > logs/final_lcs.log 2>&1 &
  M --tasks naive_string_matcher,kmp_matcher --big 320:64,1280:32,5120:4 > logs/final_match.log 2>&1 &
  wait ;;
heldout)    # the frozen-code held-out batteries of the first 19 tasks (seed 9000 + n, official val) -> results_heldout*/
  M() { modal run modal_clrs.py --seed 9000 --val --out results_heldout "$@"; }
  M --tasks insertion_sort,bubble_sort,heapsort,quicksort --big 64:32,1024:32 > logs/held_sorts.log 2>&1 &
  M --tasks minimum,binary_search,quickselect --big 64:2048,1024:256 > logs/held_one.log 2>&1 &
  M --tasks activity_selector,task_scheduling --big 64:32,1024:32 > logs/held_greedy.log 2>&1 &
  M --tasks bfs,mst_kruskal,mst_prim --big 64:32,1024:16,1024:8:4 > logs/held_trees.log 2>&1 &
  M --tasks bellman_ford,dijkstra,dag_shortest_paths --big 64:32,1024:16,1024:8:4 > logs/held_paths.log 2>&1 &
  M --tasks floyd_warshall --big 64:8,256:4 > logs/held_fw.log 2>&1 &
  M --tasks lcs_length --big 64:32,1024:8 > logs/held_lcs.log 2>&1 &
  M --tasks naive_string_matcher,kmp_matcher --big 80:2048,1280:64 > logs/held_match.log 2>&1 &
  wait
  M() { modal run modal_clrs.py --seed 9000 --out results_heldout2 "$@"; }
  M --tasks insertion_sort,bubble_sort,heapsort,quicksort --big 256:32,4096:4 > logs/held2_sorts.log 2>&1 &
  M --tasks minimum,binary_search,quickselect --big 256:256,4096:64 > logs/held2_one.log 2>&1 &
  M --tasks activity_selector,task_scheduling --big 256:32,4096:8 > logs/held2_greedy.log 2>&1 &
  M --tasks bfs,mst_kruskal,mst_prim,bellman_ford,dijkstra,dag_shortest_paths --big 256:16 > logs/held2_graphs.log 2>&1 &
  M --tasks lcs_length --big 256:32 > logs/held2_lcs.log 2>&1 &
  M --tasks naive_string_matcher,kmp_matcher --big 320:64,5120:4 > logs/held2_match.log 2>&1 &
  wait
  python heldout_table.py ;;
extra)      # Table 3, the other 11 tasks (official test, val and held-out sizes, frozen code) -> results_extra_heldout/
  M() { modal run modal_extra.py --seed 9000 --val --out results_extra_heldout "$@"; }
  M --tasks graham_scan,jarvis_march --big 64:32,256:32,1024:8 > logs/xheld_hull.log 2>&1 &
  M --tasks segments_intersect --big 4:2048 > logs/xheld_seg.log 2>&1 &
  M --tasks find_maximum_subarray_kadane --big 64:1024,256:256,1024:64,4096:16 > logs/xheld_kadane.log 2>&1 &
  M --tasks articulation_points --big 64:32,256:16,1024:4 > logs/xheld_art.log 2>&1 &
  M --tasks bridges --big 64:32,256:16 > logs/xheld_br.log 2>&1 &
  M --tasks matrix_chain_order,optimal_bst --big 64:32,128:4 > logs/xheld_dp.log 2>&1 &
  modal run modal_dfs.py --seed 9000 --val --out results_extra_heldout --tasks dfs,topo_scc --big 64:32,256:4 > logs/xheld_dfs.log 2>&1 &
  wait
  python table_extra.py ;;
times)      # fit wall clock on one core, best of 3 -> fit_times_m4*.json and final/fit_times_m4.json (ours: an M4 laptop)
  python time_fits.py && python time_fits_extra.py && python -c "
import json; a=json.load(open('fit_times_m4.json')); b=json.load(open('fit_times_m4_extra.json')); a.update(b)
json.dump(dict(note='one M4 core, one BLAS thread, best of 3 (Floyd-Warshall and the DFS family: 1 run)', fits=a), open('final/fit_times_m4.json','w'), indent=1); print(len(a), 'tasks')" ;;
few)        # 10 training inputs instead of 100 (Floyd-Warshall: 51) -> logs/few_data.log
  for t in insertion_sort bubble_sort heapsort quicksort minimum binary_search quickselect activity_selector task_scheduling \
           bfs mst_kruskal mst_prim bellman_ford dijkstra dag_shortest_paths floyd_warshall lcs_length naive_string_matcher kmp_matcher; do
    ps=2; [ $t = floyd_warshall ] && ps=5
    python -c "
import run, json
res = run.go('$t', $ps, '', 101, log=lambda m: None)
print(json.dumps(dict(task='$t', train_inputs=res['train_inputs'], fit_ms=res['fit_ms'], test64=res['test64']['score'], loops=res['test64']['loops_max'], rest=res['test64']['not_at_rest'])), flush=True)
" 2>/dev/null | tail -1
  done > logs/few_data.log ;;
gate)       # keep only the target's own conditions (default) or every condition, witnesses too -> sweep/
  mkdir -p sweep
  for scope in own all; do
    for t in insertion_sort minimum binary_search quickselect activity_selector task_scheduling bfs mst_kruskal mst_prim \
             bellman_ford dijkstra dag_shortest_paths floyd_warshall; do
      big="256:4"; [ $t = floyd_warshall ] && big=""
      GATE_SCOPE=$scope python -c "
import json, run
res = run.go('$t', 20, '$big', 101, log=lambda m: None)
json.dump(res, open('sweep/${scope}_$t.json', 'w'), default=str)
"
    done
  done ;;
proofs)     # finite certificates (sorts, minimum, Kruskal, BFS) -> cert.json; the fitted rules in words -> RULES.md
  python cert.py; python rules.py > /dev/null ;;
generator)  # our generator reproduces every official test split bit for bit -> gen_verify.json, gen_extra_verify*.json
  python gen.py verify; python gen_extra.py verify ;;
dfs)        # the registered DFS test at the completed answer (three readings) -> dfs_test.json
  python dfs_test.py ;;
tables)
  python table.py; python table_extra.py; python heldout_table.py ;;
*)
  echo "usage: ./reproduce.sh final|heldout|extra|times|few|gate|proofs|generator|dfs|tables" ;;
esac
