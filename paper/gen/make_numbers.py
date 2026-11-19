"""Every number in the manuscript, read from the result files.

Writes ../numbers.tex (one \\nmdef{key}{value} per number). The manuscript prints a number only through \\nm{key}; a
key that no file supplies prints as a red bracket, so a missing number can never pass for a measured one.

Run from the repository root:   python3 paper/gen/make_numbers.py
The result files are read from ../code/ (the layout of this repository: code/clrs, code/bench, code/walkthrough,
code/released). Set COLDCHECK_DATA to read them from another checkout with the same layout.

Two kinds of number are not read from our files, and each is marked where it is set:
  PUBLISHED  a number printed in another paper (CRAM, arXiv 2611.17309; our earlier paper, arXiv 2611.11937),
             with its table;
  RELEASED   a number read from CRAM's released checkpoints or printed by our analysis scripts of them
             (code/released/), which need CRAM's repository to rerun.
"""
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA = os.environ.get('COLDCHECK_DATA', os.path.join(os.path.dirname(PAPER), 'code'))
CLRS = os.path.join(DATA, 'clrs')
BENCH = os.path.join(DATA, 'bench')
BF = os.path.join(BENCH, 'final')
CF = os.path.join(DATA, 'clrs', 'final')
WALK = os.path.join(DATA, 'walkthrough')
REL = os.path.join(DATA, 'released')

N = {}


def put(key, val):
    N[key] = str(val)


def jl(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def exists(*p):
    return os.path.exists(os.path.join(*p))


def comma(x):
    return f'{int(round(x)):,}'


def sci(x, d=1):
    """2.0 x 10^9 style, as LaTeX."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return f'${m:.{d}f}\\times10^{{{e}}}$'


def ms(x):
    return f'{x:.0f}' if x >= 10 else f'{x:.1f}'


WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
         11: 'eleven', 12: 'twelve', 15: 'fifteen', 19: 'nineteen', 20: 'twenty', 30: 'thirty'}

# ---------------------------------------------------------------- Sudoku-Extreme (code/bench/final: the final run, engine frozen)
sx = jl(BF, 'sudoku_extreme.json')
fits = sx['fits']
assert all(f['solved'] == f['n_test'] and f['wrong_stops'] == 0 for f in fits)
put('sudoku.ntest', comma(fits[0]['n_test']))
put('sudoku.nfits', WORDS[len(fits)])
put('sudoku.wrong', sum(f['wrong_stops'] for f in fits))
warm = [f['fit_ms_warm_median7'] for f in fits]
put('sudoku.fit.lo', ms(min(warm)))
put('sudoku.fit.hi', ms(max(warm)))
put('sudoku.fit.cold.hi', ms(max(f['fit_ms_cold'] for f in fits)))
put('sudoku.kept', fits[0]['n_kept_types'])
put('sudoku.types', jl(BF, 'proofs.json')['P1_types_per_b']['3'])
assert len({f['n_kept_types'] for f in fits}) == 1
put('sudoku.cold.n', comma(fits[0]['cold_only']))
put('sudoku.cold.pct', f"{100 * fits[0]['cold_only_frac']:.1f}")
ops = st.mean(f['ops_mean'] for f in fits)
opsc = st.mean(f['ops_plus_cmp_mean'] for f in fits)
put('sudoku.ops', sci(fits[0]['ops_mean']))
put('sudoku.opscmp', sci(fits[0]['ops_plus_cmp_mean']))
put('sudoku.moves.median', f"{fits[0]['moves_median']:.0f}")
put('sudoku.moves.mean', f"{fits[0]['moves_mean']:.0f}")
put('sudoku.moves.max', comma(max(f['moves_max'] for f in fits)))
put('sudoku.wall.mean', f"{fits[0]['wall_ms_mean']:.2f}")
put('sudoku.wall.max', f"{max(f['wall_ms_max'] for f in fits):.0f}")
put('sudoku.wmin.lo', f"{min(f['wmin'] for f in fits):.2f}")
put('sudoku.tau', f"{fits[0]['tau']:g}")
put('sudoku.trainboards', '1,000')                       # CRAM's splits: 1,000 solved boards each (their App. C)

# PUBLISHED: CRAM, arXiv 2611.17309, Table 2 and abstract
CRAM = dict(sudoku_pct='99.996', sudoku_ops=2.02e9, sudoku_fit_ms=42, sudoku_fit_cold_s='0.59', handset_ops=1.08e8,
            sudoku_params='7.7K', unique_ops=8.37e5, unique_fit_ms=132, unique_fit_cpu_ms=83, hard_pct='99.9',
            hard_ops=1.41e7, hard_fit='under 30 s', trmp_pct='99.87', trmp_ops=4.57e12, trmp_fit_h='9.0',
            looped_fit_lo_h='1.2', looped_fit_hi_h='9.0', trmp_lattice_pct='98.90')
put('cram.sudoku.pct', CRAM['sudoku_pct'])
put('cram.sudoku.ops', sci(CRAM['sudoku_ops']))
put('cram.sudoku.fit', CRAM['sudoku_fit_ms'])
put('cram.sudoku.fit.cold', CRAM['sudoku_fit_cold_s'])
put('cram.handset.ops', sci(CRAM['handset_ops']))
put('cram.unique.ops', sci(CRAM['unique_ops']))
put('cram.unique.fit', CRAM['unique_fit_ms'])
put('cram.unique.fit.cpu', CRAM['unique_fit_cpu_ms'])
put('cram.hard.pct', CRAM['hard_pct'])
put('cram.hard.ops', sci(CRAM['hard_ops']))
put('cram.hard.fit', CRAM['hard_fit'])
put('trmp.pct', CRAM['trmp_pct'])
put('trmp.ops', sci(CRAM['trmp_ops']))
put('trmp.fit', CRAM['trmp_fit_h'])
put('trmp.lattice.pct', CRAM['trmp_lattice_pct'])
put('looped.fit.lo', CRAM['looped_fit_lo_h'])
put('looped.fit.hi', CRAM['looped_fit_hi_h'])
put('sudoku.ops.ratio', f"{round(CRAM['sudoku_ops'] / ops, -3):,.0f}")
put('sudoku.opscmp.ratio', f"{round(CRAM['sudoku_ops'] / opsc, -2):,.0f}")

# ---------------------------------------------------------------- the same rule at other sizes (final run)
sz = jl(BF, 'sudoku_ladder_local.json')
sz25 = jl(BF, 'sudoku_ladder_25.json')['summary']
assert sz['budget'] == sz25['budget'] == sx['budget']
for side in (4, 9, 16):
    d = sz[f'side{side}']
    assert d['wrong'] == 0
    put(f'size{side}.n', comma(d['n']))
    put(f'size{side}.solved', comma(d['solved_valid']))
    put(f'size{side}.cold', comma(d['cold_only']))
    put(f'size{side}.moves.median', comma(d['moves_median']))
put('size25.n', sz25['n'])
put('size25.final', sz25['solved_valid'])
put('size25.wrong', sz25['wrong'])
put('size25.moves.median', comma(sz25['moves_median']))
e = int(round(math.log10(sx['budget'])))
assert 10 ** e == sx['budget']
put('budget', f'$10^{{{e}}}$')
put('restart.base', comma(sx['restart_base']))

# ---------------------------------------------------------------- mazes
mz = jl(BF, 'mazes.json')
put('unique.n', comma(mz['unique_test']['n']))
put('unique.exact', comma(mz['unique_test']['exact']))
put('unique.fit', ms(mz['unique_fit']['fit_ms_warm_median7']))
put('unique.ops', sci(mz['unique_test']['ops_mean']))
put('unique.opscmp', sci(mz['unique_test']['ops_plus_cmp_mean']))
ug = mz['unique_generated']
noring = [v for k, v in ug.items() if not v['ring']]
put('edge.n', sum(v['n'] for v in noring))
put('edge.exact', sum(v['exact'] for v in noring))
put('edge.deadends', comma(sum(v['edge_dead_ends'] for v in noring)))
put('edge.sides', ', '.join(str(v['side']) for v in sorted(noring, key=lambda v: v['side'])[:-1])
    + ' and ' + str(max(v['side'] for v in noring)))
ring = [v for k, v in ug.items() if v['ring']]
put('ring.n', sum(v['n'] for v in ring))
put('ring.exact', sum(v['exact'] for v in ring))
ht = mz['hard_test']
put('hard.n', comma(ht['n']))
put('hard.exact', comma(ht['exact']))
assert ht['maze106_exact']
put('hard.fit', ms(mz['hard_fit']['fit_ms_warm_median7']))
put('hard.ops', sci(ht['ops_mean']))
put('hard.opscmp', sci(ht['ops_plus_cmp_mean']))
put('hard.steps', comma(mz['hard_fit']['n_steps']))
w = mz['hard_fit']['w']
for d in ('down', 'left', 'right', 'up'):
    put(f'hard.w.{d}', f"{w['dir:' + d]:.2f}".replace('-', '$-$'))
put('rooms.n', mz['hard_rooms']['n'])
put('rooms.exact', mz['hard_rooms']['exact'])
hl = mz['hard_loops']
put('loops.n', sum(v['n'] for v in hl.values()))
put('loops.exact', sum(v['exact'] for v in hl.values()))
put('hard.on.unique', comma(mz['hard_rule_on_unique_test']['exact']))

# ---------------------------------------------------------------- N-Queens and colouring
mu = jl(BF, 'multi.json')
q = mu['queens']
assert all(v['solved'] == v['n'] for v in q.values())
put('queens.fit', ms(mu['queens_fit']['fit_ms_warm_median7']))
put('queens.n', q['100_empty']['n'])
put('queens.cold80', q['80_k9']['cold_only'])
put('queens.cold100', q['100_k9']['cold_only'])
c = mu['colour']
put('colour.fit', ms(mu['colour_fit']['fit_ms_warm_median7']))
put('colour.solved', ', '.join(str(c[k]['solved']) for k in ('100_3.0', '100_4.5', '100_6.0')))
put('colour.solved.3', c['100_3.0']['solved'])
put('colour.solved.45', c['100_4.5']['solved'])
put('colour.solved.6', c['100_6.0']['solved'])

# ---------------------------------------------------------------- proofs (code/bench/proofs.json)
pr = jl(BF, 'proofs.json')
put('proof.mazes', comma(pr['P4_mazes_checked']))
assert pr['P4_range_violations'] == 0 and pr['P4_cold_exact'] == pr['P4_mazes_checked']
assert all(f['stop_sound'] and f['kept_equals_row_col_box'] for f in pr['P2_P3_fits'])
bs = sorted(int(b) for b in pr['P1_types_per_b'])
put('proof.b.lo', bs[0])
put('proof.b.hi', bs[-1])

# ---------------------------------------------------------------- breadth-first-search walk-through
bw = jl(WALK, 'completed_answer_bfs.json')


def pct(a, b):
    return f'{100 * a / b:.1f}'


o, dp = bw['order'], bw['depth']
put('bfs.fit', f"{o['fit_ms']:.0f}")
put('bfs.depth.64', pct(*dp['64']['at_answer']))
put('bfs.depth.1024', pct(*dp['1024']['at_answer']))
assert dp['64']['at_answer_valid'] == dp['64']['at_answer'][1]          # every depth parent is a valid BFS parent
put('bfs.order.64', pct(*o['64']['at_answer']))
put('bfs.asfitted.64', pct(o['64']['top'][0], o['64']['top'][1]))
put('bfs.kept.64', pct(o['64']['top_gate'][0], o['64']['top_gate'][1]))
put('bfs.kept.1024', pct(o['1024']['top_gate'][0], o['1024']['top_gate'][1]))
put('bfs.kept.loops', o['1024']['top_gate'][3])
put('bfs.zero.order.64', pct(o['64']['zero'][0], o['64']['zero'][1]))
put('bfs.zero.depth.64', pct(dp['64']['zero'][0], dp['64']['zero'][1]))
put('bfs.parents64', comma(o['64']['at_answer'][1]))
put('bfs.parents1024', comma(o['1024']['at_answer'][1]))
put('bfs.graphs1024', 'three')
FIGDATA = os.path.join(PAPER, 'figures', 'v1', 'data', 'bfs_run.json')
if os.path.exists(FIGDATA):
    br = jl(FIGDATA)
    r = br['runs1024'][0]
    put('bfs.run.loops', r['kept']['loops'])
    put('bfs.run.first', r['as_fitted']['settled'][0])
    put('bfs.run.firstwrong', r['as_fitted']['settled'][0] - r['as_fitted']['right'][0])
    put('bfs.run.final', r['as_fitted']['final_right'])
    put('bfs.run.n', comma(r['n'] - 1))
    put('bfs.run.stop', len(r['as_fitted']['settled']))
    put('bfs.w.n', len(br['w']))

# ---------------------------------------------------------------- CLRS-30 (code/clrs)
GROUP = dict(
    insertion_sort=1, bubble_sort=1, heapsort=1, quicksort=1, minimum=1, binary_search=1, quickselect=1,
    activity_selector=1, task_scheduling=1, bfs=1, mst_kruskal=1, mst_prim=1, lcs_length=1, naive_string_matcher=1,
    kmp_matcher=1,
    bellman_ford=2, dijkstra=2, dag_shortest_paths=2, floyd_warshall=2, graham_scan=2, jarvis_march=2,
    segments_intersect=2, matrix_chain_order=2, optimal_bst=2,
    find_maximum_subarray_kadane=3, articulation_points=3, bridges=3, dfs=3, topological_sort=3,
    strongly_connected_components=3)
SUM = {t: v for t, v in jl(CF, 'summary.json').items() if not t.startswith('_')}   # the final battery, code frozen
assert set(SUM) == set(GROUP)
tot = jl(CF, 'summary.json')['_totals']
assert tot['below_100'] == 0 and tot['tasks'] == len(GROUP)


def result(t):
    return jl(CF, t + '.json')


def fit_ms(t):
    return SUM[t]['fit_ms_m4']


assert all(SUM[t]['scores']['test64'] == 1.0 for t in GROUP)
for g in (1, 2, 3):
    ts = [t for t in GROUP if GROUP[t] == g]
    put(f'clrs.g{g}.n', len(ts))
    put(f'clrs.g{g}.w', WORDS.get(len(ts), len(ts)))
    f = [fit_ms(t) for t in ts]
    put(f'clrs.g{g}.fit.lo', f'{min(f):.0f} ms' if min(f) < 1000 else f'{min(f) / 1000:.1f} s')
    put(f'clrs.g{g}.fit.hi', f'{max(f):.0f} ms' if max(f) < 1000 else f'{max(f) / 1000:.1f} s')
    put(f'clrs.g{g}.loops', max(SUM[t]['loops_max'] for t in ts))
put('clrs.tasks', len(GROUP))
put('clrs.dfs.fit', f"{fit_ms('dfs') / 1000:.0f}")

# held-out sets: every score of the final battery except the official test
held = {t: [('final', k, v) for k, v in SUM[t]['scores'].items() if k != 'test64'] for t in GROUP}
put('held.sets', sum(len(v) for v in held.values()))
put('scores.total', tot['scores'])
misses = [(t, k, s) for t in held for _, k, s in held[t] if s != 1.0]
put('held.misses', len(misses))
assert not misses, misses
put('held.tasks', len(held))


def size_of(key):
    digits = ''.join(ch for ch in key.split('_')[0] if ch.isdigit())
    return int(digits) if digits else 0


GRAPHS = ('bfs', 'mst_kruskal', 'mst_prim')
g1sizes = {t: max(size_of(k) for _, k, _ in held[t]) for t in held if GROUP[t] == 1}
put('held.max.nodes', comma(max(g1sizes[t] for t in GRAPHS)))
put('held.max.items', comma(max(v for t, v in g1sizes.items() if t not in GRAPHS)))
put('held.max.arrays', comma(max(v for t, v in g1sizes.items() if t not in GRAPHS + ('naive_string_matcher', 'kmp_matcher', 'lcs_length'))))

# few-data test: one JSON line per task in clrs/logs/few_data.log
FEW = os.path.join(CLRS, 'logs', 'few_data.log')
if os.path.exists(FEW):
    rows = [json.loads(l) for l in open(FEW) if l.startswith('{')]
    put('few.tasks', len(rows))
    put('few.exact', sum(r['test64'] == 1.0 for r in rows))
    small = [r for r in rows if r['train_inputs'] == 10]
    put('few.fit.lo', f"{min(r['fit_ms'] for r in small):.0f}")
    put('few.fit.hi', f"{max(r['fit_ms'] for r in small):.0f}")
    by = {r['task']: r for r in rows}
    put('few.kmp', f"{100 * by['kmp_matcher']['test64']:.2f}")
    put('few.fw', f"{100 * by['floyd_warshall']['test64']:.2f}")
    put('few.fw.inputs', by['floyd_warshall']['train_inputs'])

cert = {c['task']: c for c in jl(CLRS, 'cert.json')}
put('cert.sorts.pairs', comma(cert['insertion_sort']['pairs_checked']))
put('cert.sorts.items', cert['insertion_sort']['max_items'])
put('cert.min.items', cert['minimum']['max_items'])
put('cert.bfs.cases', comma(cert['bfs']['pair_cases']))
put('cert.bfs.items', cert['bfs']['max_items'])
put('cert.kruskal.graphs', comma(cert['mst_kruskal']['random_graphs_vs_library']))
dfst = jl(CLRS, 'dfs_test.json')
put('dfs.answer', f"{100 * dfst['visible']['at_answer']:.0f}")
put('dfs.cold', f"{100 * dfst['visible']['cold']:.1f}")
put('dfs.prefix.fit', f"{fit_ms('dfs') / 1000:.0f}")
bl = result('bfs')['rules']['off']['info']
put('bfs.clrs.features', comma(bl['features']))
put('bfs.clrs.positives', comma(bl['positives']))
put('clrs.train.inputs', result('bfs')['train_inputs'])

gv = jl(CLRS, 'gen_verify.json')
assert all(all(v['test_reproduced'].values()) for v in gv.values())
put('gen.tasks', len(gv))
put('gen.labels', comma(sum(b for v in gv.values() for a, b in v['relabel_wrong_of'].values())))
assert sum(a for v in gv.values() for a, b in v['relabel_wrong_of'].values()) == 0

# PUBLISHED: our earlier paper, arXiv 2611.11937, Methods (Compute and cost) and Supplementary Table 4
put('zero.gen.minutes', 64)
put('zero.cold.minutes', 29)
put('zero.gen.mean', '97.04')

# ---------------------------------------------------------------- CRAM's released models (code/released)
pab = [jl(REL, f'run_released_pab_{s}.json')['result'] for s in range(3)]
wrong = [[r for r in p if not r['own']['equals_label']] for p in pab]
put('rel.rooms.n', len(pab[0]))
put('rel.rooms.wrong', '/'.join(sorted({str(len(x)) for x in wrong})))
longer = [[r for r in x if r['own']['valid_path'] and r['own']['length'] > r['shortest']] for x in wrong]
put('rel.rooms.longer.lo', min(len(x) for x in longer))
put('rel.rooms.longer.hi', max(len(x) for x in longer))
put('rel.rooms.first', min(r['n'] for x in wrong for r in x))
pcd_ring = pcd_noring = pcd_n = 0
for s in range(3):
    for h in (15, 30, 60):
        res = jl(REL, f'run_released_pcd_{s}_{h}.json')['result']
        for name, var in res['variants'].items():
            k = sum(m['solved'] for m in var['mazes'])
            if name == 'ring':
                pcd_ring += k
            else:
                pcd_noring += k
            if s == 0 and name == 'ring':
                pcd_n += len(var['mazes'])
assert pcd_ring == 3 * pcd_n
put('rel.ring.solved', pcd_ring // 3)                   # per seed; every seed solves every ringed maze
put('rel.noring.solved', pcd_noring)                    # summed over the three seeds
put('rel.pcd.n', pcd_n)
sc = jl(REL, 'sudoku_stop_cert.json')
put('rel.stop.bound.lo', f"{min(v['bound'] for v in sc.values()):.2f}")
put('rel.stop.bound.hi', f"{max(v['bound'] for v in sc.values()):.2f}")
put('rel.stop.thr', f"{max(v['threshold'] for v in sc.values()):.3f}")
assert all(v['proved'] for v in sc.values())
put('rel.stop.pay.lo', f"{min(v['two'] for v in sc.values()):.2f}")
put('rel.stop.earn.hi', f"{max(-v['LB_pairs'] for v in sc.values()):.2f}")
put('rel.stop.pay.hi', f"{max(v['two'] for v in sc.values()):.2f}")
put('rel.stop.earn.lo', f"{min(-v['LB_pairs'] for v in sc.values()):.2f}")
put('rel.stop.bound.hi', f"{max(v['bound'] for v in sc.values()):.2f}")
# RELEASED: CRAM's own released checkpoints (checkpoints/maze-hard/seed*/final.json, field latent.step_cost_corridor
# and latent.mass; curve[*].mass for the training log) and their Sudoku checkpoints (negative costs). Rerun with
# CRAM's repository at commit 62bbe83 to check.
put('rel.step.lo', '19.7')
put('rel.step.hi', '19.8')
put('rel.step.init', '5.4')
put('rel.mass.end', '$3.7\\times10^{8}$')
put('rel.dir.down', '9.35')
put('rel.dir.left', '1.69')
put('rel.dir.right', '$-$1.93')
put('rel.dir.up', '$-$9.37')
put('rel.neg.lo', 426)
put('rel.neg.hi', 471)
# RELEASED: printed by code/released/maze_unique_cert2.py on the three Maze-Unique checkpoints
put('rel.cert.interior', 256)
put('rel.cert.edge.bad', 12)
put('rel.cert.edge', 32)
put('rel.cert.corner.bad', 4)
put('rel.cert.corner', 8)
CRAMREPO = os.environ.get('CRAM_REPO')          # optional: CRAM's repository at commit 62bbe83, to check the above
if CRAMREPO:
    for sd in range(3):
        lat = jl(CRAMREPO, 'checkpoints', 'maze-hard', f'seed{sd}', 'final.json')['latent']
        assert float(N['rel.step.lo']) - 0.05 <= lat['step_cost_corridor'][0] <= float(N['rel.step.hi']) + 0.05

# ---------------------------------------------------------------- write
missing_src = [k for k, v in N.items() if v in ('', 'None')]
assert not missing_src, missing_src
with open(os.path.join(PAPER, 'numbers.tex'), 'w') as f:
    f.write('% Generated by gen/make_numbers.py from the result files. Do not edit by hand.\n')
    for k in sorted(N):
        f.write(f'\\nmdef{{{k}}}{{{N[k]}}}\n')
print(f'{len(N)} numbers written to numbers.tex (data: {"COLDCHECK_DATA" if "COLDCHECK_DATA" in os.environ else "../code"})')

# keys the manuscript uses that this file does not supply
used = set()
for p in glob.glob(os.path.join(PAPER, '*.tex')) + glob.glob(os.path.join(PAPER, 'sections', '*.tex')):
    import re
    used |= set(re.findall(r'\\nm\{([^}]+)\}', open(p).read()))
pending = sorted(used - set(N))
if pending:
    print('PENDING (used in the text, supplied by no file):', ', '.join(pending))

# ---------------------------------------------------------------- Supplementary Table S2: the held-out battery
COLS = [('val', ('val',)), ('64', ('gen64', 'gen80', 'gen4')), ('128 / 256', ('gen128', 'gen256', 'gen320')),
        ('1,024', ('gen1024', 'gen1280')), ('sparse 1,024', ('gen1024_sparse4',)), ('4,096', ('gen4096', 'gen5120'))]
TNAME = dict(
    insertion_sort='Insertion sort', bubble_sort='Bubble sort', heapsort='Heapsort', quicksort='Quicksort',
    minimum='Minimum', binary_search='Binary search', quickselect='Quickselect', activity_selector='Activity selection',
    task_scheduling='Task scheduling', bfs='Breadth-first search', mst_kruskal='Kruskal', mst_prim='Prim',
    lcs_length='LCS length', naive_string_matcher='Naive string matcher$^\\dagger$', kmp_matcher='KMP matcher$^\\dagger$',
    bellman_ford='Bellman--Ford', dijkstra='Dijkstra', dag_shortest_paths='DAG shortest paths',
    floyd_warshall='Floyd--Warshall', graham_scan='Graham scan', jarvis_march='Jarvis march',
    segments_intersect='Segments intersect$^\\ddagger$', matrix_chain_order='Matrix chain order', optimal_bst='Optimal BST',
    find_maximum_subarray_kadane='Maximum subarray', articulation_points='Articulation points', bridges='Bridges',
    dfs='Depth-first search', topological_sort='Topological sort', strongly_connected_components='Strong components')
cells = {}
for t in GROUP:
    for k, v in SUM[t]['scores'].items():
        if k != 'test64':
            cells[(t, k)] = dict(score=v, **({'num': SUM[t]['inputs'][k]} if SUM[t]['inputs'].get(k) else {}))
lines = ['% Generated by gen/make_numbers.py from code/clrs/final/summary.json.',
         '\\begin{tabular}{l' + 'r' * len(COLS) + '}\\toprule',
         'Task & ' + ' & '.join(c for c, _ in COLS) + ' \\\\\\midrule']
g_prev = None
for t in GROUP:
    if GROUP[t] != g_prev:
        lab = {1: 'Comparisons and counts', 2: 'One arithmetic primitive supplied',
               3: 'Algorithmic structure supplied'}[GROUP[t]]
        lines.append(f'\\multicolumn{{{len(COLS) + 1}}}{{l}}{{\\textit{{{lab}}}}} \\\\')
        g_prev = GROUP[t]
    row = [TNAME[t]]
    for _, ks in COLS:
        got = [(k, cells[(t, k)]) for k in ks if (t, k) in cells]
        if not got:
            row.append('--')
        else:
            k, v = got[0]
            s = f"{100 * v['score']:.2f}"
            if 'num' in v:
                s += f" ({v['num']})"
            if k == 'gen128':
                s += '$^\\S$'
            row.append(s)
    lines.append(' & '.join(row) + ' \\\\')
lines.append('\\bottomrule\\end{tabular}')
open(os.path.join(HERE, 'heldout_table.tex'), 'w').write('\n'.join(lines) + '\n')
print('wrote heldout_table.tex')
