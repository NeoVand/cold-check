"""The results card's numbers, read from the result files (no number typed by hand). Standard library only.
    python3 export_results.py   ->  site/static/data/results.json
Ours (code/ of this repository): bench/sudoku_extreme.json, bench/mazes.json, bench/sudoku_sizes_*.json,
walkthrough/completed_answer_bfs.json, released/run_released_*.json, clrs/results*/.
CRAM: code/ext/CRAM/results/ebr-comparison.json (its own rows), results/op-counts.json, its table records (fetch.py).
Looped models: CRAM's Table 2 (Sudoku-Extreme, Maze-Unique) and Table 8 (Maze-Hard), parsed from the text of
CRAM's paper (pdftotext of arXiv:2611.17309; path in CRAM_PAPER_TXT, default code/ext/cram.txt).
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'code'
OUT = Path(__file__).resolve().parent.parent / 'static' / 'data' / 'results.json'
J = lambda p: json.loads((CODE / p).read_text())
txt = Path(os.environ.get('CRAM_PAPER_TXT', CODE / 'ext' / 'cram.txt')).read_text().splitlines()


def num_unit(tok, unit):
    v = float(tok)
    return v * 3600 if unit == 'h' else v * 60 if unit == 'min' else v


def parse_params(t):
    m = {'K': 1e3, 'M': 1e6}
    return float(t[:-1]) * m[t[-1]] if t[-1] in m else float(t)


def table2():
    start = next(i for i, l in enumerate(txt) if l.startswith('Table 2:'))
    rows = []
    for k, l in enumerate(txt[start:start + 40]):
        s = l.strip()
        if not s or not re.match(r'^(HRM|TRM|GRAM|EqR|FRM|TRMP|EBR|CRAM|\(hand-set\))', s):
            continue
        name = re.match(r'^(\(hand-set\)‡|[A-Za-z]+§?(?: \[\d+\])?)', s).group(1)
        rest = s[len(name):].split()
        cols = []
        i = 0
        for _ in range(2):
            sol = rest[i]; i += 1
            dag = sol.endswith('†')
            sol = float(sol.rstrip('†'))
            sd = None
            if rest[i].startswith('±'):
                sd = float(rest[i][1:]); i += 1
            params = parse_params(rest[i]); i += 1
            flops = float(rest[i]); i += 1
            if rest[i] == 'none':
                fit = 0.0; i += 1
            else:
                fit = num_unit(rest[i], rest[i + 1]); i += 2
            cols.append(dict(solved=sol, sd=sd, published=dag, params=params, flops=flops, fit_s=fit))
        rows.append(dict(name=re.sub(r' \[\d+\]', '', name).rstrip('§‡'), line=start + k + 1, sudoku=cols[0], unique=cols[1]))
    return rows


def table8():
    start = next(i for i, l in enumerate(txt) if l.startswith('Table 8:'))
    rows = []
    for k, l in enumerate(txt[start:start + 16]):
        s = l.strip()
        m = re.match(r'^(HRM|TRM|FPRM|NCA|EBR|CRAM\+z)†?(?: \[\d+\])?\s+([\d.]+)(?: ± ([\d.]+))?\s+([\d.e+]+)\s+([\d.e+]+)$', s)
        if m:
            rows.append(dict(name=m.group(1), line=start + k + 1, solved=float(m.group(2)),
                             sd=float(m.group(3)) if m.group(3) else None, published='†' in s, params=float(m.group(4)),
                             flops=float(m.group(5))))
    return rows


t2, t8 = table2(), table8()
assert [r['name'] for r in t2] == ['HRM', 'TRM', 'GRAM', 'EqR', 'FRM', 'TRMP', 'EBR', 'CRAM', '(hand-set)'], [r['name'] for r in t2]
assert [r['name'] for r in t8] == ['HRM', 'TRM', 'FPRM', 'NCA', 'EBR', 'CRAM+z'], [r['name'] for r in t8]
ebr = J('ext/CRAM/results/ebr-comparison.json')['cram']
opc = {(c['benchmark'], c['row']): c for c in J('ext/CRAM/results/op-counts.json')['cells']}
sud = J('bench/final/sudoku_extreme.json')['fits']   # the final run (frozen engine, budget 10^7)
mz = J('bench/final/mazes.json')
loc = J('bench/final/sudoku_ladder_local.json')
m25 = dict(side25=J('bench/final/sudoku_ladder_25.json')['summary'])  # the 100-puzzle Modal run
bfs = J('walkthrough/completed_answer_bfs.json')
cram_sud = J('ext/CRAM/experiments/runs/table/sudoku-extreme__cram-0-fit.json')
cram_mu = J('ext/CRAM/experiments/runs/table/maze-unique__cram-sgd-fg.json')
cram_mh = J('ext/CRAM/experiments/runs/table/maze-hard__cram-latent-zmu.json')
nz = lambda w: sum(1 for v in w if v != 0)

ours = dict(
    sudoku=dict(solved=[f.get('correct', f['solved']) for f in sud], n=sud[0]['n_test'], wrong_stops=sum(f['wrong_stops'] for f in sud),
                params=sud[0]['n_kept_types'], ops=sud[0]['ops_mean'], ops_cmp=sud[0]['ops_plus_cmp_mean'],
                fit_s=[f['fit_ms_warm_median7'] / 1e3 for f in sud], cold_only=sud[0]['cold_only'],
                moves_mean=sud[0]['moves_mean'], moves_median=sud[0]['moves_median'], moves_max=sud[0]['moves_max'],
                wall_ms_mean=sud[0]['wall_ms_mean']),
    unique=dict(solved=[mz['unique_test']['exact']], n=mz['unique_test']['n'], params=nz(mz['unique_fit']['theta']),
                ops=mz['unique_test']['ops_mean'], ops_cmp=mz['unique_test']['ops_plus_cmp_mean'],
                fit_s=[mz['unique_fit']['fit_ms_warm_median7'] / 1e3]),
    hard=dict(solved=[mz['hard_test']['exact']], n=mz['hard_test']['n'], params=nz(mz['hard_fit']['w'].values()),
              ops=mz['hard_test']['ops_mean'], ops_cmp=mz['hard_test']['ops_plus_cmp_mean'],
              fit_s=[mz['hard_fit']['fit_ms_warm_median7'] / 1e3], fit_s_cold=mz['hard_fit']['fit_ms_cold'] / 1e3))
cram = dict(
    sudoku=dict(solved_pct=ebr['sudoku']['solved'], per_seed=cram_sud['per_seed'], n=cram_sud['n_test'],
                params=cram_sud['params'], params_nonzero=ebr['sudoku']['params_nonzero'],
                flops=opc[('sudoku-extreme', 'cram-0-fit')]['new'], fit_s=ebr['sudoku']['fit_s'],
                budget_failures=[s['budget_failures'] for s in cram_sud['extra']['per_seed']]),
    unique=dict(solved_pct=ebr['mazeunique']['solved'], n=cram_mu['n_test'], params=ebr['mazeunique']['params'],
                flops=opc[('maze-unique', 'cram-sgd-fg')]['new'], fit_s=ebr['mazeunique']['fit_s']),
    hard=dict(solved_pct=ebr['mazehard']['solved'], n=cram_mh['n_test'], params=ebr['mazehard']['params'],
              flops=opc[('maze-hard', 'cram-latent-zmu')]['new'], fit_s=ebr['mazehard']['fit_s']))
sizes = dict(
    ours=dict(side4=loc['side4'], side9=loc['side9'], side16=loc['side16'], side25=m25['side25'],
              unique_generated=mz['unique_generated'], hard_rooms=mz['hard_rooms'], hard_loops=mz['hard_loops']))
bfs_rows = {n: bfs['order'][n]['top_gate'] for n in ('16', '64', '256', '1024')}
# CRAM's released models on the generated mazes (released/run_released_*.json, written by released/modal_cram_check.py)
cram_edge = {}
for h in (15, 30, 60):
    for seed in range(3):
        v = J(f'released/run_released_pcd_{seed}_{h}.json')['result']['variants']
        for var in ('ring', 'noring'):
            k = f"{v[var]['side']}_{var}"
            e = cram_edge.setdefault(k, dict(side=v[var]['side'], ring=var == 'ring', n_per_seed=len(v[var]['mazes']), solved_per_seed=[]))
            assert e['n_per_seed'] == len(v[var]['mazes'])
            e['solved_per_seed'].append(sum(m['solved'] for m in v[var]['mazes']))
cram_rooms = []
for seed in range(3):
    rows = J(f'released/run_released_pab_{seed}.json')['result']
    cram_rooms.append(dict(seed=seed, n=len(rows), equals_label=sum(r['own']['equals_label'] for r in rows),
                           longer=sum(r['own']['valid_path'] and r['own']['length'] > r['shortest'] for r in rows),
                           not_path=sum(not r['own']['valid_path'] for r in rows)))
cram_loops = {}
for h in (15, 30, 60):
    d = J(f'released/run_released_loops_{h}.json')['result']
    cram_loops[str(d['side'])] = dict(n=len(d['mazes']), exact=sum(m['kind'] == 'exact' for m in d['mazes']))
# CLRS-30 fitted at the answer, all 30 tasks. The tiers, the supplied primitives, and our looped blocks' and FloydNet's
# numbers (arXiv 2611.11937 v3, Supplementary Table 4) are read from the paper's own table script,
# paper/gen/make_clrs_table.py (dicts T and P, parsed, not executed: executing it writes the
# paper's .tex). The rules below are that script's: results from results_extra_heldout/ first, then results/; fit times
# from fit_times_m4*.json (one laptop core), else the run's own fit_ms (marked); "exact to" = the largest held-out size
# at which every held-out set of the task is exact.
TIERS = {1: 'Comparisons and counts only', 2: 'One arithmetic primitive supplied',
         3: 'Algorithmic structure supplied'}


def latex_plain(x):
    return (x.replace('$\\div n$', '÷ n').replace("``", '“').replace("''", '”').replace('--', '–'))


def clrs():
    import ast
    import glob
    import os
    D = CODE / 'clrs'
    src = (ROOT / 'paper' / 'gen' / 'make_clrs_table.py').read_text()
    tree = ast.parse(src)
    dicts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('T', 'P'):
            dicts[node.targets[0].id] = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    T, P = dicts['T'], dicts['P']
    FT = json.loads((D / 'final' / 'fit_times_m4.json').read_text())['fits']
    held_dirs = [str(D / 'final')]   # the final battery: official test, val and the held-out sizes, frozen code

    def load(t):
        q = D / 'final' / f'{t}.json'
        if q.exists():
            return json.loads(q.read_text())

    def held(t):
        sizes, ok, n = [], True, 0
        for d in held_dirs:
            q = f'{d}/{t}.json'
            if not os.path.exists(q):
                continue
            for k, v in json.loads(open(q).read()).items():
                if k == 'test64' or not isinstance(v, dict) or 'score' not in v:
                    continue
                n += 1
                ok &= v['score'] == 1.0
                digits = ''.join(ch for ch in k.split('_')[0] if ch.isdigit())
                if digits:
                    sizes.append(int(digits))
        return (max(sizes) if sizes else None), ok, n

    rows = []
    for t, (name, tier, sup) in T.items():
        d = load(t)
        if d is None:
            continue
        big, ok, nheld = held(t)
        g, c, fn = P[t]
        in_ft = t in FT
        rows.append(dict(key=t, task=latex_plain(name), tier=tier, tier_label=TIERS[tier],
                         supplied=None if sup == '--' else latex_plain(sup),
                         fit_ms=FT[t]['best_ms'] if in_ft else d['fit_ms'], fit_from_run=not in_ft,
                         fit_note=FT[t].get('note') if in_ft else None,
                         test64=100 * d['test64']['score'], loops_max=d['test64']['loops_max'],
                         exact_to='fixed size' if t == 'segments_intersect' else (big if ok else 'miss') if big else 'pending',
                         heldout_sets=nheld, heldout_all_exact=ok, strings=t.endswith('matcher'),
                         paper_general=g, paper_cold=c, paper_best=max(v for v in (g, c) if v is not None), floydnet=fn))
    first19 = {Path(f).stem for f in glob.glob(str(D / 'results_heldout*' / '*.json'))}
    g19 = [r for r in rows if r['key'] in first19]
    g11 = [r for r in rows if r['key'] not in first19]
    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    return dict(rows=rows, n_tasks_total=30, tiers=TIERS,
                tier_counts={k: sum(r['tier'] == k for r in rows) for k in TIERS},
                avg=dict(test64=mean('test64'), paper_best=mean('paper_best'), floydnet=mean('floydnet')),
                heldout=dict(first=dict(tasks=len(g19), sets=sum(r['heldout_sets'] for r in g19), all_exact=all(r['heldout_all_exact'] for r in g19)),
                             other=dict(tasks=len(g11), sets=sum(r['heldout_sets'] for r in g11), all_exact=all(r['heldout_all_exact'] for r in g11))),
                source='clrs/final/<task>.json (official test, val and held-out sizes, frozen code); fit times '
                       'clrs/final/fit_times_m4.json; tiers, supplied '
                       'primitives and comparison numbers from paper/gen/make_clrs_table.py (T, P; arXiv 2611.11937 v3, Supp. Table 4)')


OUT.write_text(json.dumps(dict(
    source=dict(ours='bench/final/sudoku_extreme.json; bench/final/mazes.json; bench/final/sudoku_ladder_local.json; bench/final/sudoku_ladder_25.json; walkthrough/completed_answer_bfs.json',
                cram='ext/CRAM/results/ebr-comparison.json (cram block); ext/CRAM/results/op-counts.json; ext/CRAM/experiments/runs/table/*.json',
                looped='CRAM paper (arXiv 2611.17309) Table 2 and Table 8, parsed from the text of the paper',
                conventions='ours: one CPU core, warm median of 7 fits; operations counted as CRAM counts them (arithmetic 1, comparisons free); ops_cmp counts every comparison too. CRAM: H200 GPU warm fit times; op-counts.json convention.'),
    table2=t2, table8=t8, ours=ours, cram=cram, sizes=sizes,
    cram_generated=dict(edge=cram_edge, rooms=cram_rooms, loops=cram_loops,
                        source="released/run_released_pcd_*.json, run_released_pab_*.json, run_released_loops_*.json"),
    bfs=dict(fit_ms=bfs['order']['fit_ms'], top_gate=bfs_rows, depth_at_answer_64=bfs['depth']['64']['at_answer'],
             order_top_no_gate_64=bfs['order']['64']['top']),
    clrs=clrs()), indent=1))
print('wrote', OUT)
