"""The numbers of Fig. 5: fit times and the sizes at which one fitted rule is exact, ours, CRAM's and the looped
models'. Nothing typed: CRAM's Tables 1-3 are parsed from code/ext/cram.txt (fit times on one H200; CRAM's own two
values checked against its released results file and against the page's parser); ours are read from code/bench and
code/clrs final results (results.py: code/bench/final/summary.json, code/clrs/final/).
"""
import ast
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

import kit

PAPER = kit.ROOT / "paper"
CRAM = kit.CODE
J = lambda p: json.loads((CRAM / p).read_text())  # noqa: E731
TXT = (CRAM / "ext" / "cram.txt").read_text().splitlines()

# ------------------------------------------------------------------------------------------------ CLRS-30, tier 1 --
tree = ast.parse((PAPER / 'gen' / 'make_clrs_table.py').read_text())
T = next({kw.arg: ast.literal_eval(kw.value) for kw in n.value.keywords} for n in tree.body
         if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', None) == 'T')
tier1 = [t for t, (_, tier, _) in T.items() if tier == 1]
assert len(tier1) == 15, tier1
D = CRAM / 'clrs'
run_src = (D / 'run.py').read_text()
SIZES = ast.literal_eval(re.search(r'^SIZES = (\(.*?\))', run_src, re.M).group(1))
MATCHERS = ast.literal_eval(re.search(r'^MATCHERS = (\(.*?\))', run_src, re.M).group(1))
m20 = ast.literal_eval(re.search(r"sizes = (\(\d+,\)) if task in MATCHERS else SIZES", run_src).group(1))
fit_size = {t: max(m20) if t in MATCHERS else max(SIZES) for t in tier1}


import results as R  # noqa: E402
clrs = {}
for t in tier1:
    r = R.clrs(t)
    clrs[t] = dict(fit_s=r["fit_ms"] / 1e3, fit_size=fit_size[t], exact_to=r["exact_to"])

# ------------------------------------------------------------------------------------------- CRAM's benchmarks --
F = R.PUZ                                          # code/bench/final/summary.json
sud = F['sudoku_extreme']
assert len(sud) == 3 and all(f['solved'] == 412786 and f['wrong_stops'] == 0 for f in sud)
loc = F['ladder']
mzu, mzh, mq = F['maze_unique'], F['maze_hard'], F['queens']
assert mzu['test']['exact'] == mzu['test']['n'] and mzh['test']['exact'] == mzh['test']['n']

# training sizes: CRAM's Table 1 (answer grid M x C), the Maze-Hard data set's name, multi.py's N-Queens fit
t1 = next(i for i, l in enumerate(TXT) if l.startswith('Table 1:'))
mxc = next(l for l in TXT[t1:t1 + 12] if l.startswith('M ×C'))
M = [int(a) for a in re.findall(r'(\d+)×\d+', mxc)]
assert M == [81, 900, 64, 100, 8, 10], M          # Sudoku, Maze-Unique, N-Queens 8x8 (10x10), colouring 8 (10)
side_hard = J('released/run_released_data.json')['result']['maze_hard_test_shape'][1:]
q_src = (CRAM / 'bench' / 'multi.py').read_text()
qN = int(re.search(r'def queens_check\(.*?\n.*?\n\s+N = (\d+)', q_src).group(1))

sudoku_exact = [loc[f'side{s}']['side'] ** 2 for s in (4, 9, 16, 25) if loc[f'side{s}']['solved_valid'] == loc[f'side{s}']['n']]
assert len(sudoku_exact) == 4                                    # 25 x 25 included: 100 of 100
uni_exact = [v['side'] ** 2 for v in mzu['generated'].values() if v['exact'] == v['n']]
assert len(uni_exact) == len(mzu['generated'])
hard_exact = [int(k) ** 2 for k, v in mzh['loops'].items() if v['exact'] == v['n']]
assert len(hard_exact) == len(mzh['loops'])
q_exact = sorted({v['N'] ** 2 for v in mq['cells'].values() if v['solved'] == v['n']})
assert all(v['solved'] == v['n'] for v in mq['cells'].values())

ours = dict(
    sudoku=dict(fit_s=float(np.median([f['fit_ms_warm_median7'] for f in sud])) / 1e3,
                fits_s=[f['fit_ms_warm_median7'] / 1e3 for f in sud], fit_size=M[0], exact=sorted(set(sudoku_exact + [M[0]]))),
    unique=dict(fit_s=mzu['fit']['fit_ms_warm_median7'] / 1e3, fit_size=M[1], exact=sorted(set(uni_exact + [M[1]]))),
    hard=dict(fit_s=mzh['fit']['fit_ms_warm_median7'] / 1e3, fit_size=side_hard[0] * side_hard[1],
              exact=sorted(set(hard_exact + [side_hard[0] * side_hard[1]]))),
    queens=dict(fit_s=mq['fit']['fit_ms_warm_median7'] / 1e3, fit_size=qN * qN, exact=q_exact))


def table2():
    """CRAM Table 2 rows (the site's parser, export_results.table2): fit seconds on Sudoku-Extreme and Maze-Unique."""
    unit = lambda v, u: float(v) * (3600 if u == 'h' else 60 if u == 'min' else 1)
    start = next(i for i, l in enumerate(TXT) if l.startswith('Table 2:'))
    rows = {}
    for l in TXT[start:start + 40]:
        s = l.strip()
        if not re.match(r'^(HRM|TRM|GRAM|EqR|FRM|TRMP|EBR|CRAM|\(hand-set\))', s):
            continue
        name = re.match(r'^(\(hand-set\)‡|[A-Za-z]+§?(?: \[\d+\])?)', s).group(1)
        rest = s[len(name):].split(); i = 0; fits = []
        for _ in range(2):
            i += 1
            if rest[i].startswith('±'):
                i += 1
            i += 2
            if rest[i] == 'none':
                fits.append(None); i += 1
            else:
                fits.append(unit(rest[i], rest[i + 1])); i += 2
        rows[re.sub(r' \[\d+\]', '', name).rstrip('§‡')] = fits
    return rows


def table3_queens8():
    start = next(i for i, l in enumerate(TXT) if l.startswith('Table 3:'))
    head = next(l for l in TXT[start:start + 20] if 'HRM [62]' in l and 'CRAM' in l)
    names = [re.sub(r' \[\d+\]', '', x).strip() for x in re.findall(r'(HRM \[\d+\]|TRM \[\d+\]|GRAM \[\d+\]|EBR \[\d+\]|CRAM|\(hand-set\))', head)]
    assert names == ['HRM', 'TRM', 'GRAM', 'EBR', 'CRAM', '(hand-set)'], names
    i8 = next(i for i in range(start, start + 30) if TXT[i].strip() == '8×8')
    fl = next(l for l in TXT[i8:i8 + 6] if l.strip().startswith('Fit time'))
    toks = re.findall(r'([\d.]+) (min|h|s)\b|none', fl.split('Fit time', 1)[1])
    vals = [None if not v else float(v) * (3600 if u == 'h' else 60 if u == 'min' else 1) for v, u in toks]
    assert len(vals) == 6
    return dict(zip(names, vals))


t2 = table2()
site = {r['name']: r for r in json.loads((kit.ROOT / 'site' / 'static' / 'data' / 'results.json').read_text())['table2']}
for k, (a, b) in t2.items():                       # the same parse as the site's results card
    assert (site[k]['sudoku']['fit_s'] or None) == a and (site[k]['unique']['fit_s'] or None) == b, k
LOOPED = ['HRM', 'TRM', 'GRAM', 'EqR', 'FRM', 'TRMP', 'EBR']
t3 = table3_queens8()
ebr = J('ext/CRAM/results/ebr-comparison.json')['cram']
cram_fit = dict(sudoku=ebr['sudoku']['fit_s'], unique=ebr['mazeunique']['fit_s'], queens=t3['CRAM'])
assert round(cram_fit['sudoku'], 2) == t2['CRAM'][0] and round(cram_fit['unique'], 2) == t2['CRAM'][1]
assert abs(cram_fit['sudoku'] * 1e3 - 42) < 1 and abs(cram_fit['unique'] * 1e3 - 132) < 1      # its text: 42 and 132 ms
comp = dict(sudoku=dict(cram=cram_fit['sudoku'], looped={k: t2[k][0] for k in LOOPED}),
            unique=dict(cram=cram_fit['unique'], looped={k: t2[k][1] for k in LOOPED}),
            hard=dict(cram=None, looped={}),                         # CRAM's Table 8 gives no fit times
            queens=dict(cram=t3['CRAM'], looped={k: t3[k] for k in ('HRM', 'TRM', 'GRAM', 'EBR')}))
cram_size = dict(sudoku=[M[0]], unique=[M[1]], hard=[side_hard[0] * side_hard[1]], queens=[M[2], M[3]])

