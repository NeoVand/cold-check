"""Collect the final run into final/summary.json and list every headline number that differs from the earlier runs
(final/DIFF.md). Reads only JSON files; runs nothing."""
import json
from pathlib import Path

FINAL = Path(__file__).resolve().parent
BENCH = FINAL.parent


def load(p):
    return json.loads(Path(p).read_text())


def main():
    se = load(FINAL / 'sudoku_extreme.json')
    lad = load(FINAL / 'sudoku_ladder_local.json')
    l25 = load(FINAL / 'sudoku_ladder_25.json')['summary']
    mz = load(FINAL / 'mazes.json')
    mu = load(FINAL / 'multi.json')
    pr = load(FINAL / 'proofs.json')
    sha = load(FINAL / 'ENGINE_SHA256.json')
    keep = ('solved', 'wrong_stops', 'budget_misses', 'no_solution', 'cold_only', 'cold_only_frac', 'moves_mean',
            'moves_median', 'moves_max', 'ops_mean', 'ops_plus_cmp_mean', 'fit_ms_warm_median7', 'fit_ms_cold',
            'wall_ms_mean', 'restarts_total', 'n_kept_types', 'relations', 'tau', 'wmin')
    summ = dict(
        engine_sha256=sha, budget=se['budget'], restart_base=se['restart_base'],
        sudoku_extreme=[{k: f[k] for k in keep if k in f} | {'split': f['split']} for f in se['fits']],
        ladder={k: v for k, v in lad.items() if k.startswith('side')} | {'side25': l25},
        maze_unique=dict(fit=mz['unique_fit'], test=mz['unique_test'], generated=mz['unique_generated']),
        maze_hard=dict(fit=mz['hard_fit'], test=mz['hard_test'], rooms={k: mz['hard_rooms'][k] for k in ('n', 'exact')},
                       loops=mz['hard_loops'], rule_on_unique=mz['hard_rule_on_unique_test']),
        queens=dict(fit=mu['queens_fit'], cells=mu['queens']), colour=dict(fit=mu['colour_fit'], cells=mu['colour']),
        proofs=pr)
    (FINAL / 'summary.json').write_text(json.dumps(summ, indent=1, default=str))

    # differences from the earlier runs
    old_se = load(BENCH / 'sudoku_extreme.json')
    old_lad = load(BENCH / 'sudoku_sizes_local.json')
    old_25 = load(BENCH / 'sudoku_sizes.json')['side25']
    old_25b = load(BENCH / 'sudoku25b.json')['rows']
    old_mz = load(BENCH / 'mazes.json')
    old_mu = load(BENCH / 'multi.json')
    rows = []

    def cmp(name, a, b, fmt='{}'):
        if a != b:
            rows.append(f'| {name} | {fmt.format(a)} | {fmt.format(b)} |')
        return a == b
    for f, g in zip(old_se['fits'], se['fits']):
        for k in ('solved', 'wrong_stops', 'budget_misses', 'cold_only', 'moves_median', 'moves_max'):
            cmp(f"Sudoku-Extreme {g['split']} {k}", f[k], g[k])
        for k in ('ops_mean', 'ops_plus_cmp_mean', 'moves_mean', 'fit_ms_warm_median7', 'wall_ms_mean'):
            cmp(f"Sudoku-Extreme {g['split']} {k}", round(f[k], 1), round(g[k], 1))
    for side in ('side4', 'side9', 'side16'):
        for k in ('solved_valid', 'cold_only', 'moves_mean', 'moves_max', 'ops_mean', 'budget_misses'):
            a, b = old_lad[side][k], lad[side][k]
            cmp(f'Sudoku {side} {k}', round(a, 1) if isinstance(a, float) else a, round(b, 1) if isinstance(b, float) else b)
    cmp('Sudoku side25 solved (first registered arm, 10^6)', old_25['solved_valid'], l25['solved_valid'])
    cmp('Sudoku side25 solved (S6b (b), 10^6)', sum(r['pairs_restarts']['good'] for r in old_25b), l25['solved_valid'])
    cmp('Sudoku side25 budget misses (S6b (b), 10^6)', sum(r['pairs_restarts']['status'] == 0 for r in old_25b), l25['budget_misses'])
    for k in ('exact', 'undecided_mazes', 'ops_mean'):
        cmp(f'Maze-Unique test {k}', old_mz['unique_test'][k], mz['unique_test'][k])
    for k, v in mz['unique_generated'].items():
        cmp(f'Maze-Unique generated {k} exact', old_mz['unique_generated'][k]['exact'], v['exact'])
    cmp('Maze-Hard test exact', old_mz['hard_test']['exact'], mz['hard_test']['exact'])
    cmp('Maze-Hard rooms exact', old_mz['hard_rooms']['exact'], mz['hard_rooms']['exact'])
    for k, v in mz['hard_loops'].items():
        cmp(f'Maze-Hard loops {k}', old_mz['hard_loops'][k]['exact'], v['exact'])
    cmp('Maze-Hard rule fit ms (warm)', round(old_mz['hard_fit']['fit_ms_warm_median7'], 1), round(mz['hard_fit']['fit_ms_warm_median7'], 1))
    cmp('Maze-Unique fit ms (warm)', round(old_mz['unique_fit']['fit_ms_warm_median7'], 1), round(mz['unique_fit']['fit_ms_warm_median7'], 1))
    for k, v in mu['queens'].items():
        for kk in ('solved', 'cold_only', 'moves_mean', 'moves_max'):
            cmp(f'N-Queens {k} {kk}', old_mu['queens'][k][kk], v[kk])
    for k, v in mu['colour'].items():
        for kk in ('solved', 'moves_mean', 'moves_max'):
            cmp(f'Colouring {k} {kk}', old_mu['colour'][k][kk], v[kk])
    text = ['# Final run: numbers that differ from the earlier runs', '',
            f"Engine frozen by sha256 (final/ENGINE_SHA256.json); budget {se['budget']:,} moves, Luby restarts base "
            f"{se['restart_base']:,}, unit pairs on, for every search. Fit times and wall times are new measurements and "
            'always differ a little; they are listed too.', '', '| Number | Before | Final |', '|---|---|---|'] + rows
    (FINAL / 'DIFF.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))


if __name__ == '__main__':
    main()
