"""final/summary.json and the diff list: every score in final/ against the same test in the earlier runs
(results/, results_heldout/, results_heldout2/, results_extra_heldout/)."""
import json
from pathlib import Path
F = Path('final'); OLD = [Path(p) for p in ('results_extra_heldout', 'results_heldout2', 'results_heldout', 'results')]
fits = json.load(open(F / 'fit_times_m4.json'))['fits']
summary, diffs, n_scores, n_miss = {}, [], 0, 0
for f in sorted(F.glob('*.json')):
    if f.stem in ('summary', 'MANIFEST', 'fit_times_m4', 'ablation_level', 'shared'): continue
    r = json.load(open(f)); t = r['task']
    sc = {k: (v['score'], v.get('num')) for k, v in r.items() if k in ('test64', 'val') or k.startswith('gen')}
    loops = max(abs(v.get('loops_max', 1)) for k, v in r.items() if k in ('test64', 'val') or k.startswith('gen'))
    rest = sum(v.get('not_at_rest', 0) for k, v in r.items() if k in ('test64', 'val') or k.startswith('gen'))
    summary[t] = dict(scores={k: v[0] for k, v in sc.items()}, inputs={k: v[1] for k, v in sc.items()}, loops_max=loops,
                      not_at_rest=rest, fit_ms_m4=fits.get(t, {}).get('best_ms'))
    for k, (v, _) in sc.items():
        n_scores += 1; n_miss += v < 1.0
        for o in OLD:
            if (o / f'{t}.json').exists():
                old = json.load(open(o / f'{t}.json'))
                if k in old and 'score' in old[k] and (o.name != 'results' or k == 'test64'):
                    if abs(old[k]['score'] - v) > 1e-12: diffs.append(f'{t} {k}: final {v} vs {o.name} {old[k]["score"]}')
                    break
summary['_totals'] = dict(tasks=len(summary), scores=n_scores, below_100=n_miss)
json.dump(summary, open(F / 'summary.json', 'w'), indent=1)
print(summary['_totals']); print('diffs:', diffs or 'none')
