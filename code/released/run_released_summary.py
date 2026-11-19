"""Summarise the run_released_*.json results of modal_cram_check.py (our JSON only; no third-party code).
python3 released/run_released_summary.py > released/run_released_summary.txt"""
import json
from collections import defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent
L = {(r['seed'], r['extra'], r['sym'], r['n']): r for r in json.load(open(W / 'pointer_L.json'))}


def load(name):
    d = json.load(open(W / name))
    assert d['error'] is None, d['error']
    return d['result']


print('== P-a / P-b: Maze-Hard model on room-and-corridor mazes (own size; emb30 = embedded in a 30x30 wall board)')
agree_emb, n_emb, ptr_agree, ptr_n = 0, 0, 0, 0
for seed in range(3):
    rows = load(f'run_released_pab_{seed}.json')
    for r in rows:
        if 'emb30' in r:
            n_emb += 1
            o, e = r['own'], r['emb30']
            agree_emb += all(o[k] == e[k] for k in ('equals_label', 'valid_path', 'length', 'stopped', 'returned_first'))
        key = (seed, r['extra'], r['sym'], r['n'])
        if key in L:
            ptr_n += 1
            ptr_agree += (L[key]['pointer_first'] == r['own']['pointer_first']) and (L[key]['same'] == r['own']['pointer_equals_label'])
    for extra in (0, 2):
        print(f'-- seed {seed}, extra {extra}')
        for k in range(8):
            rs = [r for r in rows if r['extra'] == extra and r['sym'] == k]
            c = rs[0]
            cells = []
            for r in rs:
                o = r['own']
                if o['equals_label']:
                    t = 'L'
                elif o['valid_path'] and o['length'] == r['shortest']:
                    t = 'S'  # a different shortest path
                elif o['valid_path']:
                    t = f'+{o["length"] - r["shortest"]}'
                else:
                    t = 'X'  # not a path
                t += '' if o['stopped'] else '!'
                t += '' if o['pointer_equals_label'] else 'p'
                cells.append(f'{r["n"]}:{t}')
            print(f'   corridor {c["corridor"]:>5s} room {c["room"]:>5s} label {c["label_first"]:>5s} | ' + ' '.join(cells))
print('legend: L = returned the label; S = another shortest path; +k = valid path k longer; X = not a path; ! = did not stop; '
      'p = the hidden layer pointer path differs from the label')
print(f'own-size vs embedded-30x30 runs agree on {agree_emb} of {n_emb}')
print(f'model pointer (first step, equals label) agrees with pointer_L.json on {ptr_agree} of {ptr_n}')

print('\n== P-c / P-d: Maze-Unique model on spanning-tree mazes')
for h in (15, 30, 60):
    for seed in range(3):
        d = load(f'run_released_pcd_{seed}_{h}.json')
        for var in ('ring', 'noring'):
            v = d['variants'][var]
            m = v['mazes']
            tip = [x for x in m if x['edge_tips'] > 0]
            notip = [x for x in m if x['edge_tips'] == 0]
            fails = [x for x in m if not x['solved']]
            edge_only = sum(1 for x in fails if x['n_diff'] == x['n_diff_edge'])
            on_left = sum(1 for x in fails for c in x['diff'] if c['pred'] == 1)
            off_wrong = sum(1 for x in fails for c in x['diff'] if c['pred'] == 0)
            tip_diff = sum(1 for x in fails for c in x['diff'] if c['pred'] == 1 and c['edge'])
            print(f'h {h:2d} side {v["side"]:3d} seed {seed} {var:6s}: solved {sum(x["solved"] for x in m)}/{len(m)}, '
                  f'stopped {sum(x["stopped"] for x in m)}, mazes with edge tip {len(tip)} (solved {sum(x["solved"] for x in tip)}), '
                  f'without {len(notip)} (solved {sum(x["solved"] for x in notip)}); failures with all wrong cells on the edge {edge_only}/{len(fails)}; '
                  f'wrong cells left on {on_left} (on edge {tip_diff}), wrongly off {off_wrong}; mean moves {sum(x["moves"] for x in m) / len(m):.0f}, '
                  f'max moves {max(x["moves"] for x in m)} of budget {v["budget"]}, {v["seconds"]:.0f} s')

print('\n== Looped mazes, Maze-Hard seed 0 (not their training distribution)')
for h in (15, 30, 60):
    d = load(f'run_released_loops_{h}.json')
    cnt = defaultdict(int)
    for x in d['mazes']:
        cnt[x['kind']] += 1
    oth = [x for x in d['mazes'] if x['kind'] == 'other']
    print(f'side {d["side"]}: ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())) +
          f'; other: not stopped {sum(not x["stopped"] for x in oth)}, not a path {sum(not x["valid"] for x in oth)}, '
          f'longer path {sum(1 for x in oth if x["valid"] and x["length"] > x["shortest"])}; settle cap hit {sum(x["settle_hit_cap"] for x in d["mazes"])}; '
          f'caps settle {d["settle_max"]} descend {d["descend_steps"]}; {d["seconds"]:.0f} s')
