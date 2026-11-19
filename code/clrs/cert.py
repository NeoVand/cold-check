"""Finite certificates for fitted rules (sorts, minimum, Kruskal, BFS): checks that prove the rule equals the task's
definition at every size. The rules are rebuilt from results/<task>.json (the final, frozen runs) and evaluated with the
same code that ran them (cold.State.feature). Every rule checked here is exact (a conjunction), so only its conditions
matter.

Why a finite check covers every size: each condition is a comparison of the pair (u, v) itself, or "is there a third
item k with tests on (u, k) and (k, v)" (one bound variable). Take any input of any size and a pair (u, v). Delete
every item except u, v and one witness for each existential condition that holds (the rule's and the definition's).
Every comparison of u and v is unchanged; every existential that held still holds (its witness is kept); every one that
failed still fails (deleting items cannot create a witness). So the pair has the same truth values in an input of at
most 2 + W items, where W is the number of existential conditions in the rule and the definition together. If the rule
and the definition agree on every pair of every input up to that size, they agree on every pair of every input.
Where the definition needs a global quantity (the bottleneck B, the BFS depth), it is taken as given together with
the axioms it satisfies, and the check enumerates the small configurations that satisfy those axioms.
"""
import ast, itertools, json, sys
from pathlib import Path
import numpy as np

import cold, gen

HERE = Path(__file__).parent
INF = np.inf


def load_rules(task):
    r = json.load(open(HERE / 'results' / f'{task}.json'))
    out = {}
    for name, d in r['rules'].items():
        if not isinstance(d, dict) or 'gate' not in d:
            continue
        assert d['exact_gate'], (task, name, 'only exact (conjunction) rules are certified here')
        assert d['gate_size'] == len(d['gate']), 'the stored gate is complete'
        gate = []
        for s in d['gate']:
            neg = s.startswith('not ')
            gate.append((ast.literal_eval(s[4:] if neg else s), not neg))
        kind = 'pair' if name in ('off', 'emask', 'srcptr') else 'item'
        out[name] = cold.Rule(kind, gate, np.zeros(0), 0.0, [], True, {})
    return out


def n_exist(rule):
    return sum(1 for f, _ in rule.gate if f[0] in ('wit', 'tri', 'cnt'))


# ---------------------------------------------------------------------------------------------------- sorts --
def cert_sort(task):
    """Keys distinct (the library draws them from U[0, 1]). Definition: u is v's pointer iff key u < key v and no key
    lies between; v points to itself iff no key is below it (the library's pred). Every input of up to 2 + W + 1 items, every
    order of keys, every pair."""
    R = load_rules(task); W = n_exist(R['off']) + 1; M = max(2 + W, 1 + n_exist(R['diag']) + 1) + 1
    pairs = 0
    for n in range(1, M + 1):
        for perm in itertools.permutations(range(n)):
            key = np.array(perm, float); st = cold.State(n, reg={'pos': np.arange(n) / n, 'key': key}, groups=[['pos'], ['key']])
            lab = gen.label(task, [key])['pred']
            off, diag = R['off'].decide(st), R['diag'].decide(st)
            for v in range(n):
                for u in range(n):
                    got = bool(diag[v]) if u == v else bool(off[u, v])
                    assert got == (lab[v] == u), (task, perm, u, v)
                    pairs += 1
    return dict(task=task, max_items=M, needed=2 + W, pairs_checked=pairs, conditions={k: len(r.gate) for k, r in R.items()})


def cert_minimum():
    R = load_rules('minimum')['one']; M = 1 + n_exist(R) + 2; items = 0
    for n in range(1, M + 1):
        for perm in itertools.permutations(range(n)):
            key = np.array(perm, float); st = cold.State(n, reg={'pos': np.arange(n) / n, 'key': key}, groups=[['pos'], ['key']])
            m = gen.label('minimum', [key])['min']; d = R.decide(st)
            assert all(bool(d[i]) == (i == m) for i in range(n)), perm
            items += n
    return dict(task='minimum', max_items=M, items_checked=items, conditions=len(R.gate))


# -------------------------------------------------------------------------------------------------- Kruskal --
def cert_kruskal():
    """The rule uses only conditions of the pair itself: edge(u, v), and w(u, v) against B(u, v), the bottleneck
    closure. Axioms: off an edge w is missing (every comparison false); on an edge B <= w (the edge is itself a path).
    Definition (the cycle property, distinct weights): an edge is in the minimum spanning forest iff no path joins its
    ends with every weight below w, i.e. iff B = w. The check enumerates every value of the atoms the axioms allow; a
    second check runs the rule on random small graphs against the library's own Kruskal."""
    R = load_rules('mst_kruskal')['emask']; cases = 0
    for edge in (False, True):
        for rel in (['missing'] if not edge else ['lt', 'eq']):            # B < w or B = w on an edge
            st = cold.State(2, rel={'edge': np.array([[False, edge], [edge, False]])},
                            pnum={'w': np.array([[np.nan, 1.0 if edge else np.nan], [1.0 if edge else np.nan, np.nan]]),
                                  'B': np.array([[-INF, 0.5 if rel == 'lt' else 1.0], [0.5 if rel == 'lt' else 1.0, -INF]])},
                            pgroups=[['w', 'B']], groups=[['pos']], reg={'pos': np.array([0., .5])})
            got = bool(R.decide(st)[0, 1]); want = edge and rel == 'eq'
            assert got == want, (edge, rel); cases += 1
    rng = np.random.RandomState(0); graphs = 0
    import complete
    for _ in range(2000):
        n = rng.randint(2, 8); A = gen.er_graph(rng, n, rng.uniform(0.2, 0.9), weighted=True)
        lab = gen.label('mst_kruskal', [A])['in_mst']; edge = A != 0; np.fill_diagonal(edge, False)
        w = np.where(edge, A, np.nan); st = cold.State(n, reg={'pos': np.arange(n) / n}, groups=[['pos']], rel={'edge': edge},
                                                      pnum={'w': w, 'B': complete.bottleneck(w)}, pgroups=[['w', 'B']])
        m = R.decide(st).copy(); np.fill_diagonal(m, False)
        assert (m == (lab > 0)).all(); graphs += 1
    return dict(task='mst_kruskal', atom_cases=cases, random_graphs_vs_library=graphs, conditions=len(R.gate))


# ------------------------------------------------------------------------------------------------------ BFS --
def cert_bfs():
    """Depth taken as given, with the axioms the true BFS depth satisfies pairwise: at most one node at depth 0 (the
    source); an edge joins depths that differ by at most 1, and never a reached node to an unreached one (inf).
    Definition, stated per pair: v points to u iff edge(u, v), u is one level above v (finite), and no neighbour of v
    one level above has a smaller index; v points to itself iff no neighbour of v is one level above. (In a real input,
    where every reached non-source node has a neighbour one level up, the second is "v is the source or unreached".)
    The conditions of the rules and of the definition mention u, v and one k each, so 4 items suffice (u, v, one witness
    for the rule, one for the definition). Every configuration of up to 4 items: depths in {0, ..., 6, inf} (any 4
    depths keep their order, equalities and unit gaps inside this range), every set of edges the axioms allow, items
    numbered in index order (enumerating all labellings covers every order), every pair."""
    R = load_rules('bfs'); M = 2 + n_exist(R['off']) + 1
    cases = 0
    for n in range(1, M + 1):
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        for dep in itertools.product([0, 1, 2, 3, 4, 5, 6, INF], repeat=n):
            dep = np.array(dep, float)
            if (dep == 0).sum() > 1:
                continue
            allowed = [(a, b) for a, b in pairs if (np.isinf(dep[a]) == np.isinf(dep[b])) and
                       (np.isinf(dep[a]) or abs(dep[a] - dep[b]) <= 1)]
            for mask in range(1 << len(allowed)):
                E = np.zeros((n, n), bool)
                for j, (a, b) in enumerate(allowed):
                    if mask >> j & 1:
                        E[a, b] = E[b, a] = True
                pos = np.arange(n) / n
                st = cold.State(n, reg={'pos': pos, 'depth': dep}, groups=[['pos'], ['depth']], rel={'edge': E},
                                bits={'source': dep == 0, 'has_depth': np.isfinite(dep)})
                st.answer |= {'has_depth'}; st.hidden |= {'depth'}
                off, diag = R['off'].decide(st), R['diag'].decide(st)
                for v in range(n):
                    up = [k for k in range(n) if E[k, v] and np.isfinite(dep[v]) and dep[k] == dep[v] - 1]
                    assert bool(diag[v]) == (not up), ('diag', dep, E, v)
                    for u in range(n):
                        if u != v:
                            want = u in up and all(pos[k] >= pos[u] for k in up)
                            assert bool(off[u, v]) == want, ('off', dep, E, u, v)
                            cases += 1
    return dict(task='bfs', max_items=M, pair_cases=cases, conditions={k: len(r.gate) for k, r in R.items()})


if __name__ == '__main__':
    out = [cert_sort(t) for t in gen.SORTS] + [cert_minimum(), cert_kruskal(), cert_bfs()]
    for o in out:
        print(o, flush=True)
    json.dump(out, open(HERE / 'cert.json', 'w'), indent=1)
