"""The cold check fitted at the (completed) answer, and run cold from nothing. One library for every task.

State of one input: item numbers (registers, grouped by units), per-input numbers, input pair relations, pair numbers,
item bits, and the answer (item bits or a pair relation) with its completion registers.

Tests on one ordered pair (x, y), generated from the state alone (nothing per task):
  cmp   r1[x] op r2[y] for every ordered pair of registers in one unit group, op in < <= = >= >
  rel   R(x, y) and R(y, x) for every pair relation (input, answer, completion)
  pcmp  W1(x, y) op W2(x, y) for pair numbers in one unit group
  id    x = y
Item bits: input bits, per-input comparisons r[x] op g, the answer's bits.
A leg is a test, alone or and-ed with an item bit of the third item k.

Features of a target:
  pair (a, b):  own tests at (a, b); item bits of a and of b; witnesses "is there k (not a, not b) with leg1(a, k) and
                leg2(k, b)".
  item i:       item bits of i; witnesses at (i, i) (two legs on the same third item); count comparisons
                #{k: leg(k, i)} < or = #{k: leg'(k, i)} (k runs over every item, i included).
Numbers enter only through comparisons. Nothing mentions a position or a size.

Fit (no loop): the gate is every literal (feature or its negation) that held on all training positives (the
most specific conjunction); a sparse logistic regression on the same features scores what the gate lets through and
breaks ties. Columns identical on the training set are merged first.
"""
import itertools, os, time
import numpy as np
from scipy.optimize import minimize

OPS = ('lt', 'le', 'eq', 'ge', 'gt')


def cmp(a, b, op):
    """A comparison. "Not yet" (inf, in a completion register) is later than every placed value, and two "not yet"
    values are not comparable: every comparison between them is false. A missing number (nan) compares false."""
    r = {'lt': a < b, 'le': a <= b, 'eq': a == b, 'ge': a >= b, 'gt': a > b}[op]
    return r & ~((a == np.inf) & (b == np.inf))


class State:
    """One input with its current answer. reg: {name: (n,) float}; glob: {name: float}; groups: lists of names (item
    registers and globals) in the same units; rel: {name: (n, n) bool}; pnum: {name: (n, n) float}; pgroups;
    bits: {name: (n,) bool}."""

    def __init__(self, n, reg=None, glob=None, groups=None, rel=None, pnum=None, pgroups=None, bits=None, answer=()):
        self.n, self.reg, self.glob = n, dict(reg or {}), dict(glob or {})
        self.answer = set(answer)          # names of the bits / relations that hold the answer (hidden at the target cell)
        self.hidden = set()                # completion registers computed from the answer: at the target, read only as
                                           # "the other item is below me" (r[other] < r[target])
        self.groups = [list(g) for g in (groups or [])]
        self.rel, self.pnum, self.pgroups, self.bits = dict(rel or {}), dict(pnum or {}), [list(g) for g in (pgroups or [])], dict(bits or {})
        self.xgroups = []                  # (pair number, item register) pairs in the same units: N(x, y) op r[x or y]
        self._c = {}

    # ---- vocabulary (the names of every test and bit this state offers) ----
    def tests(self):
        out = [('id',)]
        for g in self.groups:
            items = [r for r in g if r in self.reg]
            for r1, r2 in itertools.product(items, items):
                for op in OPS:
                    out.append(('cmp', r1, r2, op))
        for R in sorted(self.rel):
            out += [('rel', R, 0), ('rel', R, 1)]
        for g in self.pgroups:
            for W1, W2 in itertools.permutations([w for w in g if w in self.pnum], 2):
                for op in ('lt', 'eq', 'gt'):
                    out.append(('pcmp', W1, W2, op))
        for N, r in self.xgroups:
            if N in self.pnum and r in self.reg:
                out += [('pxr', N, r, side, op) for side in ('x', 'y') for op in ('lt', 'eq', 'gt')]
        return out

    def bitnames(self):
        out = sorted(('bit', b) for b in self.bits)
        for g in self.groups:
            for r in g:
                if r in self.reg:
                    for gl in g:
                        if gl in self.glob:
                            out += [('gcmp', r, gl, op) for op in OPS]
        return out

    # ---- evaluation (cached) ----
    def test(self, t):
        key = ('t',) + t
        if key not in self._c:
            n = self.n
            if t[0] == 'id':
                v = np.eye(n, dtype=bool)
            elif t[0] == 'cmp':
                v = cmp(self.reg[t[1]][:, None], self.reg[t[2]][None, :], t[3])
            elif t[0] == 'rel':
                v = self.rel[t[1]].T if t[2] else self.rel[t[1]]
            elif t[0] == 'and':
                v = self.test(t[1]) & self.test(t[2])
            elif t[0] == 'pxr':
                r = self.reg[t[2]]
                v = cmp(self.pnum[t[1]], r[None, :] if t[3] == 'y' else r[:, None], t[4])
            else:
                v = cmp(self.pnum[t[1]], self.pnum[t[2]], t[3])
            self._c[key] = np.asarray(v, bool)
        return self._c[key]

    def bit(self, b):
        key = ('b',) + b
        if key not in self._c:
            if b[0] == 'bit':
                v = self.bits[b[1]]
            else:
                v = cmp(self.reg[b[1]], self.glob[b[2]], b[3])
            self._c[key] = np.asarray(v, bool)
        return self._c[key]

    def leg(self, l, side):
        """side 1: leg (a, k), the bit on k is the column; side 2: leg (k, b), the bit on k is the row."""
        key = ('l', side) + l
        if key not in self._c:
            t, b = l
            v = self.test(t)
            if b is not None:
                v = v & (self.bit(b)[None, :] if side == 1 else self.bit(b)[:, None])
            self._c[key] = v
        return self._c[key]

    def count(self, l):
        key = ('c',) + l
        if key not in self._c:
            v = self.leg(l, 2)
            if l[1] is not None and l[1][0] == 'bit' and l[1][1] in self.answer:   # the target's own answer stays hidden
                v = v.copy(); np.fill_diagonal(v, False)
            self._c[key] = v.sum(0)
        return self._c[key]

    def feature(self, f, kind):
        """kind 'pair': (n, n) over (a, b); kind 'item': (n,)."""
        key = ('f', kind) + f
        if key in self._c:
            return self._c[key]
        n = self.n
        if f[0] == 'own':
            v = self.test(f[1])
        elif f[0] == 'bitA':
            v = np.broadcast_to(self.bit(f[1])[:, None], (n, n)) if kind == 'pair' else self.bit(f[1])
        elif f[0] == 'bitB':
            v = np.broadcast_to(self.bit(f[1])[None, :], (n, n))
        elif f[0] == 'wit':
            L1 = self.leg(f[1], 1).astype(np.float32); L2 = self.leg(f[2], 2).astype(np.float32)
            if kind == 'pair':
                L1 = L1.copy(); L2 = L2.copy(); np.fill_diagonal(L1, 0); np.fill_diagonal(L2, 0)   # k is neither a nor b
                v = (L1 @ L2) > 0.5
            else:
                M = L1 * L2.T; np.fill_diagonal(M, 0)                                           # leg1(i, k) and leg2(k, i), k != i
                v = M.sum(1) > 0.5
        elif f[0] == 'cnt':
            c1, c2 = self.count(f[1]), self.count(f[2])
            v = (c1 < c2) if f[3] == 'lt' else (c1 == c2)
        elif f[0] == 'tri':                                  # is there k (not a, not b): leg(k, b) and N1(k, b) op N2(a, b)
            v = tri_feature(self.leg(f[1], 2), self.pnum[f[2]], self.pnum[f[3]], f[4])
        else:
            raise ValueError(f)
        self._c[key] = v
        return v


def tri_feature(L2, N1, N2, op):
    """v[a, b] = is there k, k != a, k != b, with L2[k, b] and cmp(N1[k, b], N2[a, b], op). O(n^2 log n): the column
    minimum (lt) or maximum (gt) without row a, or a sorted search (eq). Same semantics as cmp (nan false; two inf
    values not comparable)."""
    n = len(L2); L = L2.copy(); np.fill_diagonal(L, False)
    V = np.where(L & ~np.isnan(N1), N1, np.nan)
    ar = np.arange(n)
    if op in ('lt', 'gt'):
        sgn = 1.0 if op == 'lt' else -1.0
        X = np.where(np.isnan(V), np.inf, sgn * V)                       # minimise sgn * N1
        k1 = np.argmin(X, 0); m1 = X[k1, ar]
        X2 = X.copy(); X2[k1, ar] = np.inf; m2 = X2.min(0)
        best = np.where(ar[:, None] == k1[None, :], m2[None, :], m1[None, :])   # best[a, b], row a excluded
        T = sgn * N2                                                       # want best < T  (sgn*N1 < sgn*N2)
        ok = best < T
        both_inf = (np.abs(best) == np.inf) & (np.abs(T) == np.inf) & (np.sign(best) == np.sign(T))
        found = np.isfinite(best) | (best == -np.inf)                      # some k exists (X is +inf when none)
        out = ok & ~both_inf & ~np.isnan(N2)
        # a k with N1 = +inf (lt: never below anything; gt: above every finite N2) is handled by X = sgn * inf
        return out & (found | (T == np.inf))
    out = np.zeros((n, n), bool)                                           # eq
    for b in range(n):
        col = V[:, b]; vals = np.sort(col[np.isfinite(col)])
        t = N2[:, b]; fin = np.isfinite(t)
        cnt = np.zeros(n, np.int64)
        cnt[fin] = np.searchsorted(vals, t[fin], 'right') - np.searchsorted(vals, t[fin], 'left')
        selfm = fin & np.isfinite(col) & (col == t)                         # k = a matches itself: excluded
        out[:, b] = (cnt - selfm) > 0
    return out


FLIP = {'lt': 'gt', 'gt': 'lt', 'le': 'ge', 'ge': 'le', 'eq': 'eq'}


def conj_tests(st):
    """Legs that and a relation (input relation, or a comparison of pair numbers) with a comparison of registers."""
    tests = st.tests()
    R = [t for t in tests if t[0] in ('rel', 'pcmp')]; C = [t for t in tests if t[0] == 'cmp']
    return [('and', r, c) for r in R for c in C]


def visible(st, t, where):
    """Is test t visible at this place? where: 'own' (a, b) or 'leg2' (k, b), the target item being b; 'leg1i' (i, k),
    the target item being i. The target's own completion registers are read only as "below me"."""
    if t[0] == 'and':
        return visible(st, t[2], where)
    if t[0] != 'cmp' or not st.hidden:
        return True
    _, r1, r2, op = t
    if where in ('own', 'leg2') and r2 in st.hidden:
        return r1 == r2 and op == 'lt'
    if where == 'leg1i' and r1 in st.hidden:
        return r1 == r2 and op == 'gt'
    return True


def bit_visible(st, b):
    """Bits of the target item itself: the answer's bits are hidden, and so are comparisons of hidden registers."""
    return not ((b[0] == 'bit' and b[1] in st.answer) or (b[0] == 'gcmp' and b[1] in st.hidden))


def legs_of(st, side, leg_bits=True, level=None, conj=False, kind='pair'):
    """Legs on side 1 (a, k) or side 2 (k, b). The answer's bits on k are read only through the level, when one is
    given: level (r, op) lets leg (k, i) read k's answer only together with r[k] op r[i] (so the answer is read only
    downward along one order; this is what makes the run from nothing settle, Fages' tightness)."""
    tests, bits = st.tests() + (conj_tests(st) if conj else []), st.bitnames()
    where = 'leg2' if side == 2 else ('leg1i' if kind == 'item' else None)
    if where:
        tests = [t for t in tests if visible(st, t, where)]
    legs = [(t, None) for t in tests]
    if not leg_bits:
        return legs
    for b in bits:
        if b[0] == 'bit' and b[1] in st.answer and level is not None:
            r, op = level
            legs.append((('cmp', r, r, op if side == 2 else FLIP[op]), b))
        else:
            legs += [(t, b) for t in tests]
    return legs


def tri_specs(st, l2s):
    plain = [l for l in l2s if l[1] is None]
    return [(l, N1, N2, op) for l in plain for g in st.pgroups for N1 in g for N2 in g for op in ('lt', 'eq', 'gt')]


def vocabulary(st, kind, leg_bits=True, counts=True, level=None, conj=False, tri=False):
    """Every feature a state of this shape offers for a target of this kind (witnesses, counts, pair-number
    comparisons through k last, in leg order)."""
    tests, bits = st.tests(), st.bitnames()
    l1s, l2s = legs_of(st, 1, leg_bits, level, conj, kind), legs_of(st, 2, leg_bits, level, conj, kind)
    own_tests = [t for t in tests if not (t[0] == 'rel' and t[1] in st.answer) and visible(st, t, 'own')]
    if kind == 'pair':
        F = [('own', t) for t in own_tests] + [('bitA', b) for b in bits] + [('bitB', b) for b in bits if bit_visible(st, b)]
    else:
        F = [('bitA', b) for b in bits if bit_visible(st, b)]
    F += [('wit', l1, l2) for l1 in l1s for l2 in l2s]
    if kind == 'item' and counts:
        F += [('cnt', l1, l2, op) for l1 in l2s for l2 in l2s if l1 != l2 for op in ('lt', 'eq')]
    if kind == 'pair' and tri:
        F += [('tri',) + x for x in tri_specs(st, l2s)]
    return F


def block(st, kind, legs, counts, tri=False):
    """The witness (and count) features of the full vocabulary at once, in vocabulary order: (n, n, F) or (n, F).
    legs: (side-1 legs, side-2 legs)."""
    n = st.n; l1s, l2s = legs
    L1 = np.stack([st.leg(l, 1) for l in l1s]).astype(np.float32)
    L2 = np.stack([st.leg(l, 2) for l in l2s]).astype(np.float32)
    ar = np.arange(n)
    if kind == 'pair':
        L1[:, ar, ar] = 0; L2[:, ar, ar] = 0
        W = (np.einsum('pak,qkb->abpq', L1, L2, optimize=True) > 0.5).reshape(n, n, -1)
        if not tri:
            return W
        out = []
        plain = [i for i, l in enumerate(l2s) if l[1] is None]
        P2 = L2[plain]                                                   # (Q, k, b), diagonal already zero
        per = {}
        for g in st.pgroups:
            for N1 in g:
                for N2 in g:
                    for op in ('lt', 'eq', 'gt'):
                        T = cmp(st.pnum[N1][None, :, :], st.pnum[N2][:, None, :], op).astype(np.float32)   # T[a, k, b]
                        T[ar, ar, :] = 0                                                                  # k != a
                        per[(N1, N2, op)] = np.einsum('qkb,akb->abq', P2, T, optimize=True) > 0.5
        for qi, l in enumerate([l2s[i] for i in plain]):
            for g in st.pgroups:
                for N1 in g:
                    for N2 in g:
                        for op in ('lt', 'eq', 'gt'):
                            out.append(per[(N1, N2, op)][:, :, qi])
        return np.concatenate([W, np.stack(out, -1)], -1)
    M = np.einsum('pik,qki->ipqk', L1, L2, optimize=True); M[ar, :, :, ar] = 0
    W = (M.sum(-1) > 0.5).reshape(n, -1)
    if not counts:
        return W
    C = np.stack([st.count(l) for l in l2s], -1)                       # (n, L)
    P = len(l2s); iu = [(p, q) for p in range(P) for q in range(P) if p != q]
    p_, q_ = np.array([x[0] for x in iu]), np.array([x[1] for x in iu])
    lt, eq = C[:, p_] < C[:, q_], C[:, p_] == C[:, q_]
    return np.concatenate([W, np.stack([lt, eq], -1).reshape(n, -1)], -1)


def matrix(states, feats, kind, rows, legs=None, counts=True, tri=False):
    """Feature rows: rows(st) gives a boolean mask (n, n) or (n,) of the target cells used from this state. With legs
    given, feats must be vocabulary(...) and the witness/count part is computed at once."""
    X = []
    for st in states:
        m = rows(st)
        if legs is None:
            X.append(np.stack([np.asarray(st.feature(f, kind))[m] for f in feats], -1))
            continue
        head = [f for f in feats if f[0] not in ('wit', 'cnt', 'tri')]
        H = np.stack([np.broadcast_to(np.asarray(st.feature(f, kind)), m.shape) for f in head], -1) if head else np.zeros(m.shape + (0,), bool)
        X.append(np.concatenate([H[m], block(st, kind, legs, counts, tri)[m]], -1))
    return np.concatenate(X).astype(bool)


def logreg_l1(X, y, lam=1e-3, l2=1e-4):
    """Sparse logistic regression (L1 by the split-sign trick, L-BFGS-B). Returns weights, bias."""
    X = X.astype(np.float64); N, D = X.shape
    def fg(z):
        wp, wn, b = z[:D], z[D:2 * D], z[-1]; w = wp - wn
        s = X @ w + b; p = 1 / (1 + np.exp(-np.clip(s, -40, 40)))
        loss = np.mean(np.logaddexp(0, s) - y * s) + lam * (wp.sum() + wn.sum()) + l2 * (w @ w)
        g = X.T @ (p - y) / N; gw = g + 2 * l2 * w
        return loss, np.r_[gw + lam, -gw + lam, np.mean(p - y)]
    z0 = np.zeros(2 * D + 1)
    r = minimize(fg, z0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * (2 * D) + [(None, None)], options=dict(maxiter=500))
    return r.x[:D] - r.x[D:2 * D], r.x[-1]


class Rule:
    """A fitted check for one target: gate literals and a sparse score over named features."""

    def __init__(self, kind, gate, w, b, feats, exact_gate, info):
        self.kind, self.gate, self.w, self.b, self.feats, self.exact_gate, self.info = kind, gate, w, b, feats, exact_gate, info

    def used(self):
        return sorted({f for f, _ in self.gate} | set(self.feats), key=str)

    def passes(self, st):
        n = st.n; ok = np.ones((n, n) if self.kind == 'pair' else n, bool)
        for f, v in self.gate:
            ok &= (np.asarray(st.feature(f, self.kind)) == v)
        return ok

    def score(self, st):
        n = st.n; s = np.full((n, n) if self.kind == 'pair' else n, self.b)
        for f, w in zip(self.feats, self.w):
            s = s + w * np.asarray(st.feature(f, self.kind), np.float64)
        return s

    def decide(self, st):
        """Yes/no per cell: the gate, and the score where the gate alone was not exact in training."""
        ok = self.passes(st)
        return ok if self.exact_gate else ok & (self.score(st) > 0)


def fit(states, targets, kind, rows, feats=None, lam=1e-3, **vk):
    """targets(st): boolean array like rows(st). Returns a Rule and the fit time (s) of this call."""
    t0 = time.perf_counter()
    if feats is None:
        feats = vocabulary(states[0], kind, **vk)
        lb, lv, cj = vk.get('leg_bits', True), vk.get('level'), vk.get('conj', False)
        X = matrix(states, feats, kind, rows, legs=(legs_of(states[0], 1, lb, lv, cj, kind), legs_of(states[0], 2, lb, lv, cj, kind)),
                   counts=vk.get('counts', True), tri=vk.get('tri', False))
    else:
        X = matrix(states, feats, kind, rows)
    y = np.concatenate([targets(st)[rows(st)] for st in states]).astype(bool)
    # merge identical columns, drop constant ones
    keep, seen = [], {}
    for j in range(X.shape[1]):
        c = X[:, j]
        if c.all() or not c.any():
            continue
        h = c.tobytes()
        if h not in seen:
            seen[h] = j; keep.append(j)
    Xk, fk = X[:, keep], [feats[j] for j in keep]
    P = Xk[y]
    # keep every condition that held on all training positives and was seen to fail on enough other rows (a condition
    # never seen to fail carries no evidence): at least max(10, 0.5% of the rows)
    need = max(10, int(0.005 * len(y)))
    scope = os.environ.get('GATE_SCOPE', 'own')          # 'own' (default): the target's own tests and bits, as in the BFS walk-through; 'all': witnesses too
    gate = [(fk[j], bool(P[0, j])) for j in range(len(fk))
            if (P[:, j].all() or not P[:, j].any()) and int((Xk[:, j] != P[0, j]).sum()) >= need
            and (scope == 'all' or fk[j][0] in ('own', 'bitA', 'bitB'))] if len(P) else []
    passed = np.ones(len(y), bool)
    for f, v in gate:
        passed &= (Xk[:, fk.index(f)] == v)
    # then the greedy conjunction (set cover; Haussler 1988): add the condition, true on every training positive, that
    # excludes the most negatives still passing, until none passes or no condition helps
    greedy = []
    if os.environ.get('GREEDY', '1') == '1' and len(P):
        const = [j for j in range(len(fk)) if P[:, j].all() or not P[:, j].any()]
        Cv = np.array([bool(P[0, j]) for j in const])
        Xc = Xk[:, const]
        while (passed & ~y).any():
            bad = passed & ~y
            gain = (Xc[bad] != Cv[None, :]).sum(0)
            j = int(gain.argmax())
            if gain[j] == 0:
                break
            greedy.append((fk[const[j]], bool(Cv[j]))); passed &= (Xc[:, j] == Cv[j])
    gate = gate + greedy
    exact_gate = bool(not (passed & ~y).any())
    w, b = logreg_l1(Xk, y.astype(np.float64), lam=lam)
    nz = np.abs(w) > 1e-4
    rule = Rule(kind, gate, w[nz], b, [f for f, z in zip(fk, nz) if z], exact_gate,
                dict(features=len(feats), distinct=len(fk), rows=int(len(y)), positives=int(y.sum()), gate=len(gate), greedy=len(greedy),
                     gate_lets_through_negatives=int((passed & ~y).sum()), score_features=int(nz.sum()),
                     train_wrong_score=int(((Xk[:, nz] @ w[nz] + b > 0) != y).sum()),
                     train_wrong_rule=int(((passed & ((Xk[:, nz] @ w[nz] + b > 0) | exact_gate)) != y).sum())))
    return rule, time.perf_counter() - t0


# ---------------- reading a pointer and a one-of output from rules ----------------
def read_pointer(st, off, diag, none=False):
    """pi[b] for every b from an off-diagonal pair rule and a diagonal (self-pointer) item rule: the passing candidate
    with the highest score; if none passes, the highest score (none=False) or no pointer, -1 (none=True: the cold run
    writes a pointer only where the check passes)."""
    n = st.n
    so, sd = off.score(st), diag.score(st)
    S = so.copy(); np.fill_diagonal(S, sd)
    ok = off.decide(st); np.fill_diagonal(ok, diag.decide(st))
    Sg = np.where(ok, S, -np.inf)
    pi = np.where(ok.any(0), Sg.argmax(0), -1 if none else S.argmax(0))
    return pi, ok.sum(0)


def read_one(st, rule):
    ok = rule.decide(st); s = rule.score(st)
    return int(np.where(ok, s, -np.inf).argmax() if ok.any() else s.argmax()), int(ok.sum())
