"""Cold derivation and warm search on unit count conditions, one engine for every board size (numba).

State: each item (cell) keeps a bitmask of the values still possible. Units are the classes of the kept relations
(check.units). For a unit and a value v: L = items fixed to v, U = items where v is still possible.
Cold derivation (bounds on the fitted count conditions, repeated until nothing changes):
  condition n_v <= 1:  L >= 2 -> contradiction;  L == 1 -> v removed from every other item of the unit.
  condition n_v >= 1:  U == 0 -> contradiction;  U == 1 -> the one item is fixed to v.
  an item with no value left -> contradiction.
Only units touched by a change are rescanned (a queue).
Warm search on what stays open: the open item with the fewest values (ties: lowest index), its values in a random
order (seeded per puzzle), the cold derivation after each choice, undo on contradiction (depth-first). The stop is the
fitted check: energy <= tau on the full board, computed with the fitted pair costs.

Operation counts, CRAM's convention (results/op-counts.json, paper App. C.2): an addition or subtraction counts 1;
comparisons, selections, mask logic, indexing and random numbers count 0. `ops` is that count; `cmp` counts every
comparison and mask test as well, for a stricter total (ops + cmp).
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True, nogil=True)
def lowbit_index(x):
    i = 0
    while not (x >> i) & 1:
        i += 1
    return i


@njit(cache=True, nogil=True)
def propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats):
    """dom (n,) int64 masks, modified in place. queue holds unit ids to scan (inq flags). Returns False on a
    contradiction. stats[0] ops, stats[1] cmp."""
    head = 0
    tail = 0
    nu = unit_size.shape[0]
    # queue is a ring of size nu
    cnt = 0
    for u in range(nu):
        if inq[u]:
            cnt += 1
    # rebuild ring from flags
    tail = 0
    for u in range(nu):
        if inq[u]:
            queue[tail] = u
            tail += 1
    size = tail
    tail = tail % nu
    while size > 0:
        u = queue[head]
        head = (head + 1) % nu
        size -= 1
        inq[u] = False
        s = unit_size[u]
        g = ge1[u]
        l = le1[u]
        for v in range(V):
            bit = np.int64(1) << v
            L = 0
            U = 0
            last = -1
            for k in range(s):
                it = unit_items[u, k]
                d = dom[it]
                stats[1] += 1
                if d & bit:
                    U += 1
                    stats[0] += 1
                    last = it
                    stats[1] += 1
                    if d == bit:
                        L += 1
                        stats[0] += 1
            stats[1] += 4
            if l and L >= 2:
                return False
            if g and U == 0:
                return False
            changed_items = 0
            if l and L == 1 and U > 1:
                for k in range(s):
                    it = unit_items[u, k]
                    d = dom[it]
                    stats[1] += 1
                    if (d & bit) and d != bit:
                        nd = d & ~bit
                        if nd == 0:
                            return False
                        dom[it] = nd
                        for j in range(item_units.shape[1]):
                            w = item_units[it, j]
                            if w >= 0 and not inq[w]:
                                inq[w] = True
                                queue[tail] = w
                                tail = (tail + 1) % nu
                                size += 1
            elif g and U == 1 and dom[last] != bit:
                dom[last] = bit
                for j in range(item_units.shape[1]):
                    w = item_units[last, j]
                    if w >= 0 and not inq[w]:
                        inq[w] = True
                        queue[tail] = w
                        tail = (tail + 1) % nu
                        size += 1
    return True


@njit(cache=True, nogil=True)
def energy(dom, W_i, W_j, W_w, stats):
    """Energy of a full board (every mask a single bit): sum over listed pairs of w [x_i == x_j]."""
    e = 0.0
    for p in range(W_i.shape[0]):
        stats[1] += 1
        if dom[W_i[p]] == dom[W_j[p]]:
            e += W_w[p]
            stats[0] += 1
    return e


@njit(cache=True, nogil=True)
def solve(dom0, unit_items, unit_size, item_units, ge1, le1, V, W_i, W_j, W_w, tau, seed, budget, out):
    """Cold derivation, then warm search. out (int64[8]): 0 status (1 solved, 0 budget, -1 no solution),
    1 moves (choices tried), 2 ops, 3 cmp, 4 solved by cold alone (1/0), 5 max depth, 6 stop tests, 7 rejected stops.
    Returns the final masks."""
    n = dom0.shape[0]
    nu = unit_size.shape[0]
    stats = np.zeros(2, np.int64)
    queue = np.zeros(nu, np.int64)
    inq = np.ones(nu, np.bool_)
    dom = dom0.copy()
    for k in range(8):
        out[k] = 0
    ok = propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats)
    if not ok:
        out[0] = -1
        out[2] = stats[0]
        out[3] = stats[1]
        return dom
    done = True
    for i in range(n):
        if popcount(dom[i]) > 1:
            done = False
            break
    if done:
        out[6] += 1
        e = energy(dom, W_i, W_j, W_w, stats)
        if e <= tau:
            out[0] = 1
            out[4] = 1
            out[2] = stats[0]
            out[3] = stats[1]
            return dom
        out[7] += 1
    # depth-first warm search
    rng_state = np.uint64(seed * 2654435761 + 1)
    maxd = n + 1
    stack_dom = np.zeros((maxd, n), np.int64)
    stack_item = np.zeros(maxd, np.int64)
    stack_left = np.zeros(maxd, np.int64)  # values not yet tried at this level
    depth = 0
    moves = 0
    # choose first item
    best = -1
    bc = 1 << 30
    for i in range(n):
        c = popcount(dom[i])
        stats[1] += 1
        if c > 1 and c < bc:
            bc = c
            best = i
    stack_dom[0] = dom
    stack_item[0] = best
    stack_left[0] = dom[best]
    while depth >= 0:
        if moves >= budget:
            out[0] = 0
            break
        left = stack_left[depth]
        if left == 0:
            depth -= 1
            continue
        # random value among the remaining
        c = popcount(left)
        rng_state ^= rng_state << np.uint64(13)
        rng_state ^= rng_state >> np.uint64(7)
        rng_state ^= rng_state << np.uint64(17)
        pick = int(rng_state % np.uint64(c))
        x = left
        for _ in range(pick):
            x &= x - 1
        bit = x & (-x)
        stack_left[depth] = left & ~bit
        moves += 1
        dom = stack_dom[depth].copy()
        it = stack_item[depth]
        dom[it] = bit
        for j in range(item_units.shape[1]):
            w = item_units[it, j]
            if w >= 0:
                inq[w] = True
        ok = propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats)
        for u in range(nu):
            inq[u] = False
        if not ok:
            continue
        best = -1
        bc = 1 << 30
        for i in range(n):
            cc = popcount(dom[i])
            stats[1] += 1
            if cc > 1 and cc < bc:
                bc = cc
                best = i
        if best < 0:
            out[6] += 1
            e = energy(dom, W_i, W_j, W_w, stats)
            if e <= tau:
                out[0] = 1
                break
            out[7] += 1
            continue
        depth += 1
        if depth + 1 > out[5]:
            out[5] = depth + 1
        stack_dom[depth] = dom
        stack_item[depth] = best
        stack_left[depth] = dom[best]
    if depth < 0:
        out[0] = -1
    out[1] = moves
    out[2] = stats[0]
    out[3] = stats[1]
    return dom


def build(check, b):
    """Arrays for the engine on a board of block size b from a fitted check (any b)."""
    import check as ck
    V = b * b
    M = V * V
    U = ck.units(b, check['relations'])
    smax = max(len(c) for _, c in U)
    unit_items = -np.ones((len(U), smax), np.int64)
    unit_size = np.zeros(len(U), np.int64)
    ge1 = np.zeros(len(U), np.bool_)
    le1 = np.zeros(len(U), np.bool_)
    item_units_l = [[] for _ in range(M)]
    for u, (k, cells) in enumerate(U):
        unit_items[u, :len(cells)] = cells
        unit_size[u] = len(cells)
        ge1[u], le1[u] = check['conditions'][k]
        for c in cells:
            item_units_l[c].append(u)
    m = max(len(x) for x in item_units_l)
    item_units = -np.ones((M, m), np.int64)
    for c, us in enumerate(item_units_l):
        item_units[c, :len(us)] = us
    W = ck.weights_for(check, b)
    iu, ju = np.triu_indices(M, 1)
    keep = W[iu, ju] > 0
    return dict(V=V, unit_items=unit_items, unit_size=unit_size, item_units=item_units, ge1=ge1, le1=le1,
                W_i=iu[keep].astype(np.int64), W_j=ju[keep].astype(np.int64), W_w=W[iu, ju][keep].astype(np.float64),
                tau=float(check['tau']))


def masks(puzzle, V):
    """puzzle (M,) values 0 (blank) or 1..V -> masks."""
    full = (1 << V) - 1
    return np.where(puzzle > 0, np.left_shift(1, np.maximum(puzzle, 1) - 1), full).astype(np.int64)


def decode(dom):
    return np.array([lowbit_index(int(d)) + 1 if d and (d & (d - 1)) == 0 else 0 for d in dom])


def run(E, puzzle, seed=0, budget=10 ** 6):
    out = np.zeros(8, np.int64)
    dom = solve(masks(puzzle, E['V']), E['unit_items'], E['unit_size'], E['item_units'], E['ge1'], E['le1'], E['V'],
                E['W_i'], E['W_j'], E['W_w'], E['tau'], seed, budget, out)
    return dom, out


# ------------------------------------------------------------------------------------------------ second version --
# Registered after the 25 x 25 miss (the paper, Sudoku): two changes, each with a reason.
# (1) Unit pairs. Two kept conditions on overlapping units combine: if every item of unit A where v is still
#     possible lies in unit B, and A needs v once (n_v >= 1) while B allows it once (n_v <= 1), then v is removed from
#     B outside A. This is a consequence of the fitted conditions, generic over any pair of units (row-box, col-box at
#     every size); Sudoku players call it pointing / box-line reduction.
# (2) Restarts. Depth-first run times were heavy-tailed (median 85k moves, 9 of 100 over 10^6), the textbook case
#     for randomised restarts (Gomes, Selman and Kautz 1998); Luby's schedule times base moves.

@njit(cache=True, nogil=True)
def unit_pairs_pass(dom, unit_items, unit_size, pa, pb, ge1, le1, inb, V, stats, bout, bout_n):
    """One pass of the unit-pair rule over directed pairs (pa[k], pb[k]). inb[k, item]: item is in unit pb[k];
    bout[k, :bout_n[k]]: the items of pb[k] outside pa[k]. Returns -1 on a contradiction, else the number of removals."""
    removed = 0
    for k in range(pa.shape[0]):
        A = pa[k]
        if not (ge1[A] and le1[pb[k]]):
            continue
        for v in range(V):
            bit = np.int64(1) << v
            anyin = False
            inside = True
            for j in range(unit_size[A]):
                it = unit_items[A, j]
                stats[1] += 1
                if dom[it] & bit:
                    anyin = True
                    if not inb[k, it]:
                        inside = False
                        break
            if not anyin or not inside:
                continue
            for j in range(bout_n[k]):
                it = bout[k, j]
                stats[1] += 1
                if dom[it] & bit:
                    dom[it] &= ~bit
                    removed += 1
                    if dom[it] == 0:
                        return -1
    return removed


@njit(cache=True, nogil=True)
def propagate2(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats, pa, pb, inb, use_pairs, bout, bout_n):
    while True:
        ok = propagate(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats)
        if not ok:
            return False
        if not use_pairs:
            return True
        snap = dom.copy()
        r = unit_pairs_pass(dom, unit_items, unit_size, pa, pb, ge1, le1, inb, V, stats, bout, bout_n)
        if r < 0:
            return False
        if r == 0:
            return True
        for i in range(dom.shape[0]):
            if dom[i] != snap[i]:
                for j in range(item_units.shape[1]):
                    w = item_units[i, j]
                    if w >= 0:
                        inq[w] = True


@njit(cache=True, nogil=True)
def luby(i):
    # i >= 1
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while True:
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        i = i - (1 << (k - 1)) + 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1


@njit(cache=True, nogil=True)
def solve2(dom0, unit_items, unit_size, item_units, ge1, le1, V, W_i, W_j, W_w, tau, seed, budget, out,
           pa, pb, inb, use_pairs, restart_base, bout, bout_n):
    """As solve, with the unit-pair rule (use_pairs) and Luby restarts (restart_base > 0). out[6] holds restarts."""
    n = dom0.shape[0]
    nu = unit_size.shape[0]
    stats = np.zeros(2, np.int64)
    queue = np.zeros(nu, np.int64)
    inq = np.ones(nu, np.bool_)
    root = dom0.copy()
    for k in range(8):
        out[k] = 0
    ok = propagate2(root, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats, pa, pb, inb, use_pairs, bout, bout_n)
    if not ok:
        out[0] = -1
        out[2] = stats[0]; out[3] = stats[1]
        return root
    done = True
    for i in range(n):
        if popcount(root[i]) > 1:
            done = False
            break
    if done:
        e = energy(root, W_i, W_j, W_w, stats)
        if e <= tau:
            out[0] = 1; out[4] = 1; out[2] = stats[0]; out[3] = stats[1]
            return root
    rng_state = np.uint64(seed * 2654435761 + 1)
    maxd = n + 1
    stack_dom = np.zeros((maxd, n), np.int64)
    stack_item = np.zeros(maxd, np.int64)
    stack_left = np.zeros(maxd, np.int64)
    moves = 0
    run = 0
    result = root
    exhausted = False
    while moves < budget and not exhausted:
        run += 1
        limit = budget if restart_base <= 0 else restart_base * luby(run)
        run_moves = 0
        best = -1
        bc = 1 << 30
        for i in range(n):
            c = popcount(root[i])
            if c > 1 and c < bc:
                bc = c; best = i
        depth = 0
        stack_dom[0] = root
        stack_item[0] = best
        stack_left[0] = root[best]
        solved = False
        while depth >= 0:
            if moves >= budget or run_moves >= limit:
                break
            left = stack_left[depth]
            if left == 0:
                depth -= 1
                continue
            c = popcount(left)
            rng_state ^= rng_state << np.uint64(13)
            rng_state ^= rng_state >> np.uint64(7)
            rng_state ^= rng_state << np.uint64(17)
            pick = int(rng_state % np.uint64(c))
            x = left
            for _ in range(pick):
                x &= x - 1
            bit = x & (-x)
            stack_left[depth] = left & ~bit
            moves += 1
            run_moves += 1
            dom = stack_dom[depth].copy()
            it = stack_item[depth]
            dom[it] = bit
            for j in range(item_units.shape[1]):
                w = item_units[it, j]
                if w >= 0:
                    inq[w] = True
            ok = propagate2(dom, unit_items, unit_size, item_units, ge1, le1, V, queue, inq, stats, pa, pb, inb, use_pairs, bout, bout_n)
            for u in range(nu):
                inq[u] = False
            if not ok:
                continue
            best = -1
            bc = 1 << 30
            for i in range(n):
                cc = popcount(dom[i])
                if cc > 1 and cc < bc:
                    bc = cc; best = i
            if best < 0:
                e = energy(dom, W_i, W_j, W_w, stats)
                if e <= tau:
                    solved = True
                    result = dom
                    break
                continue
            depth += 1
            stack_dom[depth] = dom
            stack_item[depth] = best
            stack_left[depth] = dom[best]
        if solved:
            out[0] = 1
            break
        if depth < 0:  # the whole tree under the root was searched: no solution
            exhausted = True
            out[0] = -1
    out[1] = moves
    out[2] = stats[0]
    out[3] = stats[1]
    out[6] = run - 1
    return result


def pairs_for(E):
    """Directed pairs of units that share at least two items (row-box, col-box at every size)."""
    ui, us = E['unit_items'], E['unit_size']
    sets = [set(ui[u, :us[u]].tolist()) for u in range(len(us))]
    n = E['item_units'].shape[0]
    pa, pb = [], []
    for a in range(len(sets)):
        for b in range(len(sets)):
            if a != b and len(sets[a] & sets[b]) >= 2:
                pa.append(a); pb.append(b)
    inb = np.zeros((len(pa), n), np.bool_)
    m = max(len(sets[b] - sets[a]) for a, b in zip(pa, pb))
    bout = -np.ones((len(pa), m), np.int64)
    bout_n = np.zeros(len(pa), np.int64)
    for k, (a, b) in enumerate(zip(pa, pb)):
        inb[k, list(sets[b])] = True
        rest = sorted(sets[b] - sets[a])
        bout[k, :len(rest)] = rest
        bout_n[k] = len(rest)
    return np.array(pa, np.int64), np.array(pb, np.int64), inb, bout, bout_n


def run2(E, puzzle_masks, seed=0, budget=10 ** 6, use_pairs=True, restart_base=0):
    if 'pairs' not in E:
        E['pairs'] = pairs_for(E)
    pa, pb, inb, bout, bout_n = E['pairs']
    out = np.zeros(8, np.int64)
    dom = solve2(puzzle_masks, E['unit_items'], E['unit_size'], E['item_units'], E['ge1'], E['le1'], E['V'],
                 E['W_i'], E['W_j'], E['W_w'], E['tau'], seed, budget, out, pa, pb, inb, use_pairs, restart_base,
                 bout, bout_n)
    return dom, out
