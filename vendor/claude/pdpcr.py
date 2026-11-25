"""PDP-CR on grids: instances, flow-graph evaluation, construction and moves.

Shared by the grid-aware / generic genetic algorithm and tabu search.
Nodes: 0 = source depot, 1..n = tasks, n+1 = sink depot.
A solution is a pair (out, inn) of lists of dicts: out[i][j] = inn[j][i] = f_ij > 0.
"""
import math
import random
from collections import deque

# ----------------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------------

class Instance:
    def __init__(self, name, layout, W, H, depot, P, D, r, m, dist):
        self.name, self.layout, self.W, self.H = name, layout, W, H
        self.depot, self.P, self.D = depot, P, D
        self.n = len(P)
        self.m = m
        n = self.n
        self.r = [m] + list(r) + [m]
        self.loc_p = [depot] + list(P) + [depot]   # pickup cell of each node
        self.loc_d = [depot] + list(D) + [depot]   # delivery cell of each node
        self.c = [0] + [dist(P[i], D[i]) for i in range(n)] + [0]
        N2 = n + 2
        self.t = [[0] * N2 for _ in range(N2)]
        for i in range(N2):
            for j in range(N2):
                if i != j:
                    self.t[i][j] = dist(self.loc_d[i], self.loc_p[j])
        self.t[0][n + 1] = 0

    def lower_bound(self):
        n, t, c, r = self.n, self.t, self.c, self.r
        lb1 = max(t[0][j] + c[j] + t[j][n + 1] for j in range(1, n + 1))
        tot = 0
        for j in range(1, n + 1):
            mn = min(t[k][j] for k in range(0, n + 1) if k != j)
            tot += r[j] * (c[j] + mn)
        return max(lb1, tot / self.m)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_open_grid(n, m, W, H, seed, rmax=5):
    rng = random.Random(seed)
    depot = (0, H // 2)
    P, D, R = [], [], []
    for _ in range(n):
        p = (rng.randrange(W), rng.randrange(H))
        while True:
            d = (rng.randrange(W), rng.randrange(H))
            if d != p:
                break
        P.append(p); D.append(d); R.append(rng.randint(1, rmax))
    return Instance(f"open_n{n}_s{seed}", "open", W, H, depot, P, D, R, m, manhattan)


def make_aisle_grid(n, m, A, Y, seed, rmax=5, spacing=3):
    """Parallel aisles at x = 0, spacing, 2*spacing, ...; cross-aisles at y=0 and y=Y."""
    rng = random.Random(seed)
    depot = (0, 0)

    def dist(a, b):
        if a[0] == b[0]:
            return abs(a[1] - b[1])
        return abs(a[0] - b[0]) + min(a[1] + b[1], 2 * Y - a[1] - b[1])

    def pt():
        return (spacing * rng.randrange(A), rng.randint(1, Y - 1))

    P, D, R = [], [], []
    for _ in range(n):
        p = pt()
        while True:
            d = pt()
            if d != p:
                break
        P.append(p); D.append(d); R.append(rng.randint(1, rmax))
    W = spacing * (A - 1) + 1
    return Instance(f"aisle_n{n}_s{seed}", "aisle", W, Y + 1, depot, P, D, R, m, dist)


# ----------------------------------------------------------------------------
# Solutions
# ----------------------------------------------------------------------------

def empty_sol(n):
    return [dict() for _ in range(n + 2)], [dict() for _ in range(n + 2)]


def copy_sol(sol):
    out, inn = sol
    return [d.copy() for d in out], [d.copy() for d in inn]


def add_flow(sol, i, j, q):
    out, inn = sol
    v = out[i].get(j, 0) + q
    if v < 0:
        raise ValueError("negative flow")
    if v == 0:
        out[i].pop(j, None); inn[j].pop(i, None)
    else:
        out[i][j] = v; inn[j][i] = v


def flow(sol, i, j):
    return sol[0][i].get(j, 0)


class Eval:
    __slots__ = ("order", "pos", "z", "tau", "cmax", "crit", "crit_arcs")


def evaluate(inst, sol, eps=1e-9):
    """Topological order, start times, tails, makespan and critical arcs.
    Returns None if the support has a cycle."""
    out, inn = sol
    N2 = inst.n + 2
    indeg = [len(inn[v]) for v in range(N2)]
    q = deque(v for v in range(N2) if indeg[v] == 0)
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for w in out[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(order) < N2:
        return None
    c, t = inst.c, inst.t
    z = [0] * N2
    for v in order:
        zv = z[v] + c[v]
        tv = t[v]
        for w in out[v]:
            val = zv + tv[w]
            if val > z[w]:
                z[w] = val
    tau = [0] * N2
    for v in reversed(order):
        best = 0
        tv = t[v]
        for w in out[v]:
            val = c[v] + tv[w] + tau[w]
            if val > best:
                best = val
        tau[v] = best if out[v] else 0
    n1 = inst.n + 1
    cmax = z[n1]
    crit_arcs = []
    for v in order:
        a = z[v] + c[v]
        tv = t[v]
        for w in out[v]:
            if abs(a + tv[w] + tau[w] - cmax) <= eps:
                crit_arcs.append((v, w))
    e = Eval()
    e.order = order
    e.pos = [0] * N2
    for k, v in enumerate(order):
        e.pos[v] = k
    e.z, e.tau, e.cmax, e.crit_arcs = z, tau, cmax, crit_arcs
    e.crit = len(crit_arcs)
    return e


def check_feasible(inst, sol):
    out, inn = sol
    n, m, r = inst.n, inst.m, inst.r
    assert sum(out[0].values()) == m, "source"
    assert sum(inn[n + 1].values()) == m, "sink"
    for j in range(1, n + 1):
        assert sum(inn[j].values()) == r[j], f"inflow {j}"
        assert sum(out[j].values()) == r[j], f"outflow {j}"
    for i in range(n + 2):
        for j, f in out[i].items():
            assert inn[j][i] == f and f > 0 and f <= r[i], "arc"
    e = evaluate(inst, sol)
    assert e is not None, "cycle"
    return e.cmax


# ----------------------------------------------------------------------------
# Spatial preprocessing
# ----------------------------------------------------------------------------

def hilbert_index(N, x, y):
    d, s = 0, N // 2
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s //= 2
    return d


def symmetries(N):
    return [lambda x, y: (x, y), lambda x, y: (N - 1 - x, y),
            lambda x, y: (x, N - 1 - y), lambda x, y: (N - 1 - x, N - 1 - y),
            lambda x, y: (y, x), lambda x, y: (N - 1 - y, x),
            lambda x, y: (y, N - 1 - x), lambda x, y: (N - 1 - y, N - 1 - x)]


class Spatial:
    """Candidate lists and priority keys (computed once per instance)."""

    def __init__(self, inst, kappa):
        n = inst.n
        N = 1
        while N < max(inst.W, inst.H):
            N *= 2
        syms = symmetries(N)
        self.hil = []
        for g in syms:
            self.hil.append([hilbert_index(N, *g(*inst.loc_p[j])) for j in range(n + 2)])
        dep = [h[0] for h in self.hil]
        # put the default symmetry (curve starting nearest to the depot) first
        k0 = min(range(8), key=lambda k: dep[k])
        self.hil = [self.hil[k0]] + [self.hil[k] for k in range(8) if k != k0]
        self.cand = []
        for i in range(n + 1):
            js = sorted((j for j in range(1, n + 1) if j != i), key=lambda j: inst.t[i][j])
            self.cand.append(set(js[:kappa]) | {n + 1})
        self.cand.append(set())


# ----------------------------------------------------------------------------
# Construction (list scheduling)
# ----------------------------------------------------------------------------

def rule_keys(inst, sp, rule, rng):
    n = inst.n
    tasks = list(range(1, n + 1))
    if rule == "random":
        raw = {j: rng.random() for j in tasks}
    elif rule.startswith("hilbert"):
        k = int(rule[7:]) if len(rule) > 7 else 0
        raw = {j: sp.hil[k][j] for j in tasks}
    elif rule == "sweep":
        xs = sorted(set(inst.loc_p[j][0] for j in tasks))
        col = {x: k for k, x in enumerate(xs)}
        raw = {}
        for j in tasks:
            x, y = inst.loc_p[j]
            raw[j] = col[x] * 10 ** 6 + (y if col[x] % 2 == 0 else 10 ** 5 - y)
    elif rule == "depot":
        raw = {j: inst.t[0][j] for j in tasks}
    elif rule == "reqfirst":
        raw = {j: -inst.r[j] * 10 ** 9 + sp.hil[0][j] for j in tasks}
    else:
        raise ValueError(rule)
    ranked = sorted(tasks, key=lambda j: raw[j])
    return {j: k / n for k, j in enumerate(ranked)}


def construct(inst, sp, rule, eta, rng):
    n, m = inst.n, inst.m
    keys = rule_keys(inst, sp, rule, rng)
    order = sorted(range(1, n + 1), key=lambda j: keys[j] + eta * rng.random())
    sol = empty_sol(n)
    last = [0] * m
    rel = [0] * m
    t, c, r = inst.t, inst.c, inst.r
    for j in order:
        arr = sorted(range(m), key=lambda k: (rel[k] + t[last[k]][j], t[last[k]][j], rng.random()))
        chosen = arr[:r[j]]
        zj = max(rel[k] + t[last[k]][j] for k in chosen)
        for k in chosen:
            add_flow(sol, last[k], j, 1)
            last[k] = j
            rel[k] = zj + c[j]
    for k in range(m):
        add_flow(sol, last[k], n + 1, 1)
    return sol


# ----------------------------------------------------------------------------
# Cut redistribution
# ----------------------------------------------------------------------------

def chi_scores(inst, ev):
    """chi[zeta] = number of critical arcs crossing the cut after position zeta."""
    N2 = inst.n + 2
    diff = [0] * (N2 + 1)
    for (i, j) in ev.crit_arcs:
        diff[ev.pos[i]] += 1
        diff[ev.pos[j]] -= 1
    chi, s = [0] * N2, 0
    for k in range(N2):
        s += diff[k]
        chi[k] = s
    return chi  # valid cut positions: zeta in 1..n+1 means left = order[:zeta]; use chi[zeta-1]


def redistribute(inst, sol, ev, zeta, grid, rng, sp=None, eps=1e-9):
    """Cut redistribution. grid=False: Algorithm 3 of Nguyen et al. (one random
    critical arc per step, stop at the first failure). grid=True: candidate-list
    restriction, all critical crossing arcs tried, partial result kept."""
    sol = copy_sol(sol)
    out, inn = sol
    order = ev.order
    left = order[:zeta]
    inleft = [False] * (inst.n + 2)
    for v in left:
        inleft[v] = True
    right = order[zeta:]
    z, tau, c, t, r = ev.z, ev.tau, inst.c, inst.t, inst.r
    A = {i: z[i] + c[i] for i in left}

    def delta(i, j):
        return A[i] + t[i][j] + tau[j]

    def crossing_pos():
        return [(i, j) for i in left for j in out[i] if not inleft[j]]

    if grid:
        incr_targets = {i: [j for j in (sp.cand[i] | set(out[i])) if not inleft[j]] for i in left}
    else:
        incr_targets = {i: right for i in left}

    theta = ev.cmax

    def bfs(a, b):
        # alternating chain a -> j1 <- i2 -> j2 ... iK -> b
        parent_left = {a: None}
        parent_right = {}
        dq = deque([a])
        while dq:
            i = dq.popleft()
            if i != a and (not grid or b in sp.cand[i] or b in out[i]) and \
                    delta(i, b) < theta - eps and out[i].get(b, 0) < r[i]:
                incs, decs = [(i, b)], []
                cur = i
                while cur != a:
                    j = parent_left[cur]
                    decs.append((cur, j))
                    prev = parent_right[j]
                    incs.append((prev, j))
                    cur = prev
                return incs, decs
            for j in incr_targets[i]:
                if j == b or j in parent_right:
                    continue
                if out[i].get(j, 0) >= r[i] or delta(i, j) >= theta - eps:
                    continue
                parent_right[j] = i
                for i2 in inn[j]:
                    if inleft[i2] and i2 not in parent_left:
                        parent_left[i2] = j
                        dq.append(i2)
        return None

    while True:
        cross = crossing_pos()
        amax = [(i, j) for (i, j) in cross if delta(i, j) >= theta - eps]
        if not amax:
            theta = max(delta(i, j) for (i, j) in cross)
            continue
        rng.shuffle(amax)
        if not grid:
            amax = amax[:1]
        all_ok = True
        for (a, b) in amax:
            while out[a].get(b, 0) > 0:
                res = bfs(a, b)
                if res is None:
                    break
                incs, decs = res
                d = out[a][b]
                for (i, j) in decs:
                    d = min(d, out[i][j])
                for (i, j) in incs:
                    d = min(d, r[i] - out[i].get(j, 0))
                add_flow(sol, a, b, -d)
                for (i, j) in decs:
                    add_flow(sol, i, j, -d)
                for (i, j) in incs:
                    add_flow(sol, i, j, d)
            if out[a].get(b, 0) > 0:
                all_ok = False
                if not grid:
                    break
        if not all_ok:
            break
        cross = crossing_pos()
        new_theta = max(delta(i, j) for (i, j) in cross)
        if new_theta >= theta - eps:
            # some critical arcs remain (generic mode handles one arc at a time)
            if grid:
                break
            theta = min(theta, new_theta)
            continue
        theta = new_theta
    return sol


# ----------------------------------------------------------------------------
# Reroute and reverse moves
# ----------------------------------------------------------------------------

def reroute(inst, sol, i, j, u, v, q):
    s = copy_sol(sol)
    add_flow(s, i, j, -q); add_flow(s, u, v, -q)
    add_flow(s, i, v, q); add_flow(s, u, j, q)
    return s


def reroute_candidates(inst, sol, ev, i, j, grid, sp, k, rng):
    """Second arcs (u, v) for a reroute of arc (i, j). Returns list of (u, v, q)."""
    out = sol[0]
    r, t, pos = inst.r, inst.t, ev.pos
    fij = out[i][j]
    res = []
    if grid:
        # rank every feasible second arc by travel saving; acyclicity is tested
        # by the caller (the position/candidate-list filter of the companion
        # document left this set empty in practice)
        for u in range(inst.n + 1):
            if u == i or u == j:
                continue
            for v, fuv in out[u].items():
                if v == j or v == i:
                    continue
                q = min(fij, fuv)
                if out[i].get(v, 0) + q > r[i] or out[u].get(j, 0) + q > r[u]:
                    continue
                saving = t[i][j] + t[u][v] - t[i][v] - t[u][j]
                res.append((saving, u, v, q))
        res.sort(key=lambda x: -x[0])
        return [(u, v, q) for (_, u, v, q) in res[:k]]
    arcs = [(u, v) for u in range(inst.n + 1) for v in out[u]
            if u != i and v != j and v != i and u != j]
    rng.shuffle(arcs)
    for (u, v) in arcs:
        q = min(fij, out[u][v])
        if out[i].get(v, 0) + q > r[i] or out[u].get(j, 0) + q > r[u]:
            continue
        res.append((u, v, q))
        if len(res) >= k:
            break
    return res


def reverse(inst, sol, ev, i, j):
    """Reverse move of Poppenborg & Knust on arc (i, j); may create a cycle."""
    n = inst.n
    if i == 0 or j == n + 1:
        return None
    s = copy_sol(sol)
    out, inn = s
    fij = out[i][j]
    z, c, t = ev.z, inst.c, inst.t
    preds = sorted((u for u in inn[i]), key=lambda u: z[u] + c[u] + t[u][i])
    U, acc = [], 0
    for u in preds:
        U.append(u); acc += inn[i][u]
        if acc >= fij:
            break
    succs = sorted((w for w in out[j]), key=lambda w: t[j][w])
    V, acc2 = [], 0
    for w in succs:
        V.append(w); acc2 += out[j][w]
        if acc2 >= fij:
            break
    if acc < fij or acc2 < fij or j in U or i in V:
        return None
    add_flow(s, i, j, -fij)
    add_flow(s, j, i, fij)
    rem = fij
    for u in U:
        q = min(out[u][i], rem)
        add_flow(s, u, i, -q); add_flow(s, u, j, q); rem -= q
    rem = fij
    for w in V:
        q = min(out[j][w], rem)
        add_flow(s, j, w, -q); add_flow(s, i, w, q); rem -= q
    return s
