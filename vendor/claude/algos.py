"""Genetic algorithm and tabu search for the PDP-CR, each in a generic and a
grid-aware version (2 x 2 design of the companion document)."""
import math
import random
import time
from pdpcr import *

GRID_RULES = [f"hilbert{k}" for k in range(8)] + ["sweep", "depot", "reqfirst"]


def key_of(ev, grid):
    return (ev.cmax, ev.crit) if grid else (ev.cmax,)


def new_construction(inst, sp, grid, rng, k):
    if grid:
        return construct(inst, sp, GRID_RULES[k % len(GRID_RULES)], rng.uniform(0, 0.3), rng)
    return construct(inst, sp, "random", 0.0, rng)


def draw_cuts(inst, ev, grid, rng, k, tabu=()):
    """Draw up to k distinct cut positions zeta in 1..n+1, not tabu."""
    N2 = inst.n + 2
    cand = [z for z in range(1, N2) if z not in tabu]
    if not cand:
        return []
    if not grid:
        rng.shuffle(cand)
        return cand[:k]
    chi = chi_scores(inst, ev)
    w = [1.0 / max(1, chi[z - 1]) for z in cand]
    chosen = []
    for _ in range(min(k, len(cand))):
        tot = sum(w)
        x = rng.random() * tot
        acc = 0.0
        for idx in range(len(cand)):
            acc += w[idx]
            if acc >= x:
                break
        chosen.append(cand.pop(idx)); w.pop(idx)
    return chosen


def critical_reroute_arc(inst, ev, rng):
    arcs = [(i, j) for (i, j) in ev.crit_arcs]
    tmax = max(inst.t[i][j] for (i, j) in arcs)
    best = [(i, j) for (i, j) in arcs if inst.t[i][j] == tmax]
    return rng.choice(best)


# ----------------------------------------------------------------------------
# Tabu search
# ----------------------------------------------------------------------------

def tabu_search(inst, sp, grid, T, seed, beta=0.3, I1=150, I2=300, trace=None):
    rng = random.Random(seed)
    n = inst.n
    tl_add_size, tl_redis_size = max(1, n // 8), max(1, n // 15)
    n_reroute, n_reverse, n_redis = max(1, n // 5), max(1, n // 15), max(1, n // 15)
    t0 = time.time()
    k_restart = 0
    S = new_construction(inst, sp, grid, rng, k_restart)
    evS = evaluate(inst, S)
    best, evb = S, evS
    tl_add, tl_redis = [], []
    it = since_best = since_cur = 0
    cur_best_key = key_of(evS, grid)
    if trace is not None:
        trace.append((0.0, evb.cmax))
    while time.time() - t0 < T:
        it += 1
        cands = []  # (key, kind, sol, ev, tabu_info)
        for z in draw_cuts(inst, evS, grid, rng, n_redis, tl_redis):
            s2 = redistribute(inst, S, evS, z, grid, rng, sp)
            e2 = evaluate(inst, s2)
            cands.append((key_of(e2, grid), "redis", s2, e2, z))
        i, j = critical_reroute_arc(inst, evS, rng)
        for (u, v, q) in reroute_candidates(inst, S, evS, i, j, grid, sp, n_reroute, rng):
            if (i, v) in tl_add or (u, j) in tl_add:
                continue
            s2 = reroute(inst, S, i, j, u, v, q)
            e2 = evaluate(inst, s2)
            if e2 is not None:
                cands.append((key_of(e2, grid), "reroute", s2, e2, ((i, j), (u, v))))
        crit = [(a, b) for (a, b) in evS.crit_arcs if a != 0 and b != n + 1]
        rng.shuffle(crit)
        for (a, b) in crit[:n_reverse]:
            if (b, a) in tl_add:
                continue
            s2 = reverse(inst, S, evS, a, b)
            if s2 is None:
                continue
            e2 = evaluate(inst, s2)
            if e2 is not None:
                cands.append((key_of(e2, grid), "reverse", s2, e2, (a, b)))
        if not cands:
            S = new_construction(inst, sp, grid, rng, k_restart := k_restart + 1)
            evS = evaluate(inst, S)
            continue
        cands.sort(key=lambda x: x[0])
        top = cands[0]
        if top[0] < key_of(evS, grid):
            chosen = top
        elif top[1] == "redis" and rng.random() >= beta:
            others = [x for x in cands if x[1] != "redis"]
            chosen = others[0] if others else top
        else:
            chosen = top
        _, kind, S, evS, info = chosen
        if kind == "redis":
            tl_redis.append(info)
        elif kind == "reroute":
            tl_add.extend(info)
        else:
            tl_add.append(info)
        tl_add = tl_add[-tl_add_size:]
        tl_redis = tl_redis[-tl_redis_size:]
        if key_of(evS, grid) < key_of(evb, grid):
            best, evb = S, evS
            since_best = 0
            if trace is not None:
                trace.append((time.time() - t0, evb.cmax))
        else:
            since_best += 1
        if key_of(evS, grid) < cur_best_key:
            cur_best_key, since_cur = key_of(evS, grid), 0
        else:
            since_cur += 1
        if since_best >= I2:
            k_restart += 1
            S = new_construction(inst, sp, grid, rng, k_restart)
            evS = evaluate(inst, S)
            tl_add, tl_redis = [], []
            since_best = since_cur = 0
            cur_best_key = key_of(evS, grid)
        elif since_cur >= I1:
            S, evS = best, evb
            since_cur = 0
            cur_best_key = key_of(evS, grid)
    return best, evb, it


# ----------------------------------------------------------------------------
# Genetic algorithm
# ----------------------------------------------------------------------------

def flows_dict(sol):
    return {(i, j): f for i, d in enumerate(sol[0]) for j, f in d.items()}


def flow_distance(fa, fb):
    keys = fa.keys() | fb.keys()
    return sum(abs(fa.get(k, 0) - fb.get(k, 0)) for k in keys) / 2


def cycle_decomposition(f1, f2, rng):
    """Alternating-cycle decomposition of f2 - f1. Returns list of (arcs, alpha),
    arcs = list of ((i, j), sign)."""
    h = {}
    for k in f1.keys() | f2.keys():
        d = f2.get(k, 0) - f1.get(k, 0)
        if d != 0:
            h[k] = d
    adj = {}
    cap = {}
    for (i, j), d in h.items():
        if d > 0:
            e = (("t", i), ("h", j))
        else:
            e = (("h", j), ("t", i))
        adj.setdefault(e[0], []).append(e[1])
        cap[e] = abs(d)
    cycles = []

    def first_out(x):
        lst = adj.get(x, [])
        for y in lst:
            if cap.get((x, y), 0) > 0:
                return y
        return None

    starts = list(adj.keys())
    rng.shuffle(starts)
    for s0 in starts:
        while first_out(s0) is not None:
            path, seen = [s0], {s0: 0}
            x = s0
            while True:
                y = first_out(x)
                if y in seen:
                    cyc = path[seen[y]:] + [y]
                    break
                seen[y] = len(path)
                path.append(y)
                x = y
            edges = list(zip(cyc[:-1], cyc[1:]))
            alpha = min(cap[e] for e in edges)
            arcs = []
            for (a, b) in edges:
                cap[(a, b)] -= alpha
                if a[0] == "t":
                    arcs.append(((a[1], b[1]), +1))
                else:
                    arcs.append(((b[1], a[1]), -1))
            cycles.append((arcs, alpha))
    return cycles


def sol_from_flows(n, fd):
    s = empty_sol(n)
    for (i, j), f in fd.items():
        if f:
            add_flow(s, i, j, f)
    return s


def crossover(inst, p1, p2, grid, rng, phi=0.3, smin=0.5):
    f1, f2 = flows_dict(p1), flows_dict(p2)
    cycles = cycle_decomposition(f1, f2, rng)
    if grid:
        W, H = inst.W, inst.H
        w, h = max(1, int(W * math.sqrt(phi))), max(1, int(H * math.sqrt(phi)))
        x0, y0 = rng.randrange(max(1, W - w + 1)), rng.randrange(max(1, H - h + 1))
        TZ = {j for j in range(1, inst.n + 1)
              if x0 <= inst.loc_p[j][0] < x0 + w and y0 <= inst.loc_p[j][1] < y0 + h}
        scored = []
        for (arcs, alpha) in cycles:
            sc = sum(1 for ((i, j), s) in arcs if j in TZ) / len(arcs)
            if sc >= smin:
                scored.append((sc, rng.random(), arcs, alpha))
        scored.sort(key=lambda x: (-x[0], x[1]))
        plan = [(arcs, alpha) for (_, _, arcs, alpha) in scored]
    else:
        rng.shuffle(cycles)
        plan = [(arcs, rng.randint(0, alpha)) for (arcs, alpha) in cycles]
    g1, g2 = dict(f1), dict(f2)
    n = inst.n
    for (arcs, q) in plan:
        if q == 0:
            continue
        c1, c2 = dict(g1), dict(g2)
        for ((i, j), s) in arcs:
            c1[(i, j)] = c1.get((i, j), 0) + q * s
            c2[(i, j)] = c2.get((i, j), 0) - q * s
        s1, s2 = sol_from_flows(n, c1), sol_from_flows(n, c2)
        if evaluate(inst, s1) is not None and evaluate(inst, s2) is not None:
            g1, g2 = c1, c2
    return sol_from_flows(n, g1), sol_from_flows(n, g2)


def mutate(inst, sol, ev, grid, sp, rng, eps_m=0.2):
    arcs = [(i, j) for i in range(inst.n + 1) for j in sol[0][i]]
    for _ in range(5):
        i, j = rng.choice(arcs)
        if grid:
            cands = reroute_candidates(inst, sol, ev, i, j, True, sp, 10, rng)
            if not cands:
                continue
            u, v, q = cands[0] if rng.random() > eps_m else rng.choice(cands)
            q = rng.randint(1, q)
        else:
            cands = reroute_candidates(inst, sol, ev, i, j, False, sp, 1, rng)
            if not cands:
                continue
            u, v, q = cands[0]
            q = rng.randint(1, q)
        s2 = reroute(inst, sol, i, j, u, v, q)
        e2 = evaluate(inst, s2)
        if e2 is not None:
            return s2, e2
    return sol, ev


def intensify(inst, sol, ev, grid, sp, rng, K=3):
    k = 0
    tried = set()
    while k < K:
        zs = draw_cuts(inst, ev, grid, rng, 1, tried)
        if not zs:
            break
        z = zs[0]
        tried.add(z)
        s2 = redistribute(inst, sol, ev, z, grid, rng, sp)
        e2 = evaluate(inst, s2)
        if key_of(e2, grid) < key_of(ev, grid):
            sol, ev, k, tried = s2, e2, 0, set()
        else:
            k += 1
    return sol, ev


def genetic(inst, sp, grid, T, seed, P=30, pc=0.9, pm=0.3, pl=0.2, dmin=2, trace=None, crit_mut=False):
    rng = random.Random(seed)
    t0 = time.time()
    pop = []
    for k in range(P):
        s = new_construction(inst, sp, grid, rng, k)
        pop.append((s, evaluate(inst, s)))
    pop.sort(key=lambda x: key_of(x[1], grid))
    best = pop[0]
    if trace is not None:
        trace.append((0.0, best[1].cmax))
    gens = 0

    def tour():
        a, b = rng.sample(pop, 2)
        return a if key_of(a[1], grid) <= key_of(b[1], grid) else b

    while time.time() - t0 < T:
        gens += 1
        off = []
        while len(off) < P and time.time() - t0 < T:
            (s1, e1), (s2, e2) = tour(), tour()
            if rng.random() < pc:
                c1, c2 = crossover(inst, s1, s2, grid, rng)
                kids = [(c1, evaluate(inst, c1)), (c2, evaluate(inst, c2))]
            else:
                kids = [(s1, e1), (s2, e2)]
            for (s, e) in kids:
                if rng.random() < pm:
                    if crit_mut:
                        s, e = mutate_critical(inst, s, e, grid, sp, rng)
                    else:
                        s, e = mutate(inst, s, e, grid, sp, rng)
                if rng.random() < pl:
                    s, e = intensify(inst, s, e, grid, sp, rng)
                if len(off) < P:
                    off.append((s, e))
                if key_of(e, grid) < key_of(best[1], grid):
                    best = (s, e)
                    if trace is not None:
                        trace.append((time.time() - t0, e.cmax))
        merged = sorted(pop + off, key=lambda x: key_of(x[1], grid))
        kept, kept_f, rejected = [], [], []
        for (s, e) in merged:
            fd = flows_dict(s)
            if all(flow_distance(fd, g) >= dmin for g in kept_f):
                kept.append((s, e)); kept_f.append(fd)
            else:
                rejected.append((s, e))
            if len(kept) == P:
                break
        while len(kept) < P and rejected:
            kept.append(rejected.pop(0))
        pop = kept
    return best[0], best[1], gens


def mutate_critical(inst, sol, ev, grid, sp, rng, k_reroute=10, k_reverse=3):
    """Mutation built on the tabu search moves: best of several reroute and
    reverse moves on critical arcs (applied even if worse)."""
    n = inst.n
    best = None
    i, j = critical_reroute_arc(inst, ev, rng)
    for (u, v, q) in reroute_candidates(inst, sol, ev, i, j, grid, sp, k_reroute, rng):
        s2 = reroute(inst, sol, i, j, u, v, q)
        e2 = evaluate(inst, s2)
        if e2 is not None and (best is None or key_of(e2, grid) < key_of(best[1], grid)):
            best = (s2, e2)
    crit = [(a, b) for (a, b) in ev.crit_arcs if a != 0 and b != n + 1]
    rng.shuffle(crit)
    for (a, b) in crit[:k_reverse]:
        s2 = reverse(inst, sol, ev, a, b)
        if s2 is None:
            continue
        e2 = evaluate(inst, s2)
        if e2 is not None and (best is None or key_of(e2, grid) < key_of(best[1], grid)):
            best = (s2, e2)
    return best if best else (sol, ev)
