"""Critical-arc candidate lists for Claude's sparse-flow tabu search."""
from collections import deque
from dataclasses import dataclass
from random import Random
from time import process_time

from .tsr_relocate import relocate_task


TOLERANCE = 1e-9


@dataclass
class Move:
    kind: str
    flow: tuple
    evaluation: object
    removed: tuple = ()
    cut: int | None = None


def _creates_reroute_cycle(out, position, i, j, u, v, amount):
    """Check reachability in the modified support before copying the flow."""
    added = ((i, v), (u, j))
    backward = [(a, b) for a, b in added if position[a] >= position[b]]
    if not backward:
        return False
    removed = set()
    if out[i][j] == amount:
        removed.add((i, j))
    if out[u][v] == amount:
        removed.add((u, v))
    for tail, head in backward:
        stack, seen = [head], {head}
        while stack:
            node = stack.pop()
            if node == tail:
                return True
            for nxt in out[node]:
                if nxt not in seen and (node, nxt) not in removed:
                    seen.add(nxt)
                    stack.append(nxt)
            if node == i and v not in seen:
                seen.add(v)
                stack.append(v)
            if node == u and j not in seen:
                seen.add(j)
                stack.append(j)
    return False


def _reroute_moves(inst, flow, ev, ops, rng, budget, forbidden, best, deadline, stats):
    out = flow[0]
    critical = set(ev.crit_arcs)
    arcs = [(u, v) for u, row in enumerate(out) for v in row]
    pivots = sorted(ev.crit_arcs, key=lambda arc: -inst.t[arc[0]][arc[1]])[:3]
    available = deque()
    for i, j in pivots:
        partners = []
        for u, v in arcs:
            if i == u or j == v or i == v or u == j:
                continue
            amount = min(out[i][j], out[u][v])
            breaks_critical = amount == out[i][j] or (
                (u, v) in critical and amount == out[u][v]
            )
            added_time = inst.t[i][v] + inst.t[u][j]
            saved_time = inst.t[i][j] + inst.t[u][v] - added_time
            partners.append((not breaks_critical, -saved_time, u, v, amount))
        rng.shuffle(partners)
        # Keep a random part of each list to explore beyond short travel arcs.
        ordered = sorted(partners, key=lambda item: item[:2])
        shortlist = ordered[:3] + partners[:2] + ordered[3:budget]
        available.append((i, j, iter(shortlist + partners[2:])))

    tried = set()
    attempts = 0
    accepted = 0
    while available and accepted < budget and attempts < 8 * budget:
        if process_time() >= deadline:
            return
        i, j, partners = available.popleft()
        candidate = next(partners, None)
        if candidate is None:
            continue
        available.append((i, j, partners))
        _, _, u, v, amount = candidate
        key = tuple(sorted(((i, j), (u, v))))
        if key in tried:
            continue
        tried.add(key)
        attempts += 1
        stats["reroute_attempted"] += 1
        if _creates_reroute_cycle(out, ev.pos, i, j, u, v, amount):
            stats["reroute_cycles_filtered"] += 1
            continue
        neighbor = ops.reroute(inst, flow, i, j, u, v, amount)
        score = ops.evaluate(inst, neighbor)
        if score is None:
            raise AssertionError("Reroute cycle filter and schedule evaluator disagree.")
        if ((i, v) in forbidden or (u, j) in forbidden) and score.cmax >= best - TOLERANCE:
            continue
        accepted += 1
        stats["reroute_admissible"] += 1
        yield Move("reroute", neighbor, score, key)


def _neighbors(inst, flow, ev, ops, rng, config, tabu_arcs, tabu_cuts, best, deadline, stats):
    yield from _reroute_moves(
        inst, flow, ev, ops, rng, config["n_reroute"], tabu_arcs, best, deadline, stats
    )
    critical = [(i, j) for i, j in ev.crit_arcs if i != 0 and j != inst.n + 1]
    rng.shuffle(critical)
    accepted = 0
    for i, j in critical:
        if accepted >= config["n_reverse"] or process_time() >= deadline:
            break
        neighbor = ops.reverse(inst, flow, ev, i, j)
        score = ops.evaluate(inst, neighbor) if neighbor is not None else None
        if score is None or ((j, i) in tabu_arcs and score.cmax >= best - TOLERANCE):
            continue
        accepted += 1
        yield Move("reverse", neighbor, score, ((i, j),))

    cuts = ops.draw_cuts(inst, ev, False, rng, config["n_redistribute"], tabu_cuts)
    for cut in cuts:
        if process_time() >= deadline:
            break
        neighbor = ops.redistribute(inst, flow, ev, cut, False, rng)
        stats["cuts_tried"] += 1
        if neighbor[0] == flow[0]:
            stats["cuts_unchanged"] += 1
            continue
        yield Move("redistribute", neighbor, ops.evaluate(inst, neighbor), cut=cut)


def tabu_search(inst, initial, ops, seconds, seed, config, *, relocate=False):
    """Use exact makespan to select from the bounded, feasible candidate list."""
    started = process_time()
    deadline = started + seconds
    rng = Random(seed)
    current = best = initial
    ev = best_ev = ops.evaluate(inst, current)
    trace = [(0.0, ev.cmax)]
    tabu_arcs = deque(maxlen=config["tenure_arc"])
    tabu_cuts = deque(maxlen=config["tenure_cut"])
    stats = dict(reroute_attempted=0, reroute_cycles_filtered=0, reroute_admissible=0,
                 cuts_tried=0, cuts_unchanged=0, restarts=0,
                 relocation_trials=0, relocation_improvements=0)
    iterations = since_best = since_current = 0
    local_best = ev.cmax
    next_relocation = started
    while process_time() < deadline:
        moves = list(_neighbors(
            inst, current, ev, ops, rng, config, tabu_arcs, tabu_cuts, best_ev.cmax, deadline, stats
        ))
        iterations += 1
        if moves:
            chosen = min(moves, key=lambda move: move.evaluation.cmax)
            if (chosen.kind == "redistribute" and chosen.evaluation.cmax >= ev.cmax - TOLERANCE
                    and rng.random() >= config["beta"]):
                chosen = min((move for move in moves if move.kind != "redistribute"),
                             key=lambda move: move.evaluation.cmax, default=chosen)
            current, ev = chosen.flow, chosen.evaluation
            if chosen.cut is not None:
                tabu_cuts.append(chosen.cut)
            else:
                tabu_arcs.extend(chosen.removed)

        now = process_time()
        if ev.cmax < best_ev.cmax - TOLERANCE and now <= deadline:
            best, best_ev = current, ev
            trace.append((now - started, best_ev.cmax))
            since_best = 0
        else:
            since_best += 1
        if ev.cmax < local_best - TOLERANCE:
            local_best, since_current = ev.cmax, 0
        else:
            since_current += 1
        if now >= deadline:
            break

        if relocate and since_best >= 40 and now >= next_relocation:
            stats["relocation_trials"] += 1
            result = relocate_task(inst, best, best_ev, ops, rng, min(deadline, now + 0.10))
            now = process_time()
            next_relocation = now + 0.15
            if result is not None and now <= deadline:
                current, ev = best, best_ev = result
                trace.append((now - started, best_ev.cmax))
                local_best, since_best, since_current = ev.cmax, 0, 0
                tabu_arcs.clear()
                tabu_cuts.clear()
                stats["relocation_improvements"] += 1
                continue

        if not moves or since_best >= config["restart_new"]:
            current = ops.new_construction(inst, None, False, rng, 0)
            ev = ops.evaluate(inst, current)
            local_best, since_best, since_current = ev.cmax, 0, 0
            now = process_time()
            if ev.cmax < best_ev.cmax - TOLERANCE and now <= deadline:
                best, best_ev = current, ev
                trace.append((now - started, best_ev.cmax))
            tabu_arcs.clear()
            tabu_cuts.clear()
            stats["restarts"] += 1
        elif since_current >= config["restart_best"]:
            current, ev = best, best_ev
            local_best, since_current = ev.cmax, 0
    return best, best_ev, iterations, trace, stats
