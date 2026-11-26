"""Synchronized relocation of one critical task in a sparse PDP-CR flow."""

from dataclasses import dataclass
from time import process_time


EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class _CrossingArc:
    tail: int
    head: int
    capacity: int
    transit: float
    arrival: float
    departure: float


def _bypass_task(flow, task, ops, rng):
    """Use one pairing of incoming and outgoing flow when removing the task."""
    reduced = ops.copy_sol(flow)
    outgoing = list(flow[0][task].items())
    rng.shuffle(outgoing)
    destinations = iter(outgoing)
    head, space = next(destinations)
    for tail, units in flow[1][task].items():
        ops.add_flow(reduced, tail, task, -units)
        while units:
            if not space:
                head, space = next(destinations)
            moved = min(units, space)
            ops.add_flow(reduced, task, head, -moved)
            ops.add_flow(reduced, tail, head, moved)
            units -= moved
            space -= moved
    return reduced


def _assignment_at_threshold(arcs, required, limit, deadline):
    """Check a makespan bound and return the selected robot units."""
    mandatory = [arc for arc in arcs if arc.transit > limit]
    remaining = required - sum(arc.capacity for arc in mandatory)
    if remaining < 0:
        return None
    earliest = max((arc.arrival for arc in mandatory), default=0.0)
    longest = max((arc.departure for arc in mandatory), default=0.0)
    if earliest + longest > limit:
        return None
    fixed = [(arc, arc.capacity) for arc in mandatory]
    if not remaining:
        return fixed

    optional = [arc for arc in arcs if arc.transit <= limit]
    arrival_limits = {earliest, *(arc.arrival for arc in optional if arc.arrival > earliest)}
    for arrival in sorted(arrival_limits):
        if process_time() >= deadline:
            return None
        if arrival + longest > limit:
            break
        eligible = [arc for arc in optional
                    if arc.arrival <= arrival and arrival + arc.departure <= limit]
        if sum(arc.capacity for arc in eligible) >= remaining:
            selected = fixed.copy()
            for arc in eligible:
                amount = min(remaining, arc.capacity)
                selected.append((arc, amount))
                remaining -= amount
                if not remaining:
                    return selected
    return None


def _best_on_cut(arcs, required, incumbent, deadline):
    """Find the lowest attainable makespan below the incumbent on one cut.

    A path crosses the cut either through a retained arc or through the
    reinserted task. The optimal score is therefore one of the retained
    transit lengths or a sum of an arrival and a departure. Search the
    thresholds monotonically without enumerating robot subsets.
    """
    upper = incumbent - EPSILON
    if not arcs or process_time() >= deadline:
        return None
    best = _assignment_at_threshold(arcs, required, upper, deadline)
    if best is None:
        return None

    levels = {arc.transit for arc in arcs if arc.transit <= upper}
    for incoming in arcs:
        if process_time() >= deadline:
            break
        levels.update(incoming.arrival + outgoing.departure for outgoing in arcs
                      if incoming.arrival + outgoing.departure <= upper)
    thresholds = sorted(levels) if process_time() < deadline else ()
    low, high = 0, len(thresholds)
    while low < high and process_time() < deadline:
        middle = (low + high) // 2
        assignment = _assignment_at_threshold(arcs, required, thresholds[middle], deadline)
        if assignment is None:
            low = middle + 1
        else:
            best = assignment
            high = middle
    chosen = {arc: amount for arc, amount in best}
    untouched = max((arc.transit for arc in arcs if chosen.get(arc, 0) < arc.capacity),
                    default=0.0)
    via_task = (max(arc.arrival for arc, _ in best) +
                max(arc.departure for arc, _ in best))
    return max(untouched, via_task), best


def _critical_tasks(inst, evaluation):
    critical = sorted(evaluation.crit_arcs, key=lambda arc: -inst.t[arc[0]][arc[1]])
    return dict.fromkeys(task for arc in critical for task in arc if 1 <= task <= inst.n)


def relocate_task(inst, flow, ev, ops, rng, deadline, max_tasks=6, *, score_limit=float("inf")):
    """Return a strictly improving (flow, evaluation), or None.

    Each sampled critical task is removed with one balance-preserving bypass
    pairing. A topological cut of the residual support fixes its new place
    in the order. The cut solver exactly chooses the units to route through
    the task for that pairing and order. The process-time deadline bounds
    the search, which may return a better assignment found before timeout.
    """
    best_score, best_plan = min(ev.cmax, score_limit), None
    for task in list(_critical_tasks(inst, ev))[:max_tasks]:
        if process_time() >= deadline:
            break
        if not inst.r[task]:
            continue
        reduced = _bypass_task(flow, task, ops, rng)
        residual = ops.evaluate(inst, reduced)
        if residual is None:
            raise AssertionError("Bypassing a task in a DAG must preserve acyclicity")

        out, incoming = reduced
        order = [node for node in residual.order if node != task]
        finish = [start + duration for start, duration in zip(residual.z, inst.c)]
        arrive = [finish[node] + inst.t[node][task] for node in range(inst.n + 2)]
        depart = [inst.c[task] + inst.t[task][node] + residual.tau[node]
                  for node in range(inst.n + 2)]
        crossing = {}
        for tail in order[:-1]:
            if process_time() >= deadline:
                break
            for previous in incoming[tail]:
                del crossing[previous, tail]
            for head, units in out[tail].items():
                crossing[tail, head] = _CrossingArc(
                    tail, head, units, finish[tail] + inst.t[tail][head] + residual.tau[head],
                    arrive[tail], depart[head],
                )
            candidate = _best_on_cut(tuple(crossing.values()), inst.r[task], best_score, deadline)
            if candidate is not None and candidate[0] < best_score - EPSILON:
                best_score, assignment = candidate
                best_plan = reduced, task, assignment

    if best_plan is None:
        return None
    changed, task, assignment = best_plan
    for arc, amount in assignment:
        ops.add_flow(changed, arc.tail, arc.head, -amount)
        ops.add_flow(changed, arc.tail, task, amount)
        ops.add_flow(changed, task, arc.head, amount)
    evaluation = ops.evaluate(inst, changed)
    if evaluation is None or abs(evaluation.cmax - best_score) > 1e-7:
        raise AssertionError("Relocated task score differs from the whole DAG evaluation")
    return (changed, evaluation) if evaluation.cmax < ev.cmax - EPSILON else None
