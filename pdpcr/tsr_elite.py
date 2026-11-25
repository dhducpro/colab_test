"""Bounded ruin-and-repair search around the best PDP-CR flows."""

from time import process_time

from .tsr_relocate import (
    EPSILON,
    _best_on_cut,
    _bypass_task,
    _critical_tasks,
    _CrossingArc,
    relocate_task,
)


def _topological_order(out, incoming, excluded, rng):
    degree = [len(row) for row in incoming]
    ready = [node for node, count in enumerate(degree) if not count and node not in excluded]
    order = []
    while ready:
        index = rng.randrange(len(ready))
        node = ready[index]
        ready[index] = ready[-1]
        ready.pop()
        order.append(node)
        for successor in out[node]:
            degree[successor] -= 1
            if not degree[successor]:
                ready.append(successor)
    if len(order) != len(out) - len(excluded):
        raise AssertionError("Expected a residual DAG with the removed tasks isolated")
    return order


def _insert_task(inst, reduced, task, excluded, ops, rng, limit, deadline):
    if process_time() >= deadline:
        return None
    schedule = ops.evaluate(inst, reduced)
    if schedule is None:
        raise AssertionError("Expected an acyclic residual flow")
    out, incoming = reduced
    order = _topological_order(out, incoming, excluded, rng)
    finish = [start + duration for start, duration in zip(schedule.z, inst.c)]
    arrive = [finish[node] + inst.t[node][task] for node in range(inst.n + 2)]
    depart = [inst.c[task] + inst.t[task][node] + schedule.tau[node]
              for node in range(inst.n + 2)]
    crossing, assignment = {}, None
    for tail in order[:-1]:
        if process_time() >= deadline:
            break
        for previous in incoming[tail]:
            del crossing[previous, tail]
        for head, units in out[tail].items():
            crossing[tail, head] = _CrossingArc(
                tail, head, units, finish[tail] + inst.t[tail][head] + schedule.tau[head],
                arrive[tail], depart[head],
            )
        candidate = _best_on_cut(tuple(crossing.values()), inst.r[task], limit, deadline)
        if candidate is not None:
            limit, assignment = candidate

    if assignment is None:
        return None
    restored = ops.copy_sol(reduced)
    for arc, amount in assignment:
        ops.add_flow(restored, arc.tail, arc.head, -amount)
        ops.add_flow(restored, arc.tail, task, amount)
        ops.add_flow(restored, task, arc.head, amount)
    evaluation = ops.evaluate(inst, restored)
    if evaluation is None or abs(evaluation.cmax - limit) > 1e-7:
        raise AssertionError("Reinsertion score differs from the whole DAG evaluation")
    return restored, evaluation


def _pair_trial(inst, flow, first, second, ops, rng, limit, deadline):
    reduced = _bypass_task(flow, first, ops, rng)
    reduced = _bypass_task(reduced, second, ops, rng)
    # Either insertion order can recover a short path; trial order varies.
    if rng.random() < 0.5:
        first, second = second, first
    half = _insert_task(inst, reduced, first, {first, second}, ops, rng, float("inf"), deadline)
    if half is None:
        return None
    return _insert_task(inst, half[0], second, {second}, ops, rng, limit, deadline)


def _signature(flow):
    return tuple(tuple(sorted(row.items())) for row in flow[0])


def polish_elite(inst, flow, ev, ops, rng, deadline, on_improvement=None):
    """Return best (flow, evaluation), using at most the given CPU deadline.

    Alternate single-task relocation with a two-task removal and reinsertion
    neighborhood. The perturbation may briefly increase makespan by 3% over
    the best score; every candidate restores the original task requirements.
    """
    best_flow, best_ev = flow, ev
    current, current_ev = flow, ev
    seen = {_signature(flow)}
    failures, trial = 0, 0
    while process_time() < deadline:
        critical = [task for task in _critical_tasks(inst, current_ev) if inst.r[task]]
        if not critical:
            break
        if trial % 3 == 0 or len(critical) == 1:
            candidate = relocate_task(inst, current, current_ev, ops, rng, deadline, max_tasks=8)
        else:
            first = rng.choice(critical[:8])
            alternatives = [task for task in critical if task != first]
            adjacent = [node for node in (*current[0][first], *current[1][first])
                        if 1 <= node <= inst.n and inst.r[node]]
            second = rng.choice(adjacent if adjacent and rng.random() < 0.6 else alternatives)
            candidate = _pair_trial(inst, current, first, second, ops, rng,
                                    best_ev.cmax * 1.03, deadline)
        trial += 1

        if candidate is None or (key := _signature(candidate[0])) in seen:
            failures += 1
            if failures >= 4:
                current, current_ev = best_flow, best_ev
                failures = 0
            if len(critical) == 1:
                break
            continue
        seen.add(key)
        current, current_ev = candidate
        failures = 0
        if current_ev.cmax < best_ev.cmax - EPSILON:
            best_flow, best_ev = current, current_ev
            if on_improvement is not None:
                on_improvement(best_flow, best_ev)
    return best_flow, best_ev
