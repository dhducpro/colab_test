"""Tabu search followed by bounded search around its best robot flow."""

from random import Random
from time import process_time

import method_thao
import method_claude
from method_claude import dense_flow, prepare_sparse


BASE_FRACTION = {"thao": 0.90, "claude": 5 / 12}


def run(instance, initial, seed, seconds, settings, *, engine="thao"):
    from pdpcr.tsr_elite import polish_elite

    started = process_time()
    deadline = started + seconds
    base_seconds = seconds * BASE_FRACTION[engine]
    if engine == "thao":
        base = method_thao.run(instance, initial, seed, base_seconds, settings)
    else:
        base = method_claude.run(instance, initial, seed, base_seconds, settings,
                                 improved=True, relocate=True)
    base_ended = process_time()
    stats = base["diagnostics"]
    stats["base_engine"] = engine
    stats["score_before_polishing"] = base["reported"]
    stats["base_cpu_s"] = base_ended - started
    stats["elite_improvements"] = 0
    if base_ended >= deadline:
        return base

    inst, flow, ops = prepare_sparse(instance, base["flow"])
    ev = ops.evaluate(inst, flow)
    best, best_ev = flow, ev

    def record(candidate, evaluation):
        nonlocal best, best_ev
        now = process_time()
        if now <= deadline and evaluation.cmax < best_ev.cmax - 1e-8:
            best, best_ev = candidate, evaluation
            base["history"].append((now - started, evaluation.cmax))
            stats["elite_improvements"] += 1

    polish_elite(inst, flow, ev, ops, Random(seed), deadline, on_improvement=record)
    stats["elite_cpu_s"] = process_time() - base_ended
    return {**base, "flow": dense_flow(best), "reported": float(best_ev.cmax)}
