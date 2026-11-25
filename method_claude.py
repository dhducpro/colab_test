"""Run the generic TSR in Claude's preliminary package on the common instance."""

from functools import lru_cache
from pathlib import Path
import runpy
import time
from types import SimpleNamespace


VENDOR = Path(__file__).resolve().parent / "vendor" / "claude"


@lru_cache(maxsize=1)
def _load_tsr():
    # Claude's algos.py imports a standalone pdpcr.py. Keep its symbols in a
    # private namespace so the identically named project package is unaffected.
    namespace = runpy.run_path(str(VENDOR / "pdpcr.py"))
    source = (VENDOR / "algos.py").read_text(encoding="utf-8")
    # Expose only the common protocol inputs in the original search loop.
    # Match complete, unique fragments so vendor changes cannot silently drift.
    replacements = [
        ("from pdpcr import *", "# Functions supplied by the vendor pdpcr.py"),
        ("def tabu_search(inst, sp, grid, T, seed, beta=0.3, I1=150, I2=300, trace=None):",
         "def tabu_search(inst, sp, grid, T, seed, initial, config, beta, I1, I2, trace=None):"),
        ("tl_add_size, tl_redis_size = max(1, n // 8), max(1, n // 15)",
         "tl_add_size, tl_redis_size = config['tenure_arc'], config['tenure_cut']"),
        ("n_reroute, n_reverse, n_redis = max(1, n // 5), max(1, n // 15), max(1, n // 15)",
         "n_reroute, n_reverse, n_redis = config['n_reroute'], config['n_reverse'], config['n_redistribute']"),
        ("k_restart = 0\n    S = new_construction(inst, sp, grid, rng, k_restart)",
         "k_restart = 0\n    S = initial"),
        ("tl_add = tl_add[-tl_add_size:]", "tl_add = tl_add[-tl_add_size:] if tl_add_size else []"),
        ("tl_redis = tl_redis[-tl_redis_size:]",
         "tl_redis = tl_redis[-tl_redis_size:] if tl_redis_size else []"),
    ]
    for original, adapted in replacements:
        if source.count(original) != 1:
            raise RuntimeError(f"Unexpected Claude algos.py version at {original!r}")
        source = source.replace(original, adapted)
    exec(compile(source, str(VENDOR / "algos.py"), "exec"), namespace)
    namespace["time"] = SimpleNamespace(time=time.process_time)
    native_reroute = namespace["reroute_candidates"]

    def reroute_with_budget(inst, sol, ev, i, j, grid, sp, k, rng):
        # The original generic implementation appends one candidate when k=0.
        return native_reroute(inst, sol, ev, i, j, grid, sp, k, rng) if k else []

    namespace["reroute_candidates"] = reroute_with_budget
    return SimpleNamespace(**{name: namespace[name] for name in
                              ("empty_sol", "add_flow", "check_feasible", "tabu_search")})


def run(instance_dict, initial_dense, seed, seconds, config):
    """Return the best flow, reported makespan and number of TSR iterations."""
    vendor = _load_tsr()
    inst = SimpleNamespace(name=instance_dict["name"], m=instance_dict["robots"],
                           n=len(instance_dict["requirements"]) - 2,
                           r=instance_dict["requirements"], c=instance_dict["processing"],
                           t=instance_dict["travel"])
    initial = vendor.empty_sol(inst.n)
    for i, row in enumerate(initial_dense):
        for j, amount in enumerate(row):
            if amount:
                vendor.add_flow(initial, i, j, amount)
    vendor.check_feasible(inst, initial)
    best, evaluation, iterations = vendor.tabu_search(
        inst, None, False, seconds, seed, initial, config, beta=config["beta"],
        I1=config["restart_best"], I2=config["restart_new"],
    )
    size = inst.n + 2
    flow = [[0] * size for _ in range(size)]
    for i, arcs in enumerate(best[0]):
        for j, amount in arcs.items():
            flow[i][j] = int(amount)
    return {"flow": flow, "iterations": iterations, "reported": float(evaluation.cmax)}
