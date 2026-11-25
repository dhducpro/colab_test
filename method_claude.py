"""Run the generic TSR in Claude's preliminary package on the common instance."""

from functools import lru_cache
from pathlib import Path
import runpy
import time
from types import SimpleNamespace


from pdpcr.schedule import longest_paths
from pdpcr import tsr_improved


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
    # Preserve Claude's FIFO order and critical-arc ordering on the shared kernel.
    def evaluate(inst, sol, eps=1e-9):
        out, incoming = sol
        paths = longest_paths(out, [len(row) for row in incoming], inst.c, inst.t, fifo=True)
        if paths is None:
            return None
        ev = namespace["Eval"]()
        ev.order, ev.z, ev.tau = paths
        ev.pos = [0] * len(out)
        ev.cmax = ev.z[-1]
        ev.crit_arcs = []
        for k, i in enumerate(ev.order):
            ev.pos[i] = k
            finish = ev.z[i] + inst.c[i]
            ev.crit_arcs.extend((i, j) for j in out[i]
                                if abs(finish + inst.t[i][j] + ev.tau[j] - ev.cmax) <= eps)
        ev.crit = len(ev.crit_arcs)
        return ev

    namespace["evaluate"] = evaluate
    namespace["time"] = SimpleNamespace(time=time.process_time)
    native_reroute = namespace["reroute_candidates"]

    def reroute_with_budget(inst, sol, ev, i, j, grid, sp, k, rng):
        # The original generic implementation appends one candidate when k=0.
        return native_reroute(inst, sol, ev, i, j, grid, sp, k, rng) if k else []

    namespace["reroute_candidates"] = reroute_with_budget
    return SimpleNamespace(**{name: namespace[name] for name in (
        "empty_sol", "add_flow", "check_feasible", "tabu_search", "evaluate", "reroute",
        "reverse", "draw_cuts", "redistribute", "new_construction", "copy_sol",
    )})


def prepare_sparse(instance_dict, initial_dense):
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
    return inst, initial, vendor


def dense_flow(sparse):
    size = len(sparse[0])
    flow = [[0] * size for _ in range(size)]
    for i, arcs in enumerate(sparse[0]):
        for j, amount in arcs.items():
            flow[i][j] = int(amount)
    return flow


def run(instance_dict, initial_dense, seed, seconds, config, *, improved=False, relocate=False):
    """Return the best flow, reported makespan and number of TSR iterations."""
    inst, initial, vendor = prepare_sparse(instance_dict, initial_dense)
    diagnostics = {}
    if improved:
        best, evaluation, iterations, trace, diagnostics = tsr_improved.tabu_search(
            inst, initial, vendor, seconds, seed, config, relocate=relocate
        )
    else:
        trace = []
        best, evaluation, iterations = vendor.tabu_search(
            inst, None, False, seconds, seed, initial, config, beta=config["beta"],
            I1=config["restart_best"], I2=config["restart_new"], trace=trace,
        )
    return {"flow": dense_flow(best), "iterations": iterations, "reported": float(evaluation.cmax),
            "history": trace, "diagnostics": diagnostics}
