"""Run one solver on Thao's lr108 instance and save its checked robot flow."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter, process_time

from check_instance import evaluate, read_instance
from method_duc import greedy


DATA = Path(__file__).resolve().parent / "data" / "lr108-5-uniform.csv"
ROBOTS = 10
TSR_SETTINGS = {
    "n_reroute": 10, "n_reverse": 3, "n_redistribute": 3,
    "tenure_arc": 6, "tenure_cut": 3, "beta": 0.3,
    "restart_best": 300, "restart_new": 600,
}
TOLERANCE = 1e-7


def run_one(method, seed, seconds):
    if method == "ga":
        from method_ga_colab import GA_SETTINGS as settings, run
    elif method == "thao":
        from method_thao import run
        settings = TSR_SETTINGS
    elif method == "duc":
        from method_duc import run
        settings = TSR_SETTINGS
    elif method == "claude":
        from method_claude import run
        settings = TSR_SETTINGS
    else:
        raise ValueError(f"Unknown solver {method!r}")

    instance = read_instance(DATA, ROBOTS)
    initial = greedy(instance)
    initial_cmax = evaluate(instance, initial)
    zero = run(instance, initial, seed, 0, settings)
    if zero["flow"] != initial or not math.isclose(
        zero["reported"], initial_cmax, rel_tol=0, abs_tol=TOLERANCE
    ):
        raise ValueError(f"{method} did not reproduce the common starting flow and makespan")

    wall_start = perf_counter()
    cpu_start = process_time()
    result = run(instance, initial, seed, seconds, settings)
    cpu_used = process_time() - cpu_start
    wall_used = perf_counter() - wall_start

    score = evaluate(instance, result["flow"])
    if not math.isclose(result["reported"], score, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError(f"{method} reports {result['reported']}, independent evaluation gives {score}")
    if score > initial_cmax + TOLERANCE:
        raise ValueError(f"{method} returned a worse solution than its starting incumbent")

    diagnostics = result.get("diagnostics", {})
    if method == "thao" and diagnostics["accepted_redistribute"]:
        if not 1 <= diagnostics["max_cut_tabu_length"] <= TSR_SETTINGS["tenure_cut"]:
            raise ValueError("Thao cut tabu list is not using the configured capacity")
    return {
        "method": method, "seed": seed, "cpu_budget_s": seconds,
        "cpu_s": cpu_used, "wall_s": wall_used,
        "initial_cmax": initial_cmax, "makespan": score, "feasible": True,
        "tsr_iterations": result["iterations"] if method != "ga" else None,
        "ga_generations": diagnostics.get("generations_completed"),
        "ga_offspring": diagnostics.get("offspring_completed"),
        "flow": result["flow"], "settings": settings, "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("thao", "duc", "claude", "ga"))
    parser.add_argument("seed", type=int)
    parser.add_argument("seconds", type=float)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")

    record = run_one(args.method, args.seed, args.seconds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output_json)
