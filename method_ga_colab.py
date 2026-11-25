"""Run the project's pure GA v2 using the TSR campaign's CPU-time convention."""
from contextlib import ExitStack
from time import process_time
from unittest.mock import patch

from pdpcr import core, flow_operators, genetic, search


GA_SETTINGS = {
    "population_size": 16,
    "generations": 10**9,
    "crossover_rate": 0.9,
    "mutation_rate": 0.4,
    "tournament_size": 3,
    "duplicate_limit": 1,
    "crossover_method": "cycle",
    "mutation_beta": 0.05,
    "mutation_strategy": "mixed",
    "local_search_steps": 0,
    "intensification_probability": 0.0,
}


def run(instance_dict, initial_dense, seed, seconds, config):
    """Insert the shared flow as the first individual of the initial population."""
    instance = core.Instance(**instance_dict)
    with ExitStack() as clock:
        for module in (genetic, flow_operators, search):
            clock.enter_context(patch.object(module, "perf_counter", process_time))
        result = genetic.genetic_search(
            instance, seed=seed, time_limit=seconds, initial=initial_dense, **GA_SETTINGS
        )
    return {
        "flow": result["flow"].tolist(),
        "reported": float(result["objective"]),
        "iterations": result["generations_completed"],
        "diagnostics": {
            "generations_completed": result["generations_completed"],
            "offspring_completed": result["offspring_completed"],
            "partial_generation_offspring": result["partial_generation_offspring"],
            "population_size": result["config"]["population_size"],
            "initialization_cpu_seconds": result["statistics"]["initialization_seconds"],
            "variant": result["variant"],
        },
    }
