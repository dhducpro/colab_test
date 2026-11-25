"""Run the original PDPCR pilot TSR with a shared start and CPU deadline."""
from time import process_time
from unittest.mock import patch

if __package__:
    from .vendor.duc.pdpcr import core, search
else:
    from vendor.duc.pdpcr import core, search


def greedy(instance_dict):
    instance = core.Instance(**instance_dict)
    return core.greedy(instance).tolist()


def run(instance_dict, initial_dense, seed, seconds, config):
    instance = core.Instance(**instance_dict)
    # These five sizes are fixed by the original pilot code (tiny_min_candidates=False).
    # Check the requested values before claiming the comparison uses shared parameters.
    original_sizes = {
        "n_reroute": instance.n // 5,
        "n_reverse": instance.n // 15,
        "n_redistribute": instance.n // 15,
        "tenure_arc": instance.n // 8,
        "tenure_cut": instance.n // 15,
    }
    for name, size in original_sizes.items():
        if config[name] != size:
            raise ValueError(f"Đức TSR fixes {name} at {size} for {instance.n} tasks")

    flow = core.checked_flow(instance, initial_dense)
    # The runner executes each solver in its own process. All deadline checks in
    # this implementation, including those inside the neighborhoods, use this clock.
    with patch.object(search, "perf_counter", process_time):
        result = search.tabu_search(
            instance,
            iterations=10**9,
            seed=seed,
            beta=config["beta"],
            tiny_min_candidates=False,
            initial=flow,
            restart_best=config["restart_best"],
            restart_new=config["restart_new"],
            time_limit=seconds,
        )
    return {
        "flow": result["flow"].tolist(),
        "iterations": result["iterations_completed"],
        "reported": float(result["objective"]),
    }
