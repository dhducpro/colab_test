"""Adapter for Thao's writing_paper TabuSearch, commit c85c247.

The upstream code uses node -1 for the returning depot. The benchmark uses
the last nonnegative index. Both representations use the supplied travel and
processing times. Neighborhood generation, admissibility, and restarts use
the upstream implementation, including its original tabu-list updates.
"""

from functools import lru_cache
import math
from pathlib import Path
import random
import sys
from time import process_time
from types import ModuleType
from unittest.mock import patch

import networkx as nx
import numpy as np


VENDOR = Path(__file__).resolve().parent / "vendor" / "thao"


class _TimeLimit(Exception):
    pass


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream search source at {old!r}")
    return source.replace(old, new)


def _instrument(source):
    # Keep the dependency optional for the four separate integer-programming
    # methods. The tabu search and its randomized restart do not use them.
    source = _replace_once(source, "from docplex.mp.model import Model",
                           "def Model(*args, **kwargs):\n"
                           "    from docplex.mp.model import Model as docplex_model\n"
                           "    return docplex_model(*args, **kwargs)")
    source = source.replace("np.infty", "np.inf")  # NumPy 2
    # Supply the common initial graph. Everything below its upstream conversion
    # to a flow graph uses TabuSearch.search() itself.
    source = _replace_once(source, "init_sol = self._init_solution(total_robots)",
                           "init_sol = self.common_initial")
    source = _replace_once(source, "for i in range(max_iters):",
                           "for i in range(max_iters):\n"
                           "            self.check_budget()")
    source = _replace_once(source, "for neighbor in neighbors:\n                if neighbor in affected_add",
                           "for neighbor in neighbors:\n                self.check_budget()\n"
                           "                if neighbor in affected_add")
    source = _replace_once(source, "best_solution_fitness = best_neighbor_fitness",
                           "best_solution_fitness = best_neighbor_fitness\n"
                           "                self.checkpoint(best_solution, best_solution_fitness)")
    source = _replace_once(source, "best_solution_fitness = cur_best_makespan",
                           "best_solution_fitness = cur_best_makespan\n"
                           "                    self.checkpoint(best_solution, best_solution_fitness)")
    source = _replace_once(source, "for i in range(cfg.VIET_RANDOM_ITE):\n                    solve_viet",
                           "for i in range(cfg.VIET_RANDOM_ITE):\n                    self.check_budget()\n"
                           "                    solve_viet")
    source = _replace_once(source, "cur_fitnesses.append(best_neighbor_fitness)",
                           "cur_fitnesses.append(best_neighbor_fitness)\n"
                           "            self.completed += 1")
    # Protect an upstream fallback on a tied redistribute move when all generated
    # arc candidates were rejected as tabu. No admissible arc move exists then.
    source = _replace_once(source, "if beta > cfg.TABU_REDISTRIBUTE_PROB and affected_add:",
                           "if beta > cfg.TABU_REDISTRIBUTE_PROB and affected_add and best_neighbor_0 is None:\n"
                           "                    self.diagnostics['tie_guard_uses'] += 1\n"
                           "                if beta > cfg.TABU_REDISTRIBUTE_PROB and best_neighbor_0 is not None:")
    # The sampled cut is a NumPy scalar. Adding it with += leaves the list empty.
    source = _replace_once(source, "tl_redistribute += affected_redistribute[best_neighbor]",
                           "tl_redistribute.append(int(affected_redistribute[best_neighbor]))\n"
                           "                self.diagnostics['accepted_redistribute'] += 1")
    source = _replace_once(source,
                           "[tl_redistribute.pop(i) for i in range(len(tl_redistribute) - cfg.TABU_TL_REDISTRIBUTE_SIZE)]",
                           "del tl_redistribute[:len(tl_redistribute) - cfg.TABU_TL_REDISTRIBUTE_SIZE]\n"
                           "            self.diagnostics['max_cut_tabu_length'] = max(\n"
                           "                self.diagnostics['max_cut_tabu_length'], len(tl_redistribute))")
    # The source divides by the requested candidate counts in a logging-only
    # f-string; zero candidates is a valid benchmark control.
    source = source.replace("/cfg.TABU_NUM_REROUTE_NEIGHBORS}", "/max(1, cfg.TABU_NUM_REROUTE_NEIGHBORS)}")
    source = source.replace("/cfg.TABU_NUM_REVERSE_NEIGHBORS}", "/max(1, cfg.TABU_NUM_REVERSE_NEIGHBORS)}")
    return source


@lru_cache(maxsize=None)
def _compile_module(name):
    path = VENDOR / f"{name}.py"
    source = path.read_text(encoding="utf-8-sig")
    if name == "heuristic":
        source = _instrument(source)
    return compile(source, str(path), "exec")


def _load_modules(config, seed):
    modules = {}
    # Run the vintage stand-alone modules with their own config and relative
    # dependencies. Avoid shadowing the project's modules after this load.
    with patch.dict(sys.modules):
        for name in ("config", "problem_variables", "helpers", "heuristic"):
            path = VENDOR / f"{name}.py"
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            exec(_compile_module(name), module.__dict__)
            if name == "config":
                module.RND_SEED = seed
                module.TABU_REDISTRIBUTE_PROB = config["beta"]
                module.TABU_FIND_FROM_ELITE_AFTER = config["restart_best"]
                module.TABU_RESTART_AFTER = config["restart_new"]
                module.TABU_NUM_REROUTE_NEIGHBORS = config["n_reroute"]
                module.TABU_NUM_REVERSE_NEIGHBORS = config["n_reverse"]
                module.TABU_NUM_REDISTRIBUTE_NEIGHBORS = config["n_redistribute"]
                module.TABU_MAX_TL_ADD_SIZE = module.TABU_MAX_TL_DROP_SIZE = config["tenure_arc"]
                module.TABU_TL_REDISTRIBUTE_SIZE = config["tenure_cut"]
                # These upstream function defaults are bound when heuristic.py
                # is imported. Apply the three-neighborhood notebook settings here.
                module.TABU_REROUTE_ON_CRITICAL = config.get("reroute_on_critical", True)
                module.TABU_REVERSE_ON_CRITICAL = config.get("reverse_on_critical", True)
                module.TABU_REROUTE_LONGEST_EDGE = config.get("reroute_longest_edge", True)
                module.TABU_LOG_PARAMETERES = False
            modules[name] = module
    return modules["problem_variables"], modules["heuristic"]


def run(instance_dict, initial_dense, seed, seconds, config):
    variables, heuristic = _load_modules(config, seed)
    random.seed(seed)
    np.random.seed(seed)
    travel = np.asarray(instance_dict["travel"], dtype=float)
    processing = np.asarray(instance_dict["processing"], dtype=float)
    requirements = instance_dict["requirements"]
    size = len(processing)
    sink = size - 1
    ids = range(sink)
    # The original VGeneralizedSolver tracks a separate geometric diagnostic
    # when appending to a Path. Its scheduling decisions use the matrices below.
    placeholder = np.zeros(2)
    tasks = [variables.Task(placeholder, placeholder, processing[i], int(requirements[i]), i)
             for i in ids]
    processed = variables.Processed(tasks[0], tasks[1:], None, None)
    c_operate = {i: processing[i] for i in ids}
    c_travel = {i: {j: travel[i, sink] if j == 0 else travel[i, j] for j in ids}
                for i in ids}
    search = heuristic.TabuSearch(processed, c_travel, c_operate)
    graph = nx.DiGraph()
    graph.add_nodes_from([*ids, -1])
    for i, row in enumerate(initial_dense):
        for j, weight in enumerate(row):
            if weight:
                graph.add_edge(-1 if i == sink else i, -1 if j == sink else j, weight=int(weight))
    best_graph = graph
    best_score = search._get_fitness(graph)
    if not math.isfinite(best_score):
        raise ValueError("The initial flow does not reach the returning depot.")
    deadline = process_time() + seconds

    def check_budget():
        if process_time() >= deadline:
            raise _TimeLimit

    def checkpoint(candidate, score):
        nonlocal best_graph, best_score
        check_budget()
        best_graph, best_score = candidate, score

    search.common_initial = graph
    search.completed = 0
    search.diagnostics = {"tie_guard_uses": 0, "accepted_redistribute": 0, "max_cut_tabu_length": 0}
    search.check_budget = check_budget
    search.checkpoint = checkpoint
    try:
        best_graph, best_score, _, _ = search.search(instance_dict["robots"], 10**9, config["tenure_arc"])
    except _TimeLimit:
        pass
    flow = [[0] * size for _ in range(size)]
    for i, j, data in best_graph.edges(data=True):
        flow[sink if i == -1 else i][sink if j == -1 else j] = int(data["weight"])
    return {"flow": flow, "reported": float(best_score), "iterations": search.completed,
            "diagnostics": search.diagnostics}
