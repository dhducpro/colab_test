"""Same-start CPU-budget benchmarks for the PDP-CR action plan."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import mean, stdev
from time import perf_counter, process_time
from unittest.mock import patch

import check_instance
import method_thao
import method_claude
from method_ga_colab import GA_SETTINGS
from pdpcr import core, flow_evolution, flow_operators, genetic, pilot_genetic, search
from run_benchmark import write_csv
from vendor.duc.pdpcr import search as duc_search

ROOT = Path(__file__).resolve().parent
PILOT_GA = dict(population_size=16, crossover_rate=0.9, mutation_rate=0.4,
                tournament_size=3, local_search_steps=2, cycle_trials=4)
CHECKPOINTS = (0.10, 0.25, 0.50, 1.00)
QUICK = ("thao", "duc", "claude", "ga_pure", "ga_pilot", "mls")
ABLATIONS = {
    "ga_no_crossover": {"crossover_rate": 0.0},
    "ga_single_cycle": {"single_random_cycle": True},
    "ga_no_order": {"mutation_strategy": "mixed"},
    "ga_no_descent": {"local_search_steps": 0},
    "ga_descent4": {"local_search_steps": 4},
    "ga_pop8": {"population_size": 8},
    "ga_pop32": {"population_size": 32},
    "ga_mutation02": {"mutation_rate": 0.2},
    "ga_mutation06": {"mutation_rate": 0.6},
}
EXTRA = (*ABLATIONS, "ga_cut", "ga_cut_mutation", "ga_reroute_mutation", "ga_cut_memetic",
         "mls_cut", "ms_tsr", "duc_beta05", "duc_beta07", "duc_restart150")
PRESETS = {"quick": QUICK, "pilot": ("duc", "claude", "ga_pure", "ga_pilot", "mls"),
           "calibration": ("duc", "claude", "duc_beta05", "duc_beta07", "duc_restart150"),
           "ablation": QUICK + EXTRA}


def load_instance(path):
    path = Path(path)
    if path.suffix == ".csv":
        return check_instance.read_instance(path, 10)
    return json.loads(path.read_text(encoding="utf-8"))


def ts_settings(n):
    return dict(n_reroute=max(1, n // 5), n_reverse=max(1, n // 15),
                n_redistribute=max(1, n // 15), tenure_arc=max(1, n // 8),
                tenure_cut=max(1, n // 15), beta=0.3, restart_best=300, restart_new=600)


def descent(instance, initial, seed, seconds, cut_only=False):
    rng = random.Random(seed)
    started = process_time()
    deadline = started + seconds
    best = genetic.individual(instance, initial)
    history = [(0.0, best.objective)]
    current = best
    restarts = evaluations = 0

    def update(candidate):
        nonlocal best
        if candidate.objective < best.objective:
            best = candidate
            history.append((process_time() - started, best.objective))

    while process_time() < deadline:
        if cut_only:
            current, stats = genetic.cut_intensify(
                instance, current, rng, max_nonimproving=instance.size,
                max_solves=10**9, deadline=deadline)
            evaluations += stats["solves"]
        else:
            current, stats = genetic.local_descent(
                instance, current, rng, max_steps=10**9, deadline=deadline, on_improvement=update)
            evaluations += sum(stats[name] for name in ("reroute", "reverse", "redistribute"))
        update(current)
        if process_time() < deadline:
            current = genetic.individual(instance, core.greedy(instance, rng.getrandbits(32), randomize=True))
            update(current)
            restarts += 1
    return {"flow": best.flow, "objective": best.objective, "history": history,
            "statistics": {"restarts": restarts, "neighbors": evaluations}}


def duc_tsr(instance, initial, seed, seconds, settings):
    return duc_search.tabu_search(
        instance, iterations=10**9, seed=seed, tiny_min_candidates=True, initial=initial,
        time_limit=seconds, beta=settings["beta"], restart_best=settings["restart_best"],
        restart_new=settings["restart_new"],
    )


def multi_start_tsr(instance, initial, seed, seconds, settings):
    rng = random.Random(seed)
    started = process_time()
    best = {"flow": initial, "objective": core.evaluate(instance, initial).makespan}
    history = [(0.0, best["objective"])]
    iterations = 0
    for restart in range(5):
        offset = process_time() - started
        piece = min(seconds / 5, seconds - offset)
        if piece <= 0:
            break
        result = duc_tsr(instance, initial, seed if not restart else rng.getrandbits(32), piece, settings)
        for step in result["history"]:
            if step["best"] < history[-1][1]:
                history.append((offset + step["seconds"], step["best"]))
        if result["objective"] < best["objective"]:
            best = result
        iterations += result["iterations_completed"]
        initial = core.greedy(instance, rng.getrandbits(32), randomize=True)
    return {**best, "history": history, "iterations_completed": iterations}


def solve(instance, initial, seed, seconds, method, settings):
    if method in ("thao", "claude"):
        implementation = method_thao if method == "thao" else method_claude
        result = implementation.run(vars(instance), initial.tolist(), seed, seconds, settings)
        return {**result, "objective": result["reported"], "iterations_completed": result["iterations"]}
    if method == "ms_tsr":
        return multi_start_tsr(instance, initial, seed, seconds, settings)
    if method.startswith("duc"):
        return duc_tsr(instance, initial, seed, seconds, settings)
    if method.startswith("mls"):
        return descent(instance, initial, seed, seconds, method == "mls_cut")
    if method == "ga_pilot" or method in ABLATIONS:
        options = {**PILOT_GA, **ABLATIONS.get(method, {})}
        return pilot_genetic.run_ga(instance, seed=seed, time_limit=seconds, initial=initial, **options)
    overrides = {"ga_cut": {"crossover_method": "cut"},
                 "ga_cut_mutation": {"mutation_strategy": "random_cut"},
                 "ga_reroute_mutation": {"mutation_strategy": "reroute"},
                 "ga_cut_memetic": {"intensification_probability": 0.2}}
    return genetic.genetic_search(instance, seed=seed, time_limit=seconds, initial=initial,
                                  **{**GA_SETTINGS, **overrides.get(method, {})})


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_seed(path, method, seed, seconds, destination, source):
    instance = core.Instance(**load_instance(path))
    initial = core.greedy(instance)
    starting = check_instance.evaluate(vars(instance), initial.tolist())
    settings = ts_settings(instance.n)
    settings.update({"duc_beta05": {"beta": 0.5}, "duc_beta07": {"beta": 0.7},
                     "duc_restart150": {"restart_best": 150, "restart_new": 300}}.get(method, {}))
    ga_settings = {
        "ga_pilot": PILOT_GA,
        **{name: {**PILOT_GA, **options} for name, options in ABLATIONS.items()},
    }.get(method)
    if method == "thao":
        import_module("pandas")  # Load the vendor module's dependency before timing its search.
        for module in ("config", "problem_variables", "helpers", "heuristic"):
            method_thao._compile_module(module)
    elif method == "claude":
        method_claude._load_tsr()
    with ExitStack() as stack:
        for module in (genetic, pilot_genetic, flow_operators, flow_evolution, search, duc_search):
            stack.enter_context(patch.object(module, "perf_counter", process_time))
        started_cpu, started_wall = process_time(), perf_counter()
        solution = solve(instance, initial, seed, seconds, method, settings)
        cpu, wall = process_time() - started_cpu, perf_counter() - started_wall
    flow = solution["flow"].tolist() if hasattr(solution["flow"], "tolist") else solution["flow"]
    score = check_instance.evaluate(vars(instance), flow)
    schedule = core.evaluate(instance, flow)
    routes = core.decompose_routes(instance, flow)
    replay = core.replay_routes(instance, routes, schedule.starts)
    if any(abs(value - score) > 1e-7 for value in (solution["objective"], replay["actual_return_max"])):
        raise ValueError(f"{method}: reported objective differs from independent evaluation and replay")
    if score > starting + 1e-7:
        raise ValueError(f"{method}: returned a worse incumbent than the common start")

    history = [(0.0, starting)]
    for point in solution["history"]:
        stamp, value = (point["seconds"], point["best"]) if isinstance(point, dict) else point
        if value < history[-1][1]:
            history.append((stamp, value))
    statistics = solution.get("statistics", solution.get("diagnostics", {}))
    attempted = statistics.get("cycle_exchanges_attempted", 0)
    record = dict(instance=instance.name, tasks=instance.n, robots=instance.robots,
                  data_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  source_sha256=source, method=method, seed=seed, budget=seconds,
                  cpu_s=cpu, wall_s=wall, initial=starting, makespan=score,
                  iterations=solution.get("iterations_completed"),
                  iterations_per_cpu_s=solution.get("iterations_completed", 0) / max(cpu, 1e-12),
                  generations=solution.get("generations_completed"),
                  offspring=solution.get("offspring_completed"),
                  exchange_acceptance=statistics.get("cycle_exchanges_accepted", 0) / attempted if attempted else None,
                  descent_cpu_share=(statistics.get("local_search_seconds", 0.0)
                                     + statistics.get("intensification_seconds", 0.0)) / max(cpu, 1e-12),
                  trace=history, flow=flow, routes=routes, tsr_settings=settings,
                  ga_settings=solution.get("config", ga_settings),
                  statistics=statistics, diversity=solution.get("population_history", []))
    write_json(destination, record)
    return record


def reports(out, records):
    keys = ("instance", "tasks", "robots", "method", "seed", "budget", "cpu_s", "wall_s",
            "initial", "makespan", "iterations", "iterations_per_cpu_s", "generations", "offspring",
            "exchange_acceptance", "descent_cpu_share")
    rows = [{k: run[k] for k in keys} for run in records]
    write_csv(out / "runs.csv", rows)
    anytime = []
    for run in records:
        for fraction in CHECKPOINTS:
            score = min(value for stamp, value in run["trace"] if stamp <= fraction * run["budget"])
            anytime.append(dict(instance=run["instance"], method=run["method"], seed=run["seed"],
                                fraction=fraction, makespan=score))
    write_csv(out / "anytime.csv", anytime)

    groups = {}
    for row in rows:
        groups.setdefault((row["instance"], row["method"]), []).append(row)
    summary = []
    for (instance, method), runs in groups.items():
        scores = [r["makespan"] for r in runs]
        average = lambda key: mean(r[key] for r in runs if r[key] is not None) if any(
            r[key] is not None for r in runs) else None
        summary.append(dict(instance=instance, method=method, seeds=len(scores), mean=mean(scores),
                            sd=stdev(scores) if len(scores) > 1 else None, best=min(scores),
                            mean_cpu_s=average("cpu_s"), iterations_per_cpu_s=average("iterations_per_cpu_s"),
                            exchange_acceptance=average("exchange_acceptance"),
                            descent_cpu_share=average("descent_cpu_share")))
    write_csv(out / "summary.csv", summary)
    ratios = []
    for instance in dict.fromkeys(row["instance"] for row in summary):
        speed = {row["method"]: row["iterations_per_cpu_s"] for row in summary if row["instance"] == instance}
        if speed.get("claude"):
            for method, rate in speed.items():
                if method.startswith("duc"):
                    ratios.append(dict(instance=instance, method=method, ratio_to_claude=rate / speed["claude"]))
    if ratios:
        write_csv(out / "tsr_throughput.csv", ratios)


def paired_tests(out):
    import csv
    from scipy.stats import wilcoxon
    with (out / "summary.csv").open(encoding="utf8", newline="") as handle:
        averages = {(r["instance"], r["method"]): float(r["mean"])
                    for r in csv.DictReader(handle)}
    tests = []
    for left in ("ga_pure", "ga_pilot"):
        for right in sorted({m for _, m in averages} - {left}):
            paired = [a - averages[i, right] for (i, m), a in sorted(averages.items())
                      if m == left and (i, right) in averages]
            if len(paired) >= 2:
                test = wilcoxon(paired, alternative="two-sided") if any(paired) else None
                tests.append(dict(method=left, baseline=right, instances=len(paired),
                                  mean_paired_difference=mean(paired),
                                  statistic=float(test.statistic) if test else 0.0,
                                  p_value=float(test.pvalue) if test else 1.0))
    if tests:
        write_csv(out / "wilcoxon.csv", tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="+", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=QUICK + EXTRA)
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seconds-by-size", action="store_true", help="Use 30/60/120 CPU s for 50/100/200 tasks")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=Path("hung_results"))
    args = parser.parse_args()
    args.methods = args.methods or list(PRESETS[args.preset])
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.workers < 1
            or len(set(args.seeds)) != len(args.seeds)
            or any(not 0 <= seed < 2**32 for seed in args.seeds)
            or len(set(args.methods)) != len(args.methods)):
        parser.error("Use positive limits and unique methods and seeds in [0, 2**32).")
    names = [load_instance(p)["name"] for p in args.instances]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in (".", "..") for n in names):
        parser.error("Instance names must be distinct filenames.")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted([*ROOT.glob("*.py"), *(ROOT / "pdpcr").rglob("*.py"), *(ROOT / "vendor").rglob("*.py")])
    fingerprint = hashlib.sha256()
    for path in sources:
        fingerprint.update(path.relative_to(ROOT).as_posix().encode())
        fingerprint.update(path.read_bytes())
    source = fingerprint.hexdigest()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    manifest = dict(code_sha256=source, methods=args.methods, seeds=args.seeds, workers=args.workers,
                    seconds=args.seconds, seconds_by_size=args.seconds_by_size,
                    python=platform.python_version(), platform=platform.platform(),
                    packages={name: version(name)
                              for name in ("numpy", "pandas", "networkx", "scipy")},
                    inputs={name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in zip(names, args.instances)})
    config = args.out / "config.json"
    if config.exists() and json.loads(config.read_text(encoding="utf8")) != manifest:
        parser.error("The existing output directory belongs to a different campaign.")
    write_json(config, manifest)
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for path, name in zip(args.instances, names):
            n = len(load_instance(path)["requirements"]) - 2
            seconds = {50: 30, 100: 60, 200: 120}.get(n, args.seconds) if args.seconds_by_size else args.seconds
            for method in args.methods:
                jobs = {}
                for seed in args.seeds:
                    key = (name, method, seed)
                    output = args.out / f"{name}__{method}__{seed}.json"
                    if output.exists():
                        records[key] = json.loads(output.read_text(encoding="utf-8"))
                    else:
                        jobs[workers.submit(run_seed, path, method, seed, seconds, output, source)] = key
                for job in as_completed(jobs):
                    key = jobs[job]
                    row = job.result()
                    records[key] = row
                    print(f"{key[0]} {key[1]} seed {key[2]}: {row['makespan']:.4f} ({row['cpu_s']:.2f} CPU s)", flush=True)
                reports(args.out, [records[k] for k in sorted(records)])
    paired_tests(args.out)


if __name__ == "__main__":
    main()
