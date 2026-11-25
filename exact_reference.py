"""Compute CPLEX bounds from the existing PDP-CR LP and greedy MIP start."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import check_instance
from pdpcr.core import Instance, decompose_routes, evaluate, replay_routes
from pdpcr.cplex_cli import solve_cplex, write_model


def reference(path, out, seconds, executable=None):
    data = (check_instance.read_instance(path, 10) if path.suffix == ".csv"
            else json.loads(path.read_text(encoding="utf-8")))
    instance = Instance(**data)
    if executable:
        result = solve_cplex(instance, out, seconds, executable=executable)
        result["incumbent"] = result["objective"]
        result["optimal"] = result["status"] == "optimal"
    else:
        result = solve_python_cplex(instance, out, seconds)
    (out / "reference.json").write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")
    return result


def solve_python_cplex(instance, out, seconds):
    try:
        import cplex
        from cplex.exceptions import CplexSolverError
    except ImportError as error:
        raise RuntimeError("Set --cplex-exe to a licensed CPLEX executable or install the licensed Python API.") from error

    write_model(instance, out)
    with (out / "cplex.log").open("w", encoding="utf-8") as log:
        with cplex.Cplex() as model:
            for set_stream in (model.set_log_stream, model.set_results_stream,
                               model.set_warning_stream, model.set_error_stream):
                set_stream(log)
            model.read(str(out / "model.lp"))
            model.MIP_starts.read(str(out / "start.mst"))
            model.parameters.timelimit.set(seconds)
            model.parameters.threads.set(1)
            model.parameters.mip.tolerances.mipgap.set(0)
            model.parameters.mip.tolerances.absmipgap.set(0)
            try:
                model.solve()
            except CplexSolverError as error:
                if "1016" in str(error):
                    raise RuntimeError("This 50-task model exceeds the Community Edition size limit. "
                                       "Run with licensed IBM ILOG CPLEX.") from error
                raise
            status = model.solution.get_status()
            bound = model.solution.MIP.get_best_objective()
            if not math.isfinite(bound) or abs(bound) >= cplex.infinity:
                bound = None
            result = dict(instance=instance.name, status=model.solution.get_status_string(),
                          lower_bound=bound, incumbent=None, optimal=False, seconds=seconds)
            if model.solution.is_primal_feasible():
                flow = np.zeros((instance.size, instance.size), dtype=int)
                for name, value in zip(model.variables.get_names(), model.solution.get_values()):
                    if name.startswith("x_"):
                        _, i, j = name.split("_")
                        if abs(value - round(value)) > 1e-6:
                            raise ValueError("Noninteger CPLEX robot flow.")
                        flow[int(i), int(j)] = round(value)
                score = check_instance.evaluate(vars(instance), flow.tolist())
                replay = replay_routes(instance, decompose_routes(instance, flow), evaluate(instance, flow).starts)
                if (abs(score - replay["actual_return_max"]) > 1e-6
                        or (bound is not None and bound > score + 1e-6)):
                    raise ValueError("CPLEX incumbent or bound failed validation.")
                result.update(incumbent=score, optimal=status == model.solution.status.MIP_optimal,
                              flow=flow.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--runs", type=Path, help="Optional runs.csv for reporting gaps")
    parser.add_argument("--cplex-exe", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    bounds = {}
    for path in args.instances:
        result = reference(path, args.out / path.stem, args.seconds, args.cplex_exe)
        bounds[result["instance"]] = result["lower_bound"]
        print({key: result[key] for key in ("instance", "status", "lower_bound", "incumbent", "optimal")},
              flush=True)
    if args.runs:
        with args.runs.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row["instance"] in bounds]
            fields = reader.fieldnames + ["cplex_lower_bound", "gap_to_bound_percent"]
        if rows:
            with (args.out / "gaps.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    bound = bounds[row["instance"]]
                    writer.writerow(dict(row, cplex_lower_bound=bound,
                                         gap_to_bound_percent=100 * (float(row["makespan"]) - bound) / bound
                                         if bound is not None and bound > 0 else None))


if __name__ == "__main__":
    main()
