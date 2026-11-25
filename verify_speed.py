"""Check fixed-work equivalence against the earlier Colab code and measure CPU throughput."""
import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile, copytree
import subprocess
import sys
from tempfile import TemporaryDirectory


def measure(paths, seconds, fixed):
    from time import process_time
    from unittest.mock import patch

    from check_instance import read_instance
    from pdpcr import core, flow_operators, genetic, redistribute, search

    records = {}

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    for path in paths:
        instance = core.Instance(**(read_instance(path, 10) if path.suffix == ".csv"
                                     else json.loads(path.read_text(encoding="utf-8"))))
        if fixed:
            checks = []
            for seed in (0, 1):
                ts = search.tabu_search(instance, iterations=40, seed=seed, time_limit=10**6,
                                        beta=0.3, tiny_min_candidates=False)
                ga = genetic.genetic_search(instance, seed=seed, generations=2, time_limit=10**6)
                checks.append({
                    "tsr": [(s["iteration"], s["current"], s["best"], s["move"]) for s in ts["history"]],
                    "tsr_flow": ts["flow"].tolist(), "tsr_generated": ts["generated"],
                    "ga_flow": ga["flow"].tolist(), "ga_objectives": ga["population_objectives"],
                    "ga_history": [(s["generation"], s["offspring"], s["best"]) for s in ga["history"]],
                    "ga_stats": {k: v for k, v in ga["statistics"].items() if "seconds" not in k},
                })
            initial = core.greedy(instance, seed=3, randomize=True)
            cuts = [redistribute.redistribute_cut(instance, initial, cut)[0].tolist()
                    for cut in range(1, instance.size)]
            records[instance.name] = {"fixed_work_sha256": digest(checks + [cuts])}
        else:
            with (patch.object(search, "perf_counter", process_time),
                  patch.object(genetic, "perf_counter", process_time),
                  patch.object(flow_operators, "perf_counter", process_time)):
                ts = search.tabu_search(instance, iterations=10**9, seed=0, time_limit=seconds,
                                        beta=0.3, tiny_min_candidates=False)
                ga = genetic.genetic_search(instance, seed=0, generations=10**9, time_limit=seconds)
            records[instance.name] = dict(tsr_iterations=ts["iterations_completed"],
                                          ga_offspring=ga["offspring_completed"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="+", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--out", type=Path, default=Path("performance.json"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--stage", choices=("fixed", "timed"))
    args = parser.parse_args()
    if args.stage:
        print(json.dumps(measure(args.instances, args.seconds, args.stage == "fixed")))
        return
    if args.original is None:
        parser.error("--original must point to the earlier Colab checkout")
    script = Path(__file__).resolve()
    inputs = [str(path.resolve()) for path in args.instances]

    def stage(source, mode):
        # -c puts the requested checkout before the study folder on sys.path.
        bootstrap = ("import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                     "runpy.run_path(sys.argv.pop(1), run_name='__main__')")
        raw = subprocess.check_output([sys.executable, "-c", bootstrap, str(source), str(script),
                                       "--stage", mode, "--seconds", str(args.seconds), *inputs],
                                      text=True)
        return json.loads(raw)

    old, new = args.original.resolve(), script.parent
    with TemporaryDirectory(dir=Path.cwd(), prefix="audit_") as tmp:
        baseline, patched = Path(tmp) / "before", Path(tmp) / "after_patch"
        for folder in (baseline, patched):
            copytree(old / "pdpcr", folder / "pdpcr")
            copyfile(old / "check_instance.py", folder / "check_instance.py")
        subprocess.run(["git", "apply", str(new / "performance/pdpcr_speed_patch.diff")],
                       cwd=patched, check=True)
        for folder in (baseline, patched):
            core = folder / "pdpcr/core.py"
            # The older constructor rejected zero-duration tasks present in Li–Lim.
            # Relax the validation in both audit copies without changing search.
            core.write_text(core.read_text(encoding="utf-8").replace(
                "c <= 0 for c in self.processing[1:-1]", "c < 0 for c in self.processing[1:-1]"),
                encoding="utf-8")
        before, patch, after = (stage(folder, "fixed") for folder in (baseline, patched, new))
        if before != patch or patch != after:
            raise RuntimeError(f"Fixed-work equivalence failed:\n{before}\n{patch}\n{after}")
        result = {"fixed_work": after, "cpu_seconds_per_method": args.seconds,
                  "before": stage(baseline, "timed"), "official_patch": stage(patched, "timed"),
                  "shared_kernel": stage(new, "timed")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
