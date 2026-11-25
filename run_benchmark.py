"""Compare three TSR implementations and GA v2 on Thao's lr108 instance."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from importlib.metadata import version

from colab_worker import DATA, ROBOTS, TSR_SETTINGS
from method_ga_colab import GA_SETTINGS


ROOT = Path(__file__).resolve().parent
METHODS = {"thao": "TSR Thao", "duc": "TSR Duc", "claude": "TSR Claude", "ga": "GA v2"}


def write_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def benchmark(seeds, seconds, workers, output):
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    sources = sorted([*ROOT.glob("*.py"), *(ROOT / "pdpcr").rglob("*.py"), *(ROOT / "vendor").rglob("*.py")])
    config = {
        "instance": DATA.name, "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "tasks": 50, "robots": ROBOTS, "seeds": seeds, "cpu_seconds_per_run": seconds,
        "workers": workers, "schedule": "parallel seeds within each method",
        "tsr": TSR_SETTINGS, "ga": GA_SETTINGS, "python": platform.python_version(),
        "platform": platform.platform(), "numpy": version("numpy"),
        "networkx": version("networkx"), "pandas": version("pandas"),
        "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }
    manifest = output / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing results in {output} use a different experiment configuration.")
    else:
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temporary.replace(manifest)

    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    def run_seed(method, seed):
        path = output / f"{method}_{seed}.json"
        reused = path.exists()
        if not reused:
            log = logs / f"{method}_{seed}.log"
            with log.open("wb") as stream:
                worker = subprocess.run(
                    [sys.executable, "-X", "faulthandler", str(ROOT / "colab_worker.py"),
                     method, str(seed), str(seconds), str(path)],
                    env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
            if worker.returncode:
                with log.open(encoding="utf-8", errors="replace") as stream:
                    details = "".join(deque(stream, maxlen=50)).strip()
                if not details and worker.returncode < 0:
                    details = f"Worker was terminated by signal {-worker.returncode}."
                raise RuntimeError(f"{METHODS[method]}, seed {seed}: exit {worker.returncode}.\n"
                                   f"{details}\nFull log: {log}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["method"], record["seed"], record["cpu_budget_s"], record["feasible"]) != (method, seed, seconds, True):
            raise ValueError(f"Unexpected result in {path}")
        row = {key: value for key, value in record.items() if key not in ("flow", "settings", "diagnostics")}
        row["method"] = METHODS[method]
        return row, reused

    rows = {}
    order = [(method, seed) for method in METHODS for seed in seeds]
    with ThreadPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
        for method, label in METHODS.items():
            failures = []
            print(f"{label}: starting seeds {seeds}, up to {min(workers, len(seeds))} at once", flush=True)
            pending = {executor.submit(run_seed, method, seed): seed for seed in seeds}
            for result in as_completed(pending):
                seed = pending[result]
                try:
                    row, reused = result.result()
                except (OSError, RuntimeError, ValueError) as exc:
                    failures.append(seed)
                    print(f"ERROR at {label}, seed {seed}\n{exc}", file=sys.stderr, flush=True)
                    continue
                rows[method, seed] = row
                write_csv(output / "runs.csv", [rows[run] for run in order if run in rows])
                cached = " (saved result)" if reused else ""
                print(f"  {label}, seed {seed}: makespan {row['makespan']:.4f} | {row['cpu_s']:.2f} CPU s",
                      cached, flush=True)
            if failures:
                raise SystemExit(f"{label}: failed seeds {sorted(failures)}. Completed results are in {output}.")
    rows = [rows[run] for run in order]

    summary = []
    for label in METHODS.values():
        group = [row for row in rows if row["method"] == label]
        scores = [row["makespan"] for row in group]
        best = min(group, key=lambda row: row["makespan"])
        summary.append({"method": label, "seeds": len(scores), "best_makespan": best["makespan"],
                        "best_seed": best["seed"],
                        "mean_makespan": statistics.mean(scores),
                        "sd_makespan": statistics.stdev(scores) if len(scores) > 1 else None})
    write_csv(output / "summary.csv", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("--seeds must be distinct integers in [0, 2**32)")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    benchmark(args.seeds, args.seconds, args.workers, args.output.resolve())
