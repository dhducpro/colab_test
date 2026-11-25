"""Run and report the matched PDP-CR benchmark from Colab."""

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from statistics import mean


INSTANCE_NAME = "lr108-5-uniform"
BASELINE = "thao_original"
INSTANCE_SHA256 = "e3da6196bc21af771d93bfadf8f3ea7ad87c0e9b690ef47c6cc028ce96576343"
PACKAGE_NAMES = ("numpy", "pandas", "networkx", "scipy")
HEARTBEAT_SECONDS = 30


def child_environment(source):
    env = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    env["PYTHONUTF8"] = "1"

    installed = {}
    for package in (*PACKAGE_NAMES, "python-dateutil"):
        with suppress(importlib.metadata.PackageNotFoundError):
            installed[package] = importlib.metadata.version(package)
    pinned = {"numpy": "2.2.6", "pandas": "2.2.3", "networkx": "3.6.1"}
    has_pins = all(installed.get(package) == version for package, version in pinned.items())
    scipy_version = installed.get("scipy", "0.0").split(".")
    dateutil_version = installed.get("python-dateutil", "0.0").split(".")
    compatible = (
        has_pins and scipy_version[0] == "1" and scipy_version[1].isdigit()
        and int(scipy_version[1]) >= 15 and dateutil_version[0] == "2"
        and dateutil_version[1].isdigit() and int(dateutil_version[1]) >= 9
    )
    if not compatible:
        packages = source / f"packages_{sys.version_info.major}_{sys.version_info.minor}"
        stamp = packages / ".installed"
        if not stamp.is_file():
            print("Đang cài thư viện cho benchmark...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                 "--upgrade", "--target", str(packages), "-r", str(source / "requirements.txt")],
                check=True,
            )
            stamp.touch()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(packages), env.get("PYTHONPATH")))
        )
    return env


def show_finished(out, methods, seeds, already_printed):
    counts = dict.fromkeys(methods, 0)
    for method in methods:
        for seed in seeds:
            path = out / f"{INSTANCE_NAME}__{method}__{seed}.json"
            if not path.is_file():
                continue
            counts[method] += 1
            if (method, seed) not in already_printed:
                record = json.loads(path.read_text(encoding="utf-8"))
                print(
                    f"{method}, seed {seed}  |  makespan {record['makespan']:.4f}"
                    f"  |  {record['cpu_s']:.2f} giây CPU",
                    flush=True,
                )
                already_printed.add((method, seed))
    return counts


def run_benchmark(source, instance, out, env, methods, seconds, seeds):
    command = [
        sys.executable, "-u", "-B", "run_hung_study.py",
        "--instances", str(instance), "--methods", *methods,
        "--seconds", str(seconds), "--workers", str(len(seeds)),
        "--seeds", *map(str, seeds), "--out", str(out),
    ]
    log_path = out / "benchmark.log"
    seen = set()
    show_finished(out, methods, seeds, seen)
    print(
        f"Thứ tự chạy {methods[0]} → {methods[1]}. "
        f"Mỗi nhóm có {len(seeds)} seed chạy song song.",
        flush=True,
    )
    started = time.monotonic()
    next_report = started + HEARTBEAT_SECONDS
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            while proc.poll() is None:
                time.sleep(2)
                counts = show_finished(out, methods, seeds, seen)
                now = time.monotonic()
                if now >= next_report and proc.poll() is None:
                    progress = " | ".join(f"{m} {counts[m]}/{len(seeds)}" for m in methods)
                    print(f"Đang chạy ({(now - started) / 60:.1f} phút thực). {progress}.",
                          flush=True)
                    next_report = now + HEARTBEAT_SECONDS
        except KeyboardInterrupt:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            proc.wait()
            raise
    show_finished(out, methods, seeds, seen)
    if proc.returncode != 0:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 12000))
            print(log.read().decode("utf-8", errors="replace"), flush=True)
        raise RuntimeError(f"Benchmark dừng với mã lỗi {proc.returncode}. Nhật ký {log_path}")


def report_results(out, seeds, ts_method):
    with (out / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = {(row["method"], int(row["seed"])): row for row in csv.DictReader(handle)}
    pairs = []
    tolerance = 1e-7
    print(f"\n{'Seed':>5} {'TSR gốc':>13} {'TS cải tiến':>15} {'Giảm (%)':>11}")
    for seed in seeds:
        original, proposed = rows[BASELINE, seed], rows[ts_method, seed]
        initial = float(original["initial"])
        if abs(initial - float(proposed["initial"])) > tolerance:
            raise RuntimeError(f"Hai thuật toán dùng nghiệm đầu khác nhau ở seed {seed}.")
        ref, result = float(original["makespan"]), float(proposed["makespan"])
        delta = ref - result
        percent = 100 * delta / ref if ref else 0.0
        pairs.append({
            "seed": seed, "initial": initial, "thao_original": ref, ts_method: result,
            "makespan_reduction": delta, "reduction_pct": percent,
        })
        print(f"{seed:>5} {ref:>13.4f} {result:>15.4f} {percent:>11.2f}")
    old_mean = mean(row["thao_original"] for row in pairs)
    new_mean = mean(row[ts_method] for row in pairs)
    wins = sum(row["makespan_reduction"] > tolerance for row in pairs)
    ties = sum(abs(row["makespan_reduction"]) <= tolerance for row in pairs)
    losses = len(pairs) - wins - ties
    pct = 100 * (old_mean - new_mean) / old_mean if old_mean else 0.0
    print(f"\nMakespan trung bình  |  Thảo {old_mean:.4f}  |  TS cải tiến {new_mean:.4f}")
    print(f"Mức giảm của makespan trung bình {pct:+.2f}%  |  Thắng {wins}, hòa {ties}, thua {losses}")
    with (out / "paired_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)


def run_colab(seconds, seeds, ts_method, output_root, revision):
    """Run both methods with the same start and return the result archive path."""
    if ts_method not in ("thao_elite", "claude_elite"):
        raise ValueError("TS_METHOD phải là thao_elite hoặc claude_elite.")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("TIME_LIMIT phải là một số dương hữu hạn.")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("SEEDS phải chứa các số nguyên từ 0 đến 2**32 - 1.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Các seed phải khác nhau.")

    source = Path(__file__).resolve().parent
    instance = source / "data" / f"{INSTANCE_NAME}.csv"
    if hashlib.sha256(instance.read_bytes()).hexdigest() != INSTANCE_SHA256:
        raise RuntimeError("Dữ liệu không khớp với instance benchmark.")
    env = child_environment(source)
    package_probe = (
        "import importlib.metadata as md,json;"
        f"print(json.dumps([md.version(name) for name in {PACKAGE_NAMES!r}]))"
    )
    versions = subprocess.check_output([sys.executable, "-c", package_probe], env=env, text=True).strip()
    settings = [revision, platform.platform(), platform.python_version(), versions, seconds, seeds, ts_method]
    tag = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:10]
    out = Path(output_root).resolve() / f"{INSTANCE_NAME}_{seconds:g}s_{ts_method}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Kết quả từng seed đã xong được ghi ở {out}", flush=True)

    methods = (BASELINE, ts_method)
    run_benchmark(source, instance, out, env, methods, seconds, seeds)
    report_results(out, seeds, ts_method)
    bundle = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    print(f"\nZIP kết quả  {bundle}", flush=True)
    return bundle
