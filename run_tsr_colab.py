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
from urllib.request import urlopen


INSTANCE_NAME = "lr108-5-uniform"
BASELINE = "thao_original"
INTEGRATED = "thao_integrated"
TWO_PHASE = "thao_elite"
METHOD_LABELS = {
    BASELINE: "TSR gốc",
    INTEGRATED: "Tích hợp từ đầu",
    TWO_PHASE: "TSR hai pha",
    "claude_elite": "Claude hai pha",
}
THAO_DATA_REVISION = "c85c247550f6232e6aec519b15824e283ee87883"
THAO_DATA_URL = (
    "https://raw.githubusercontent.com/thaonc97/thao_phd_model/"
    f"{THAO_DATA_REVISION}/data/lim/100"
)
N50_INSTANCES = ("lc101-5-uniform-n50", "lr101-5-uniform-n50", "lrc101-5-uniform-n50")
INSTANCE_SHA256 = {
    INSTANCE_NAME: "e3da6196bc21af771d93bfadf8f3ea7ad87c0e9b690ef47c6cc028ce96576343",
    "lc101-5-uniform-n50": "2f487d3ea94cd5fff90db7929988edd198b395481271634a6c8ebb535fd3aed6",
    "lr101-5-uniform-n50": "6ebb2c928910558aec73ee1f220cf3ef259f5dcd0c7a65d23f086147ac90108d",
    "lrc101-5-uniform-n50": "76c120e496c6c6d2f7ca5f9cf93ba95c4e0d2bcc97b29c39fb4c03eaab4a6a90",
}
PACKAGE_NAMES = ("numpy", "pandas", "networkx", "scipy")
HEARTBEAT_SECONDS = 30


def instance_path(source, name):
    path = source / "data" / f"{name}.csv"
    remote = name in N50_INSTANCES and not path.is_file()
    if remote:
        original_name = name.removesuffix("-n50")
        print(f"Đang tải dữ liệu {original_name} từ GitHub của Thảo...", flush=True)
        with urlopen(f"{THAO_DATA_URL}/{original_name}.csv", timeout=60) as response:
            lines = response.read().decode("utf-8-sig").splitlines()
        selected = lines[:52]  # Header, depot, and tasks 1 through 50.
        indices = [line.split(",", 1)[0] for line in selected[1:]]
        if indices != [str(index) for index in range(51)]:
            raise RuntimeError(f"Bộ dữ liệu {original_name} thiếu depot hoặc tác vụ ID 1–50.")
        payload = ("\n".join(selected) + "\n").encode("utf-8")
    else:
        payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != INSTANCE_SHA256[name]:
        raise RuntimeError(f"Dữ liệu không khớp với instance benchmark {name}.")
    if remote:
        path.write_bytes(payload)
    return path


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


def show_finished(out, groups, seeds, already_printed):
    counts = dict.fromkeys(groups, 0)
    for instance, method in groups:
        for seed in seeds:
            path = out / f"{instance}__{method}__{seed}.json"
            if not path.is_file():
                continue
            counts[instance, method] += 1
            key = (instance, method, seed)
            if key not in already_printed:
                record = json.loads(path.read_text(encoding="utf-8"))
                print(
                    f"{instance} | {method}, seed {seed} | makespan {record['makespan']:.4f}"
                    f" | {record['cpu_s']:.2f} giây CPU",
                    flush=True,
                )
                already_printed.add(key)
    return counts


def run_benchmark(source, instances, out, env, methods, seconds, seeds):
    command = [
        sys.executable, "-u", "-B", "run_hung_study.py",
        "--instances", *map(str, instances), "--methods", *methods,
        "--seconds", str(seconds), "--workers", str(len(seeds)),
        "--seeds", *map(str, seeds), "--out", str(out),
    ]
    log_path = out / "benchmark.log"
    seen = set()
    groups = [(path.stem, method) for path in instances for method in methods]
    total = len(groups) * len(seeds)
    show_finished(out, groups, seeds, seen)
    print(
        f"Với mỗi instance: {' → '.join(methods)}. "
        f"Mỗi nhóm có {len(seeds)} seed chạy song song. Tổng {total} lượt.",
        flush=True,
    )
    started = time.monotonic()
    interval = HEARTBEAT_SECONDS if len(instances) == 1 else 60
    next_report = started + interval
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            while proc.poll() is None:
                time.sleep(2)
                counts = show_finished(out, groups, seeds, seen)
                now = time.monotonic()
                if now >= next_report and proc.poll() is None:
                    current = next((group for group in groups if counts[group] < len(seeds)), None)
                    progress = f"Đã xong {sum(counts.values())}/{total} lượt"
                    if current:
                        progress += f" | {current[0]} / {current[1]} {counts[current]}/{len(seeds)}"
                    print(f"Đang chạy ({(now - started) / 60:.1f} phút thực). {progress}.", flush=True)
                    next_report = now + interval
        except KeyboardInterrupt:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            proc.wait()
            raise
    show_finished(out, groups, seeds, seen)
    if proc.returncode != 0:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 12000))
            print(log.read().decode("utf-8", errors="replace"), flush=True)
        raise RuntimeError(f"Benchmark dừng với mã lỗi {proc.returncode}. Nhật ký {log_path}")


def report_results(out, seeds, ts_method, names=(INSTANCE_NAME,), *, baseline=BASELINE, suffix=""):
    with (out / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = {(row["instance"], row["method"], int(row["seed"])): row for row in csv.DictReader(handle)}
    pairs = []
    summaries = []
    tolerance = 1e-7
    for name in names:
        instance_pairs = []
        label_baseline, label_proposed = METHOD_LABELS[baseline], METHOD_LABELS[ts_method]
        print(f"\n{name}")
        print(f"{'Seed':>5} {label_baseline:>16} {label_proposed:>16} {'Giảm (%)':>11}")
        for seed in seeds:
            original, proposed = rows[name, baseline, seed], rows[name, ts_method, seed]
            initial = float(original["initial"])
            if abs(initial - float(proposed["initial"])) > tolerance:
                raise RuntimeError(f"Hai thuật toán dùng nghiệm đầu khác nhau trên {name}, seed {seed}.")
            ref, result = float(original["makespan"]), float(proposed["makespan"])
            delta = ref - result
            percent = 100 * delta / ref if ref else 0.0
            pair = {"instance": name} if len(names) > 1 else {}
            pair.update({"seed": seed, "initial": initial, baseline: ref, ts_method: result,
                         "makespan_reduction": delta, "reduction_pct": percent})
            instance_pairs.append(pair)
            print(f"{seed:>5} {ref:>16.4f} {result:>16.4f} {percent:>11.2f}")
        old_mean = mean(row[baseline] for row in instance_pairs)
        new_mean = mean(row[ts_method] for row in instance_pairs)
        wins = sum(row["makespan_reduction"] > tolerance for row in instance_pairs)
        ties = sum(abs(row["makespan_reduction"]) <= tolerance for row in instance_pairs)
        losses = len(instance_pairs) - wins - ties
        pct = 100 * (old_mean - new_mean) / old_mean if old_mean else 0.0
        print(f"Makespan trung bình  |  {label_baseline} {old_mean:.4f}  |  {label_proposed} {new_mean:.4f}")
        print(f"Mức giảm {pct:+.2f}%  |  Thắng {wins}, hòa {ties}, thua {losses}")
        pairs.extend(instance_pairs)
        summaries.append(dict(instance=name, seeds=len(seeds), baseline=baseline, proposed=ts_method,
                              baseline_mean=old_mean, proposed_mean=new_mean, reduction_pct=pct,
                              wins=wins, ties=ties, losses=losses))

    write_report(out / f"paired_results{suffix}.csv", pairs)
    if len(names) > 1:
        write_report(out / f"comparison_summary{suffix}.csv", summaries)


def write_report(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_colab(seconds, seeds, ts_method, output_root, revision, instances=(INSTANCE_NAME,),
              *, compare_two_phase=False):
    """Run methods with the same start and return the result archive path."""
    if ts_method not in (TWO_PHASE, "claude_elite", INTEGRATED):
        raise ValueError("TS_METHOD không hợp lệ.")
    if compare_two_phase and ts_method != INTEGRATED:
        raise ValueError("So sánh ba bản yêu cầu phương pháp tích hợp.")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("TIME_LIMIT phải là một số dương hữu hạn.")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("SEEDS phải chứa các số nguyên từ 0 đến 2**32 - 1.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Các seed phải khác nhau.")

    names = tuple(instances)
    if not names or len(set(names)) != len(names) or any(name not in INSTANCE_SHA256 for name in names):
        raise ValueError("Chọn các instance benchmark hợp lệ và khác nhau.")
    source = Path(__file__).resolve().parent
    paths = [instance_path(source, name) for name in names]
    env = child_environment(source)
    package_probe = (
        "import importlib.metadata as md,json;"
        f"print(json.dumps([md.version(name) for name in {PACKAGE_NAMES!r}]))"
    )
    versions = subprocess.check_output([sys.executable, "-c", package_probe], env=env, text=True).strip()
    settings = [revision, platform.platform(), platform.python_version(), versions, seconds, seeds, ts_method]
    if names != (INSTANCE_NAME,):
        settings.append(names)
    methods = (BASELINE, TWO_PHASE, ts_method) if compare_two_phase else (BASELINE, ts_method)
    if compare_two_phase:
        settings.append(methods)
    tag = hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:10]
    suite = names[0] if len(names) == 1 else f"pdpcr_{len(names)}x50"
    out = Path(output_root).resolve() / f"{suite}_{seconds:g}s_{ts_method}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Kết quả từng seed đã xong được ghi ở {out}", flush=True)

    run_benchmark(source, paths, out, env, methods, seconds, seeds)
    report_results(out, seeds, ts_method, names)
    if compare_two_phase:
        report_results(out, seeds, TWO_PHASE, names, suffix="_two_phase")
        report_results(out, seeds, ts_method, names, baseline=TWO_PHASE, suffix="_vs_two_phase")
    bundle = Path(shutil.make_archive(str(out), "zip", root_dir=out))
    print(f"\nZIP kết quả  {bundle}", flush=True)
    return bundle
