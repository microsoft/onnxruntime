# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Compare clean uncached/cold/warm builds in a disposable Visual Studio build tree."""

import argparse
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from build import windows_sccache_globals


def benchmark(build_dir, config, target, parallel, output_dir):
    build_dir = Path(build_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for line in (build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            name, value = line.split("=", 1)
            cache[name.split(":", 1)[0]] = value
    generator = cache["CMAKE_GENERATOR"]
    if not generator.startswith("Visual Studio "):
        raise ValueError("The benchmark requires a configured Visual Studio build tree")
    architecture = cache.get("CMAKE_GENERATOR_PLATFORM", "").lower()
    cuda = cache.get("onnxruntime_USE_CUDA", "OFF") == "ON"
    args = SimpleNamespace(
        cache_tool="sccache",
        cmake_generator=generator,
        arm64=architecture == "arm64",
        arm64ec=architecture == "arm64ec",
        arm=architecture == "arm",
        x86=architecture == "win32",
        msvc_toolset=None,
        use_cuda=cuda,
    )
    controlled = {"CLToolPath", "CLToolExe", "CudaToolkitNvccPath", "ForceImportBeforeCppTargets"}
    globals_ = [
        item
        for item in cache.get("CMAKE_VS_GLOBALS", "").split(";")
        if item and item.split("=", 1)[0] not in controlled
    ]
    for item in ("UseMultiToolTask=true", "EnforceProcessCountAcrossBuilds=true"):
        if item not in globals_:
            globals_.append(item)
    result = {
        "commit": os.environ.get("GITHUB_SHA"),
        "machine": platform.platform(),
        "processor_count": os.cpu_count(),
        "parallel": parallel,
        "generator": generator,
        "config": config,
        "target": target,
        "scope": "Clean target compilation and linking; excludes setup, packaging, tests and snapshot transfer.",
        "phases": [],
    }

    def save():
        (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def run(command, environment, log_name):
        print(subprocess.list2cmdline([str(value) for value in command]), flush=True)
        start = time.perf_counter()
        with (output_dir / log_name).open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="ort-sccache-benchmark-") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        environment = dict(
            os.environ,
            SCCACHE_DIR=directory,
            SCCACHE_CACHE_SIZE="10G",
            SCCACHE_SERVER_PORT=str(port),
            SCCACHE_LOCAL_RW_MODE="READ_WRITE",
        )
        result["sccache"] = subprocess.check_output(["sccache", "--version"], text=True).strip()
        result["compiler"] = subprocess.check_output(["where.exe", "cl"], text=True).strip()
        save()
        for phase in ("uncached", "cold", "warm"):
            phase_result = {"phase": phase}
            result["phases"].append(phase_result)
            if phase != "warm":
                selected_globals = list(globals_)
                if phase == "cold":
                    cuda_home = str(Path(cache["CMAKE_CUDA_COMPILER"]).parent.parent) if cuda else ""
                    selected_globals.extend(windows_sccache_globals(args, str(build_dir), cuda_home))
                phase_result["configure_seconds"] = run(
                    [
                        "cmake",
                        "-S",
                        cache["CMAKE_HOME_DIRECTORY"],
                        "-B",
                        str(build_dir),
                        "-DCMAKE_VS_GLOBALS=" + ";".join(selected_globals),
                        "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT="
                        + ("ProgramDatabase" if phase == "uncached" else "Embedded"),
                        "-Donnxruntime_BUILD_CACHE=" + ("OFF" if phase == "uncached" else "ON"),
                    ],
                    environment,
                    f"{phase}-configure.log",
                )
            server_started = False
            try:
                if phase != "uncached":
                    environment["SCCACHE_LOCAL_RW_MODE"] = "READ_ONLY" if phase == "warm" else "READ_WRITE"
                    run(["sccache", "--start-server"], environment, f"{phase}-start.log")
                    server_started = True
                    location = json.loads(
                        subprocess.check_output(
                            ["sccache", "--show-stats", "--stats-format=json"], env=environment, text=True
                        )
                    )
                    if not location["cache_location"].startswith("Local disk:"):
                        raise RuntimeError("The benchmark must use only its isolated local cache")
                    run(["sccache", "--zero-stats"], environment, f"{phase}-zero.log")
                phase_result["build_seconds"] = run(
                    [
                        "cmake",
                        "--build",
                        str(build_dir),
                        "--config",
                        config,
                        "--target",
                        target,
                        "--clean-first",
                        "--parallel",
                        str(parallel),
                        "--",
                        "/verbosity:minimal",
                    ],
                    environment,
                    f"{phase}-build.log",
                )
                if server_started:
                    stats = json.loads(
                        subprocess.check_output(
                            ["sccache", "--show-stats", "--stats-format=json"], env=environment, text=True
                        )
                    )
                    phase_result["cache"] = stats
                    save()
                    counts = stats["stats"]["cache_misses" if phase == "cold" else "cache_hits"]["counts"]
                    if not sum(counts.values()) or stats["stats"]["compile_fails"]:
                        raise RuntimeError(f"No successful {phase} cache evidence; inspect results.json and build logs")
            finally:
                if server_started:
                    run(["sccache", "--stop-server"], environment, f"{phase}-stop.log")
                save()

    baseline, cold, warm = (entry["build_seconds"] for entry in result["phases"])
    result["warm_speedup"] = baseline / warm
    result["warm_time_saved_percent"] = 100 * (baseline - warm) / baseline
    result["cold_overhead_percent"] = 100 * (cold - baseline) / baseline
    save()
    summary = (
        f"### Windows compiler-cache benchmark: `{target}`\n\n"
        "| Clean build | Seconds |\n| --- | ---: |\n"
        f"| Uncached | {baseline:.2f} |\n| Cold cache (read/write) | {cold:.2f} |\n"
        f"| Warm cache (read-only) | {warm:.2f} |\n\n"
        f"Warm speedup: {result['warm_speedup']:.2f}x; time saved: {result['warm_time_saved_percent']:.1f}%; "
        f"cold overhead: {result['cold_overhead_percent']:.1f}%.\n\n"
        f"{result['scope']} PCH passthrough and cache misses are included. "
        "One sequential trial per phase on the same runner, not a whole-job speedup or a statistical estimate.\n"
    )
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
            output.write(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, help="Disposable, already configured CMake build directory")
    parser.add_argument("--config", default="RelWithDebInfo")
    parser.add_argument("--target", default="onnxruntime")
    parser.add_argument("--parallel", type=int, default=os.cpu_count())
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    benchmark(args.build_dir, args.config, args.target, args.parallel, args.output_dir)


if __name__ == "__main__":
    main()
