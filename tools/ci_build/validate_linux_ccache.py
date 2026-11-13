# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Measure real cold/warm ORT builds without publishing PR compiler caches."""

import json
import os
import subprocess
import time
from pathlib import Path


def validate_statistics(phase, hits, misses):
    if phase == "cold" and misses < 100:
        raise RuntimeError(f"Cold build did not populate a real ORT cache: {misses} misses")
    if phase == "warm" and (hits < 100 or hits / (hits + misses) < 0.9):
        raise RuntimeError(f"Warm ORT cache hit rate below 90%: {hits} hits, {misses} misses")


def main():
    phase = os.environ["CACHE_PHASE"]
    if phase not in ("cold", "warm"):
        raise ValueError(f"Unknown cache phase: {phase}")
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    output = Path(os.environ["RUNNER_TEMP"]) / "ccache-validation"
    output.mkdir(exist_ok=True)
    image = os.environ["VALIDATION_IMAGE"]
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    # Different host directory depths, but identical compiler-visible container paths.
    cache_workspace = output / ("cold" if phase == "cold" else "warm/deeper/workspace")
    cache_workspace.mkdir(parents=True, exist_ok=True)
    if phase == "warm":
        cold = json.loads((output / "cold.json").read_text())
        if cold["image_id"] != image_id:
            raise RuntimeError("Cold/warm images differ; refusing a cross-toolchain comparison")
        members = subprocess.check_output(["tar", "-tf", str(output / "cache.tar")], text=True).splitlines()
        if not members or any(not name.startswith(".cache/ccache/") or ".." in name.split("/") for name in members):
            raise RuntimeError("Snapshot contains non-workspace-relative cache entries")
        subprocess.run(["tar", "-xf", str(output / "cache.tar"), "-C", str(cache_workspace)], check=True)
    cache = cache_workspace / ".cache/ccache"
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "-R", "a+rwX", str(cache)], check=True)
    (workspace / ".cache/ccache").mkdir(parents=True, exist_ok=True)
    build = output / "build"
    build.mkdir()
    build.chmod(0o777)
    docker = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{workspace}:/onnxruntime_src",
        "--volume",
        f"{build}:/build",
        "--volume",
        f"{cache}:/onnxruntime_src/.cache/ccache",
        "--workdir",
        "/onnxruntime_src",
        "-e",
        "CCACHE_DIR=/onnxruntime_src/.cache/ccache",
        "-e",
        "CCACHE_UMASK=000",
        image,
        "bash",
        "-ec",
    ]

    def run(command, name):
        started = time.monotonic()
        with (output / f"{phase}-{name}.log").open("w") as log:
            subprocess.run([*docker, command], stdout=log, stderr=subprocess.STDOUT, check=True)
        return time.monotonic() - started

    flags = (
        "python3 tools/ci_build/build.py --build_dir /build --config Debug "
        "--cmake_generator Ninja --parallel 8 --skip_submodule_sync --build_shared_lib "
        "--use_binskim_compliant_compile_flags --use_cache --minimal_build extended "
        "--skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF"
    )
    run(f"{flags} --update", "configure")
    elapsed = run(f"ccache --max-size=5G && ccache --zero-stats && {flags} --build", "build")
    stats = dict(
        line.split("\t") for line in subprocess.check_output([*docker, "ccache --print-stats"], text=True).splitlines()
    )
    hits = int(stats["direct_cache_hit"]) + int(stats["preprocessed_cache_hit"])
    misses = int(stats["cache_miss"])
    if not (build / "Debug/libonnxruntime.so").exists():
        raise RuntimeError("ORT shared library was not built")
    validate_statistics(phase, hits, misses)
    result = {"phase": phase, "image_id": image_id, "seconds": elapsed, "hits": hits, "misses": misses, "stats": stats}
    (output / f"{phase}.json").write_text(json.dumps(result, indent=2))
    if phase == "cold":
        subprocess.run(
            ["tar", "--posix", "-cf", str(output / "cache.tar"), "-P", "-C", str(cache_workspace), ".cache/ccache"],
            check=True,
        )
    summary = f"| {phase} | {elapsed:.3f} | {hits} | {misses} |\n"
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as report:
        report.write("| ORT build | Seconds | Hits | Misses |\n| --- | ---: | ---: | ---: |\n")
        if phase == "warm":
            report.write(f"| cold | {cold['seconds']:.3f} | {cold['hits']} | {cold['misses']} |\n")
        report.write(summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
