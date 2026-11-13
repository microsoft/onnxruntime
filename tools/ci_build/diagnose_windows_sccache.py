# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Capture compiler-key inputs for explicit Windows cache experiments."""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from build import windows_sccache_globals


def command_records(text):
    source = None
    for line in text.splitlines():
        if line.startswith("^"):
            source = line[1:]
        elif line.strip():
            if source is None:
                raise ValueError("Compiler command without a source record")
            yield source, line


def compiler_responses(directory):
    responses = []
    for path in sorted(directory.rglob("*.rsp")):
        raw = path.read_bytes()
        command = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        if re.search(r"(?<!\S)/c(?=\s|$)", command):
            responses.append({"file": str(path), "command": command})
    return responses


def probe(build_dir, output_dir):
    build_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\nproject(probe LANGUAGES CXX)\nadd_executable(probe main.cpp)\n",
        encoding="utf-8",
    )
    (source / "main.cpp").write_text(
        '#include <iostream>\nint main() { std::cout << "cache probe\\n"; }\n', encoding="utf-8"
    )
    args = SimpleNamespace(
        cache_tool="sccache",
        cmake_generator="Visual Studio 17 2022",
        arm64=False,
        arm64ec=False,
        arm=False,
        x86=False,
        msvc_toolset=None,
        use_cuda=False,
    )
    tree = build_dir / "build"
    globals_ = windows_sccache_globals(args, tree, None)
    environment = os.environ.copy()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        environment["SCCACHE_SERVER_PORT"] = str(listener.getsockname()[1])
    environment["SCCACHE_DIR"] = str(build_dir / "cache")
    environment["SCCACHE_LOG"] = "sccache::compiler::c=trace,sccache::compiler::msvc=debug"
    results = []
    for preserve in (False, True):
        for warm in (False, True):
            phase = f"{'preserve' if preserve else 'delete'}-{'warm' if warm else 'seed'}"
            environment.pop("MSBUILDPRESERVETOOLTEMPFILES", None)
            if preserve:
                environment["MSBUILDPRESERVETOOLTEMPFILES"] = "1"
            environment["SCCACHE_LOCAL_RW_MODE"] = "READ_ONLY" if warm else "READ_WRITE"
            debug_log = output_dir / f"{phase}-keys.log"
            environment["SCCACHE_ERROR_LOG"] = str(debug_log)

            def run(command):
                return subprocess.check_output(command, env=environment, stderr=subprocess.STDOUT, text=True)

            run(["sccache", "--start-server"])
            try:
                if not results:
                    run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(tree),
                            "-G",
                            args.cmake_generator,
                            "-A",
                            "x64",
                            "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded",
                            "-DCMAKE_VS_GLOBALS="
                            + ";".join([*globals_, "UseMultiToolTask=true", "EnforceProcessCountAcrossBuilds=true"]),
                        ]
                    )
                run(["sccache", "--zero-stats"])
                log = run(["cmake", "--build", str(tree), "--config", "RelWithDebInfo", "--clean-first"])
                (output_dir / f"{phase}-build.log").write_text(log, encoding="utf-8")
                run([str(tree / "RelWithDebInfo" / "probe.exe")])
                results.append(
                    {"phase": phase, "cache": json.loads(run(["sccache", "--show-stats", "--stats-format=json"]))}
                )
            finally:
                run(["sccache", "--stop-server"])
            for line in debug_log.read_text(encoding="utf-8").splitlines():
                if "preprocess: Some(" not in line or "main.cpp" not in line:
                    continue
                command = [
                    json.loads(token)
                    for token in re.findall(r'"(?:\\.|[^"\\])*"', line.split("preprocess: Some(", 1)[1])
                ]
                result = subprocess.run(command, env=environment, cwd=tree, capture_output=True, check=True)
                (output_dir / f"{phase}-preprocessed.txt").write_bytes(result.stdout)
    (output_dir / "probe-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        stats = result["cache"]["stats"]
        print(result["phase"], "hits", stats["cache_hits"]["counts"], "misses", stats["cache_misses"]["counts"])


def collect(build_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cl")
    if compiler is None:
        raise FileNotFoundError("Configure the Visual C++ environment before collecting diagnostics")
    responses = compiler_responses(Path(tempfile.gettempdir()) / "MSBuildTemp")
    if not responses:
        raise RuntimeError("No preserved compiler response files; set MSBUILDPRESERVETOOLTEMPFILES=1 before building")
    records = []
    selected = set()
    for log in sorted(build_dir.rglob("CL.command.1.tlog")):
        if "CompilerId" in str(log):
            continue
        # MSBuild's tracking logs use UTF-16 and uppercase path spellings.
        for source, command in command_records(log.read_text(encoding="utf-16")):
            record = {
                "source": source,
                "log": str(log.relative_to(build_dir)),
                "command": command,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "cwd": str(log.parents[3]),
            }
            records.append(record)
            if re.search(r"/(?:Yc|Yu|Fp)", command):
                continue
            target = log.parent.parent.parent.name
            if target in selected or len(selected) >= 4:
                continue
            selected.add(target)
            # Tracking logs normalize path casing and can omit per-invocation options.
            matches = [
                response
                for response in responses
                if source.replace("\\", "/").casefold() in response["command"].replace("\\", "/").casefold()
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one actual compiler response for {source}, found {len(matches)}")
            record["response"] = matches[0]
            # /EP suppresses compilation; use the real compiler, never the cache wrapper.
            preprocess = re.sub(r"(?<!\S)/c(?=\s|$)", "/EP", matches[0]["command"])
            result = subprocess.run(
                subprocess.list2cmdline([compiler]) + " " + preprocess + " /WX-",
                cwd=log.parents[3],
                capture_output=True,
                check=True,
            )
            filename = f"preprocessed-{len(selected)}.txt"
            (output_dir / filename).write_bytes(result.stdout)
            (output_dir / f"preprocessed-{len(selected)}.stderr.txt").write_bytes(result.stderr)
            record["preprocessed"] = filename
            record["preprocessed_sha256"] = hashlib.sha256(result.stdout).hexdigest()
    if not records or not selected:
        raise RuntimeError("No cacheable compiler command records found")
    result = {
        "compiler": compiler,
        "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        "responses": responses,
        # Only cache-key inputs, not the complete environment (which can contain credentials).
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CL",
                "_CL_",
                "INCLUDE",
                "LIB",
                "SCCACHE_BASEDIRS",
                "SCCACHE_C_CUSTOM_CACHE_BUSTER",
                "SCCACHE_EXTRAFILES",
                "SDKROOT",
                "CCC_OVERRIDE_OPTIONS",
                "MACOSX_DEPLOYMENT_TARGET",
                "IPHONEOS_DEPLOYMENT_TARGET",
                "TVOS_DEPLOYMENT_TARGET",
                "WATCHOS_DEPLOYMENT_TARGET",
            )
        },
        "commands": records,
    }
    (output_dir / "compiler-inputs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Captured {len(records)} compiler commands and {len(selected)} preprocessed translation units")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe", action="store_true", help="Isolate MSBuild response-file preservation in a small fixture"
    )
    args = parser.parse_args()
    (probe if args.probe else collect)(args.build_dir.resolve(), args.output_dir.resolve())
