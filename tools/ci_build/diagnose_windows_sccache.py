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


def preprocessing_commands(text):
    commands = []
    for line in text.splitlines():
        if "preprocess: Some(" in line:
            commands.append(
                [json.loads(token) for token in re.findall(r'"(?:\\.|[^"\\])*"', line.split("preprocess: Some(", 1)[1])]
            )
    return commands


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
    environment["SCCACHE_BASEDIRS"] = str(build_dir)
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
                            "-DCMAKE_CXX_FLAGS=/MP /guard:cf /Qspectre /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100",
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
    commands = preprocessing_commands(Path(os.environ["SCCACHE_ERROR_LOG"]).read_text(encoding="utf-8"))
    if not commands:
        raise RuntimeError("No actual preprocessing commands found in the sccache diagnostic log")
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
            # Use the actual command, not tracking logs that normalize paths and omit options.
            matches = [
                command
                for command in commands
                if source.replace("\\", "/").casefold() == command[-1].replace("\\", "/").casefold()
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one actual preprocessing command for {source}, found {len(matches)}")
            record["preprocess_command"] = matches[0]
            # /EP suppresses compilation; use the real compiler, never the cache wrapper.
            result = subprocess.run(
                matches[0],
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


def probe_target(build_dir, output_dir):
    """Capture real preprocessor stdout inside one existing full-build MSBuild target."""
    compiler = shutil.which("cl")
    if compiler is None:
        raise FileNotFoundError("Configure the Visual C++ environment before probing")
    msbuild = Path(os.environ["VSINSTALLDIR"]) / "MSBuild" / "Current" / "Bin" / "MSBuild.exe"
    project = build_dir / "onnxruntime_provider_bridge_valid_fixture.vcxproj"
    if not msbuild.is_file() or not project.is_file():
        raise FileNotFoundError(f"Full-build target probe requires {msbuild} and {project}")
    output_dir.mkdir(parents=True, exist_ok=False)
    proxy = output_dir / "compiler" / "x64"
    proxy.mkdir(parents=True)
    csc = Path(os.environ["WINDIR"]) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
    subprocess.run(
        [
            str(csc),
            "/nologo",
            "/target:exe",
            f"/out:{proxy / 'cl.exe'}",
            str(Path(__file__).with_name("capture_windows_compiler.cs")),
        ],
        check=True,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(proxy) + os.pathsep + environment["PATH"],
            "SCCACHE_DIAGNOSTIC_COMPILER": compiler,
            "SCCACHE_DIR": str(output_dir / "cache"),
            "SCCACHE_LOCAL_RW_MODE": "READ_WRITE",
            "SCCACHE_LOG": "sccache::compiler::c=trace,sccache::compiler::msvc=debug",
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        environment["SCCACHE_SERVER_PORT"] = str(listener.getsockname()[1])
    results = []
    for preserve in (False, True):
        for repeat in (1, 2):
            phase = f"{'preserve' if preserve else 'delete'}-{repeat}"
            directory = output_dir / phase
            directory.mkdir()
            environment.pop("MSBUILDPRESERVETOOLTEMPFILES", None)
            if preserve:
                environment["MSBUILDPRESERVETOOLTEMPFILES"] = "1"
            environment["SCCACHE_DIAGNOSTIC_OUTPUT"] = str(directory)
            environment["SCCACHE_ERROR_LOG"] = str(directory / "keys.log")
            # Recompile only this fixture's object; never clean or relink the test binaries.
            obj = (
                build_dir
                / "onnxruntime_provider_bridge_valid_fixture.dir"
                / "RelWithDebInfo"
                / "provider_bridge_test_library.obj"
            )
            obj.unlink(missing_ok=True)
            subprocess.run(["sccache", "--start-server"], env=environment, check=True)
            try:
                subprocess.run(["sccache", "--zero-stats"], env=environment, check=True)
                with (directory / "build.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            str(msbuild),
                            str(project),
                            "/t:ClCompile",
                            "/p:Configuration=RelWithDebInfo",
                            "/p:Platform=x64",
                            "/p:BuildProjectReferences=false",
                            "/p:ExecutablePath=" + environment["PATH"].replace(";", "%3B"),
                            "/verbosity:minimal",
                            "/nologo",
                        ],
                        cwd=build_dir,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                stats = json.loads(
                    subprocess.check_output(
                        ["sccache", "--show-stats", "--stats-format=json"], env=environment, text=True
                    )
                )
                if stats["stats"]["requests_executed"] != 1:
                    raise RuntimeError(f"Expected one cacheable fixture compilation, got {stats}")
                if not any(
                    "provider_bridge_test_library.cc" in path.read_text(encoding="utf-8").lower()
                    for path in directory.glob("*.inputs.txt")
                ):
                    raise RuntimeError("The diagnostic compiler proxy did not capture the fixture preprocessing")
                results.append({"phase": phase, "cache": stats})
            finally:
                subprocess.run(["sccache", "--stop-server"], env=environment, check=True)
    (output_dir / "probe-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--probe", action="store_true", help="Isolate MSBuild response-file preservation in a small fixture"
    )
    mode.add_argument(
        "--probe-target", action="store_true", help="Capture actual full-target preprocessing byte streams"
    )
    args = parser.parse_args()
    action = probe_target if args.probe_target else probe if args.probe else collect
    action(args.build_dir.resolve(), args.output_dir.resolve())
