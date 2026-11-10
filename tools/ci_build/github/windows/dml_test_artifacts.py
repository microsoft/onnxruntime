#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Stage and relocate the runtime-only artifacts for the Windows DirectML CI tests."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

CONFIG = "RelWithDebInfo"
REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = "dml-test-paths.json"


def require_runtime(build_dir: Path):
    for name in (
        "onnxruntime_test_all.exe",
        "onnxruntime_provider_test.exe",
        "onnxruntime.dll",
        "DirectML.dll",
        "onnxruntime_test_python.py",
        "onnxruntime_test_python_iobinding.py",
        "onnxruntime_test_python_dmlgraph.py",
        "testdata",
        "onnxruntime/capi/onnxruntime_pybind11_state.pyd",
        "onnxruntime/capi/DirectML.dll",
    ):
        if not (build_dir / CONFIG / name).exists():
            raise FileNotFoundError(f"Required DirectML test artifact is missing: {build_dir / CONFIG / name}")
    if not (build_dir / "CTestTestfile.cmake").is_file():
        raise FileNotFoundError(f"Missing CTest registration: {build_dir}")
    if len(list((build_dir / CONFIG / "dist").glob("*.whl"))) != 1:
        raise ValueError("Expected exactly one DirectML wheel in the test artifact")


def stage(build_dir: Path, output: Path):
    require_runtime(build_dir)
    # Keep fixtures and Python packages intact; leave intermediates and debug symbols on the CPU runner.
    shutil.copytree(
        build_dir / CONFIG,
        output / CONFIG,
        ignore=shutil.ignore_patterns("*.obj", "*.pch", "*.lib", "*.exp", "*.pdb", "*.tlog", "*.dir"),
    )
    for ctest_file in build_dir.rglob("CTestTestfile.cmake"):
        destination = output / ctest_file.relative_to(build_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ctest_file, destination)

    # CTest can reference ONNX fixtures outside the runtime output directory.
    cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    onnx_source = re.search(r"^onnx_SOURCE_DIR:[^=]+=(.+)$", cache, re.MULTILINE)
    if onnx_source:
        data = Path(onnx_source[1].strip()) / "onnx" / "backend" / "test" / "data"
        if data.is_dir():
            shutil.copytree(data, output / data.relative_to(build_dir))
        else:
            print(f"ONNX backend fixtures are not present in the original build: {data}")

    (output / MANIFEST).write_text(
        json.dumps(
            {"build": build_dir.as_posix(), "repo": REPO_ROOT.as_posix(), "python": Path(sys.executable).as_posix()}
        ),
        encoding="utf-8",
    )


def prepare(build_dir: Path):
    require_runtime(build_dir)
    paths = json.loads((build_dir / MANIFEST).read_text(encoding="utf-8"))
    replacements = {}
    for key, new in (
        ("build", build_dir.as_posix()),
        ("repo", REPO_ROOT.as_posix()),
        ("python", Path(sys.executable).as_posix()),
    ):
        replacements[paths[key]] = new
        replacements[paths[key].replace("/", "\\\\")] = new
    pattern = re.compile("|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True)))
    # Nested CTest registrations and Python commands also contain producer-machine paths.
    for ctest_file in build_dir.rglob("CTestTestfile.cmake"):
        text = ctest_file.read_text(encoding="utf-8")
        text = pattern.sub(lambda match: replacements[match[0]], text)
        ctest_file.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare"))
    parser.add_argument("--build-dir", type=Path, required=True, help="The RelWithDebInfo CMake build directory.")
    parser.add_argument("--output", type=Path, help="Fresh artifact staging directory (stage only).")
    args = parser.parse_args()
    if args.mode == "stage":
        if args.output is None:
            parser.error("--output is required for stage")
        stage(args.build_dir.resolve(), args.output.resolve())
    else:
        prepare(args.build_dir.resolve())


if __name__ == "__main__":
    main()
