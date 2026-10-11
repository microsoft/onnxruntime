# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Build the experimental KV-cache runtime using a pinned, local ONNX fork."""

import argparse
import pathlib
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-source", type=pathlib.Path, required=True, help="Read-only ONNX fork repository")
    parser.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build") / "kvcache-native")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    revision = (root / "cmake" / "kvcache_onnx_revision.txt").read_text().strip()
    source = args.onnx_source.resolve()
    build = args.build_dir.resolve()
    checkout = build / "onnx-source"
    if not checkout.exists():
        build.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"Build ONNX checkout {actual} differs from pin {revision}; use a new build directory")
    cmake = shutil.which("cmake") or str(pathlib.Path(sys.executable).parent / "cmake.exe")
    ctest = shutil.which("ctest") or str(pathlib.Path(sys.executable).parent / "ctest.exe")
    command = [
        sys.executable,
        str(root / "tools" / "ci_build" / "build.py"),
        "--config",
        "Release",
        "--build_dir",
        str(build),
        "--cmake_path",
        cmake,
        "--ctest_path",
        ctest,
        "--update",
        "--skip_tests",
        "--skip_submodule_sync",
        "--disable_ml_ops",
        "--build_wheel",
        "--parallel",
        str(args.parallel),
        "--cmake_extra_defines",
        f"FETCHCONTENT_SOURCE_DIR_ONNX={checkout}",
        "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER",
    ]
    if not args.configure_only:
        command += ["--build", "--target", "onnxruntime_kvcache_test", "--target", "onnxruntime_pybind11_state"]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
