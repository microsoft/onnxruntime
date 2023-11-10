#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]


def parse_args():
    parser = argparse.ArgumentParser(description="Builds ORT and checks the binary size.")

    parser.add_argument("build_check_binsize_config", type=pathlib.Path, help="Path to configuration file.")
    parser.add_argument("--build_dir", type=pathlib.Path, required=True, help="Path to build directory.")
    parser.add_argument("--threshold_size_in_bytes", type=int, help="Binary size limit in bytes.")
    parser.add_argument(
        "--with_debug_info", action="store_true", help="Whether to include debug information in the build."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.build_check_binsize_config) as config_file:
        config = json.load(config_file)

    config_type = config["type"]
    os = config["os"]
    arch = config["arch"]
    build_params = config["build_params"]
    build_config = "MinSizeRel"  # could make this configurable if needed
    # Build and install protoc
    protobuf_installation_script = (
        REPO_ROOT
        / "tools"
        / "ci_build"
        / "github"
        / "linux"
        / "docker"
        / "inference"
        / "x64"
        / "python"
        / "cpu"
        / "scripts"
        / "install_protobuf.sh"
    )
    subprocess.run(
        [
            str(protobuf_installation_script),
            "-p",
            str(pathlib.Path(args.build_dir) / "installed"),
            "-d",
            str(REPO_ROOT / "cmake" / "deps.txt"),
        ],
        shell=False,
        check=True,
    )
    # build ORT
    build_command = (
        [sys.executable, str(REPO_ROOT / "tools/ci_build/build.py"), *build_params]
        + (["--cmake_extra_defines", "ADD_DEBUG_INFO_TO_MINIMAL_BUILD=ON"] if args.with_debug_info else [])
        # put the following options last so they don't get overridden by build_params
        + [
            f"--build_dir={args.build_dir}",
            f"--config={build_config}",
            "--update",
            "--build",
            "--parallel",
            "--test",
            "--path_to_protoc_exe",
            str(pathlib.Path(args.build_dir) / "installed" / "bin" / "protoc"),
        ]
    )

    subprocess.run(build_command, check=True)

    # check binary size
    check_binary_size_command = (
        [
            sys.executable,
            str(REPO_ROOT / "tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py"),
            f"--os={os}",
            f"--arch={arch}",
            f"--build_config={config_type}",
        ]
        + ([f"--threshold={args.threshold_size_in_bytes}"] if args.threshold_size_in_bytes else [])
        + [str(args.build_dir / build_config / "libonnxruntime.so")]
    )

    subprocess.run(check_binary_size_command, check=True)


if __name__ == "__main__":
    main()
