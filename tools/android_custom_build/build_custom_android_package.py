#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pathlib
import shutil
import subprocess


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_OPS_CONFIG_RELATIVE_PATH = "tools/ci_build/github/android/mobile_package.required_operators.config"
DEFAULT_BUILD_SETTINGS_RELATIVE_PATH = "tools/ci_build/github/android/default_mobile_aar_build_settings.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Builds a custom ONNX Runtime Android package.")

    parser.add_argument("--onnxruntime_branch",
                        help="The specific ONNX Runtime branch to build. "
                        "Supported for 1.11 (rel-1.11.0) and later.")

    parser.add_argument("--include_ops_by_config", type=pathlib.Path,
                        help="The configuration file specifying which ops to include. "
                        f"The default is {DEFAULT_OPS_CONFIG_RELATIVE_PATH} in the ONNX Runtime repo.")

    parser.add_argument("--build_settings", type=pathlib.Path,
                        help="The configuration file specifying the build.py options. "
                        f"The default is {DEFAULT_BUILD_SETTINGS_RELATIVE_PATH} in the ONNX Runtime repo.")

    parser.add_argument("--config", choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"], default="Release",
                        help="The build configuration.")

    parser.add_argument("--working_dir", type=pathlib.Path, required=True,
                        help="The directory used to store intermediate and output files.")

    parser.add_argument("--docker_image_tag", default="onnxruntime-android-custom-build:latest",
                        help="The tag for the build Docker image.")

    parser.add_argument("--docker_path", default=shutil.which("docker"),
                        help="The path to docker.")

    args = parser.parse_args()

    if args.docker_path is None:
        raise ValueError("Unable to determine docker path. Please specify with --docker_path.")

    return args


def main():
    args = parse_args()

    docker_build_args = ["--build-arg", f"ONNXRUNTIME_BRANCH={args.onnxruntime_branch}"] \
        if args.onnxruntime_branch else []

    docker_build_cmd = [args.docker_path, "build",
                        "--tag", args.docker_image_tag,
                        "--file", str(SCRIPT_DIR / "Dockerfile"),
                        ] + docker_build_args + [str(SCRIPT_DIR)]

    subprocess.run(docker_build_cmd, check=True)

    working_dir = args.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)
    working_dir = working_dir.resolve()

    # copy over any custom build configuration files
    config_files = [f for f in [args.include_ops_by_config, args.build_settings] if f]
    if config_files:
        input_dir = working_dir / "input"
        input_dir.mkdir(exist_ok=True)
        for config_file in config_files:
            shutil.copy(config_file, input_dir)

    output_dir = working_dir / "output"
    output_dir.mkdir(exist_ok=True)

    container_ops_config_file = \
        f"/workspace/shared/input/{args.include_ops_by_config.name}" if args.include_ops_by_config \
        else f"/workspace/onnxruntime/{DEFAULT_OPS_CONFIG_RELATIVE_PATH}"

    container_build_settings_file =\
        f"/workspace/shared/input/{args.build_settings.name}" if args.build_settings \
        else f"/workspace/onnxruntime/{DEFAULT_BUILD_SETTINGS_RELATIVE_PATH}"

    docker_run_cmd = [args.docker_path, "run",
                      "--rm", "-it",
                      f"--volume={str(working_dir)}:/workspace/shared",
                      args.docker_image_tag,
                      "/usr/bin/env", "python3",
                      "/workspace/onnxruntime/tools/ci_build/github/android/build_aar_package.py",
                      "--build_dir=/workspace/shared/output",
                      f"--config={args.config}",
                      f"--include_ops_by_config={container_ops_config_file}",
                      container_build_settings_file,
                      ]

    subprocess.run(docker_run_cmd, check=True)

    print("Finished building Android package at '{}'.".format(output_dir / "aar_out"))


if __name__ == "__main__":
    main()
