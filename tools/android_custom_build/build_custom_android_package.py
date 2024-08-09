#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def is_windows():
    return sys.platform.startswith("win")


def run(cmd_arg_list, **kwargs):
    print(f"Running command:\n  {shlex.join(cmd_arg_list)}")
    kwargs.update({"check": True})
    return subprocess.run(cmd_arg_list, **kwargs)  # noqa: PLW1510


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Builds a custom ONNX Runtime Android package.
                    This script first builds a Docker image with the ONNX Runtime Android build environment
                    dependencies. Then, from a Docker container with that image, it calls the ONNX Runtime build
                    scripts to build a custom Android package. The resulting package will be under
                    <working_dir>/output/aar_out. See https://onnxruntime.ai/docs/build/custom.html for more
                    information about custom builds."""
    )

    parser.add_argument(
        "working_dir", type=pathlib.Path, help="The directory used to store intermediate and output files."
    )

    parser.add_argument(
        "--onnxruntime_branch_or_tag",
        help="The ONNX Runtime branch or tag to build. "
        "Supports branches and tags starting from 1.11 (branch rel-1.11.0 or tag v1.11.0). "
        "If unspecified, builds the latest.",
    )

    parser.add_argument(
        "--onnxruntime_repo_url", help="The ONNX Runtime repo URL. If unspecified, uses the official repo."
    )

    parser.add_argument(
        "--include_ops_by_config",
        type=pathlib.Path,
        help="The optional configuration file specifying which ops to include. "
        "Such a configuration file is generated during ONNX to ORT format model conversion. "
        "When providing this option, consider also enabling op type support reduction by specifying "
        "the build.py --enable_reduced_operator_type_support option in the build settings configuration file.",
    )

    parser.add_argument(
        "--build_settings",
        type=pathlib.Path,
        required=True,
        help="The configuration file specifying the build.py options. "
        "For an example, see tools/ci_build/github/android/default_full_aar_build_settings.json.",
    )

    default_config = "Release"
    parser.add_argument(
        "--config",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        default=default_config,
        help=f"The build configuration. The default is {default_config}.",
    )

    default_docker_image_tag = "onnxruntime-android-custom-build:latest"
    parser.add_argument(
        "--docker_image_tag",
        default=default_docker_image_tag,
        help=f"The tag for the Docker image. The default is {default_docker_image_tag}.",
    )

    parser.add_argument(
        "--docker_container_name",
        help="The name of the Docker container that is run (given to the --name option of `docker run`). "
        "If unspecified, the container will be automatically removed. "
        "Keeping the container may be useful for debugging.",
    )

    parser.add_argument(
        "--docker_path",
        default=shutil.which("docker"),
        help="The path to docker. If unspecified, docker should be in PATH.",
    )

    args = parser.parse_args()

    if not args.build_settings.is_file():
        raise ValueError(f"--build_settings argument is not a file: {args.build_settings}")

    if args.include_ops_by_config is not None and not args.include_ops_by_config.is_file():
        raise ValueError(f"--include_ops_by_config argument is not a file: {args.include_ops_by_config}")

    if args.docker_path is None:
        raise ValueError("Unable to determine docker path. Please provide it with --docker_path.")

    return args


def main():
    args = parse_args()

    docker_build_image_args = []
    if args.onnxruntime_branch_or_tag:
        docker_build_image_args += ["--build-arg", f"ONNXRUNTIME_BRANCH_OR_TAG={args.onnxruntime_branch_or_tag}"]
    if args.onnxruntime_repo_url:
        docker_build_image_args += ["--build-arg", f"ONNXRUNTIME_REPO={args.onnxruntime_repo_url}"]
    if not is_windows():
        docker_build_image_args += ["--build-arg", f"BUILD_UID={os.geteuid()}"]

    docker_build_image_cmd = [
        args.docker_path,
        "build",
        "--tag",
        args.docker_image_tag,
        "--file",
        str(SCRIPT_DIR / "Dockerfile"),
        *docker_build_image_args,
        str(SCRIPT_DIR),
    ]

    run(docker_build_image_cmd)

    working_dir = args.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)
    working_dir = working_dir.resolve()

    # copy over custom build configuration files
    config_files = [f for f in [args.include_ops_by_config, args.build_settings] if f]
    if config_files:
        input_dir = working_dir / "input"
        input_dir.mkdir(exist_ok=True)
        for config_file in config_files:
            shutil.copy(config_file, input_dir)

    output_dir = working_dir / "output"
    output_dir.mkdir(exist_ok=True)

    container_ops_config_file = (
        f"/workspace/shared/input/{args.include_ops_by_config.name}" if args.include_ops_by_config else None
    )

    container_build_settings_file = f"/workspace/shared/input/{args.build_settings.name}"

    # enable use of Ctrl-C to stop when running interactively
    docker_run_interactive_args = ["-it"] if sys.stdin.isatty() else []

    docker_container_build_cmd = [
        args.docker_path,
        "run",
        *docker_run_interactive_args,
        f"--name={args.docker_container_name}" if args.docker_container_name is not None else "--rm",
        f"--volume={working_dir}:/workspace/shared",
        args.docker_image_tag,
        "/bin/bash",
        "/workspace/scripts/build.sh",
        args.config,
        "/workspace/shared/output",
        container_build_settings_file,
    ]

    if container_ops_config_file is not None:
        docker_container_build_cmd += [container_ops_config_file]

    run(docker_container_build_cmd)

    print("Finished building Android package at '{}'.".format(output_dir / "aar_out"))


if __name__ == "__main__":
    main()
