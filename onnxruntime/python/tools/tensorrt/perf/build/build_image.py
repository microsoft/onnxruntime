# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Builds an Ubuntu-based Docker image with TensorRT.
"""

import argparse
import os
import pty
import shlex
import sys
from typing import List, Optional

TRT_DOCKER_FILES = {
    "8.4": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_6_tensorrt8_4",
    "8.5": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_8_tensorrt8_5",
    "8.6": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_8_tensorrt8_6",
    "BIN": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_tensorrt_bin",
}


def run_cmd(cmd: List[str]) -> Optional[int]:
    """
    Runs a shell command and returns the process's return code.

    :param cmd: List of command strings.

    :return: The return code.
    """

    escaped_cmd = " ".join(map(shlex.quote, cmd))
    print(f"[CMD] {escaped_cmd}\n")

    return pty.spawn(cmd)


def get_common_docker_build_args(args: argparse.Namespace) -> List[str]:
    """
    Returns a list of common 'docker build' command-line arguments/options.

    :param args: Arguments to the script.

    :return: A list of common 'docker build' arguments.
    """

    command = [
        "--no-cache",
        "-t",
        f"{args.image_name}",
        "--build-arg",
        f"CMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        "--build-arg",
        f"ONNXRUNTIME_BRANCH={args.branch}",
    ]
    if args.use_tensorrt_oss_parser:
        command.extend(
            [
                "--build-arg",
                "PARSER_CONFIG=--use_tensorrt_oss_parser",
            ]
        )
    return command


def is_valid_ver_str(version: str, min_comps: int = 0, max_comps: int = 0) -> bool:
    """
    Returns a boolean indicating if the string argument is a 'valid' version string.

    :param version: String representing a version (e.g., 8.2.3).
    :param min_comps: The minimum number of expected version components.
                      Set to 0 to ignore.
    :param max_comps: The maximum number of expected version components.
                      Set to 0 to ignore.

    :return: True if the string is valid (i.e., positive integers separated by dots)
    """

    if not version:
        return False

    ver_nums = version.split(".")
    num_comps = len(ver_nums)

    if num_comps < min_comps:
        return False

    if num_comps > max_comps > 0:
        return False

    return all(num.isdecimal() for num in ver_nums)


def docker_build_trt(args: argparse.Namespace):
    """
    Builds a Docker image that installs TensorRT from a public repository.

    :param args: The arguments to this script.
    """

    if not is_valid_ver_str(args.trt_version, min_comps=2, max_comps=4):
        print(f"[ERROR]: Invalid TensorRT version '{args.trt_version}'", file=sys.stderr)
        sys.exit(1)

    vers_comps = args.trt_version.split(".")
    trt_ver_key = f"{vers_comps[0]}.{vers_comps[1]}"

    if trt_ver_key not in TRT_DOCKER_FILES:
        print(f"[ERROR]: TensorRT version '{args.trt_version}' is currently unsupported", file=sys.stderr)
        sys.exit(1)

    docker_file = TRT_DOCKER_FILES[trt_ver_key]
    docker_file_path = os.path.normpath(os.path.join(args.repo_path, docker_file))

    if not os.path.isfile(docker_file_path):
        print(f"[ERROR]: Invalid docker file path '{docker_file_path}'", file=sys.stderr)
        sys.exit(1)

    common_args = get_common_docker_build_args(args)
    cmd_ret = run_cmd(["docker", "build", *common_args, "-f", f"{docker_file_path}", "."])

    if cmd_ret != 0:
        print(f"[ERROR]: docker build command failed with return code {cmd_ret}", file=sys.stderr)
        sys.exit(1)


def docker_build_trt_bin(args: argparse.Namespace):
    """
    Builds a Docker image that installs TensorRT from a tar.gz package containing binaries.
    See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar

    :param args: The arguments to this script.
    """

    docker_file = TRT_DOCKER_FILES["BIN"]
    docker_file_path = os.path.normpath(os.path.join(args.repo_path, docker_file))

    if not is_valid_ver_str(args.trt_version, 4, 4):
        print(
            "[ERROR]: Must specify a valid TensorRT version for binary TensorRT installs (e.g., 8.x.x.x)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not is_valid_ver_str(args.tar_cuda_version, 2, 2):
        print("[ERROR]: Must specify a valid CUDA version for binary TensorRT installs (e.g., 11.x)", file=sys.stderr)
        sys.exit(1)

    if not is_valid_ver_str(args.tar_cudnn_version, 2, 2):
        print("[ERROR]: Must specify a valid cuDNN version for binary TensorRT installs (e.g., 8.x)", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(docker_file_path):
        print(f"[ERROR]: Invalid docker file path '{docker_file_path}'", file=sys.stderr)
        sys.exit(1)

    if not args.trt_bins_dir or not os.path.isdir(args.trt_bins_dir):
        print(f"[ERROR]: Invalid TensorRT bin directory '{args.trt_bins_dir}'", file=sys.stderr)
        sys.exit(1)

    common_args = get_common_docker_build_args(args)
    cmd_ret = run_cmd(
        [
            "docker",
            "build",
            *common_args,
            "--build-arg",
            f"TAR_TRT_VERSION={args.trt_version}",
            "--build-arg",
            f"TAR_CUDA_VERSION={args.tar_cuda_version}",
            "--build-arg",
            f"TAR_CUDNN_VERSION={args.tar_cudnn_version}",
            "--build-arg",
            f"TRT_BINS_DIR={args.trt_bins_dir}",
            "-f",
            f"{docker_file_path}",
            ".",
        ]
    )

    if cmd_ret != 0:
        print(f"[ERROR]: docker build command failed with return code {cmd_ret}", file=sys.stderr)
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns an object with each argument as a field.

    :return: An object whose fields represent the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo_path", required=True, help="Path to the onnxruntime repository")
    parser.add_argument("-i", "--image_name", required=True, help="The resulting Docker image name")
    parser.add_argument("-b", "--branch", default="main", help="Name of the onnxruntime git branch to checkout")
    parser.add_argument("-t", "--trt_version", default="8.6.1.6", help="TensorRT version (e.g., 8.6.1.6)")
    parser.add_argument("-a", "--cuda_arch", default="75", help="CUDA architecture (e.g., 75)")

    # Command-line options for installing TensorRT from binaries.
    parser.add_argument(
        "--install_bin",
        action="store_true",
        default=False,
        help="Enable to install TensorRT from tar.gz binary package",
    )
    parser.add_argument(
        "--tar_cuda_version",
        default="",
        help="CUDA version (e.g., 11.8) used to find TensorRT EA binary tar.gz package",
    )
    parser.add_argument(
        "--tar_cudnn_version",
        default="",
        help="CUDA version (e.g., 8.6) used to find TensorRT EA binary tar.gz package",
    )
    parser.add_argument("--trt_bins_dir", default="", help="Directory containing TensorRT tar.gz package")
    parser.add_argument(
        "--use_tensorrt_oss_parser",
        action="store_true",
        default=False,
        help="Use TensorRT OSS Parser",
    )

    return parser.parse_args()


def main() -> int:
    """
    Script entry point. Builds an Ubuntu-based Docker image with TensorRT.

    :return: 0 on success, 1 on error.
    """

    args = parse_arguments()

    if args.install_bin:
        docker_build_trt_bin(args)
    else:
        docker_build_trt(args)

    return 0


if __name__ == "__main__":
    main()
