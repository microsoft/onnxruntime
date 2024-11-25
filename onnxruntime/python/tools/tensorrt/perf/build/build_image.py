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
import subprocess
import sys
from typing import List, Optional

TRT_DOCKER_FILES = {
    "8.6.cuda_11_8_cudnn_8": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_8_tensorrt8_6",
    "8.6.cuda_12_3_cudnn_9": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda12_3_tensorrt8_6",
    "10.5.cuda_11_8_cudnn_8": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_tensorrt10",
    "10.5.cuda_12_5_cudnn_9": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda12_tensorrt10",
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

    if args.trt_version not in TRT_DOCKER_FILES:
        print(f"[ERROR]: TensorRT version '{args.trt_version}' is currently unsupported", file=sys.stderr)
        sys.exit(1)

    docker_file = TRT_DOCKER_FILES[args.trt_version]
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
        print("[ERROR]: Must specify a valid CUDA version for binary TensorRT installs (e.g., 12.4)", file=sys.stderr)
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
            f"TRT_BINS_DIR={args.trt_bins_dir}",
            "-f",
            f"{docker_file_path}",
            ".",
        ]
    )

    if cmd_ret != 0:
        print(f"[ERROR]: docker build command failed with return code {cmd_ret}", file=sys.stderr)
        sys.exit(1)


def overwrite_onnx_tensorrt_commit_id(commit_id):
    """
    Overwrite onnx-tensorrt commit id in cmake/deps.txt.
    """
    deps_file_path = "../../../../../../cmake/deps.txt"
    line_index = None
    zip_url = None

    with open(deps_file_path) as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith("onnx_tensorrt"):
            parts = line.split(";")
            zip_url = ";".join([parts[0], f"https://github.com/onnx/onnx-tensorrt/archive/{commit_id}.zip", parts[2]])
            line_index = i
            break

    if line_index and zip_url:
        wget_command = f"wget {zip_url.split(';')[1]} -O temp.zip"
        subprocess.run(wget_command, shell=True, check=True)

        sha1sum_command = "sha1sum temp.zip"
        result = subprocess.run(sha1sum_command, shell=True, capture_output=True, text=True, check=True)
        hash_value = result.stdout.split()[0]

        lines[line_index] = zip_url.split(";")[0] + ";" + zip_url.split(";")[1] + ";" + hash_value + "\n"

        with open(deps_file_path, "w") as file:
            file.writelines(lines)

        print(f"Updated deps.txt with new commit id {commit_id} and hash {hash_value}")

        # Verify updated deps.txt
        try:
            with open(deps_file_path) as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("onnx_tensorrt"):
                        print(line.strip())
                        break
        except Exception as e:
            print(f"Failed to read the file: {e}")

        os.remove("temp.zip")
    else:
        print("onnx_tensorrt commit id overwrite failed, entry not found in deps.txt")


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns an object with each argument as a field.

    :return: An object whose fields represent the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo_path", required=True, help="Path to the onnxruntime repository")
    parser.add_argument("-i", "--image_name", required=True, help="The resulting Docker image name")
    parser.add_argument("-b", "--branch", default="main", help="Name of the onnxruntime git branch to checkout")
    parser.add_argument(
        "-t", "--trt_version", default="8.6.cuda_11_8_cudnn_8", help="TensorRT version (e.g., 8.6.cuda_11_8_cudnn_8)"
    )
    parser.add_argument("-a", "--cuda_arch", default="75", help="CUDA architecture (e.g., 75)")
    parser.add_argument("-o", "--oss_parser_commit_id", default="", help="commit id of onnx-tensorrt")

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
        help="CUDA version (e.g., 12.4) used to find TensorRT EA binary tar.gz package",
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
        if args.oss_parser_commit_id != "":
            overwrite_onnx_tensorrt_commit_id(args.oss_parser_commit_id)
        docker_build_trt(args)

    return 0


if __name__ == "__main__":
    main()
