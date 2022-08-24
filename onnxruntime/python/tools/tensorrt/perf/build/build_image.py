# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from typing import List
import os
import shlex
import subprocess
import sys


def run_cmd(cmd):
    """
    Runs a shell command and returns the process's return code.

    :param cmd: List of command strings.

    :return: The return code.
    """

    print("[CMD] %s" % " ".join(map(shlex.quote, cmd)))
    return 0
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")

    while True:
        lines = proc.stdout.readlines()

        if not lines and proc.poll() is not None:
            break

        if lines:
            sys.stdout.writelines(lines)

    return proc.poll()
    """


def get_common_docker_build_args(args: argparse.Namespace) -> List[str]:
    """
    Returns a list of common 'docker build' command-line arguments/options.

    :param args: Arguments to the script.

    :return: A list of common 'docker build' arguments.
    """

    return ["--no-cache", "-t", f"ort-{args.branch}", "--build-arg", f"CMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
            "--build-arg", f"ONNXRUNTIME_BRANCH={args.branch}"]


TRT_DOCKER_FILES = {
    "8.0": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_4_tensorrt8_0",
    "8.2": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_4_tensorrt8_2",
    "8.4": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_6_tensorrt8_4",
    "BIN": "tools/ci_build/github/linux/docker/Dockerfile.ubuntu_tensorrt_BIN"
}

def docker_build_trt(args: argparse.Namespace):
    docker_file = TRT_DOCKER_FILES[args.trt_version]
    docker_file_path = os.path.normpath(os.path.join(args.repo_path, docker_file))

    if not os.path.isfile(docker_file_path):
        print("[ERROR]: Invalid docker file path '%s'" % str(docker_file_path), file=sys.stderr)
        exit(1)

    common_args = get_common_docker_build_args(args)
    #cmd_ret = run_cmd(["docker", "build", *common_args, "-f", f"{docker_file_path}", "."])
    cmd_ret = run_cmd(["cat", "build_image.py"])

    if cmd_ret != 0:
        print("[ERROR]: docker build command failed with return code %d" % cmd_ret)
        exit(1)


def docker_build_trt_bin(args: argparse.Namespace):
    docker_file = TRT_DOCKER_FILES["BIN"]
    docker_file_path = os.path.normpath(os.path.join(args.repo_path, docker_file))

    if not args.cuda_version:
        print("[ERROR]: Must specify CUDA version for binary TensorRT installs", file=sys.stderr)
        exit(1)

    if not args.cudnn_version:
        print("[ERROR]: Must specify cuDNN version for binary TensorRT installs", file=sys.stderr)
        exit(1)

    if not os.path.isfile(docker_file_path):
        print("[ERROR]: Invalid docker file path '%s'" % str(docker_file_path), file=sys.stderr)
        exit(1)

    common_args = get_common_docker_build_args(args)
    cmd_ret = run_cmd(["docker", "build", *common_args, "--build-arg", f"TRT_VERSION={args.trt_version}",
                       "--build-arg", "ARCH=x86_64", "--build-arg", f"CUDA_VERSION={args.cuda_version}",
                       "--build-arg", f"CUDNN_VERSION={args.cudnn_version}",
                       "-f", f"{docker_file_path}", "."])
    #cmd_ret = run_cmd(["cat", "build_image.py"])

    if cmd_ret != 0:
        print("[ERROR]: docker build command failed with return code %d" % cmd_ret)
        exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments and returns an object with each argument as a field.

    :return: An object whose fields represent the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo_path", required=True, help="Path the onnxruntime repository")
    parser.add_argument("-b", "--branch", default="main", help="Name of the git branch to checkout")
    parser.add_argument("-t", "--trt_version", default="8.4.1.5", help="TensorRT version (e.g., 8.4.1.5)")
    parser.add_argument("-a", "--cuda_arch", default="75", help="CUDA architecture")

    # Command-line options for installing TensorRT from binaries.
    parser.add_argument("--install_bin", action="store_true", default=False,
                        help="Enable to install TensorRT from tar.gz binary package")
    parser.add_argument("--cuda_version", help="CUDA version (e.g., 11.8) used to find TensorRT EA binary tar.gz package")
    parser.add_argument("--cudnn_version", help="CUDA version (e.g., 8.6) used to find TensorRT EA binary tar.gz package")

    return parser.parse_args()


def trt_major_minor_version(full_version: str) -> str:
    if not full_version:
        return ""

    ver_nums = full_version.split(".")
    ver_nums_len = len(ver_nums)

    if ver_nums_len < 2 or ver_nums_len > 4:
        return ""

    return f"{ver_nums[0]}.{ver_nums[1]}"


def main() -> int:
    """
    Script entry point.

    :return: 0 on success, 1 on error.
    """

    args = parse_arguments()
    trt_version = trt_major_minor_version(args.trt_version)

    if not trt_version:
        print("[ERROR]: Invalid TensorRT version '%s'" % args.trt_version, file=sys.stderr)
        exit(1)

    if not args.install_bin and trt_version not in TRT_DOCKER_FILES:
        print("[ERROR]: TensorRT version '%s' is currently unsupported" % args.trt_version, file=sys.stderr)
        exit(1)

    if args.install_bin:
        docker_build_trt_bin(args)
    else:
        docker_build_trt(args)

    return 0

if __name__ == "__main__":
    main()

