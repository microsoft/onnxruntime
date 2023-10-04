#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

from logger import get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.join(REPO_DIR, "tools", "python"))
from util import run  # noqa: E402

log = get_logger("get_docker_image")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a docker image and push it to a remote Azure Container Registry."
        "The content in the remote registry can be used as a cache when we need to build the thing again."
        "The user must be logged in to the container registry."
    )

    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to the Dockerfile.")
    parser.add_argument("--context", default=".", help="Path to the build context.")
    parser.add_argument(
        "--docker-build-args", default="", help="Arguments that will be passed to the 'docker build' command."
    )

    parser.add_argument(
        "--container-registry",
        help="The Azure container registry name. If not provided, no container registry will be used.",
    )
    parser.add_argument("--repository", required=True, help="The image repository name.")

    parser.add_argument("--use_imagecache", action="store_true", help="use cached image in pipeline cache")

    parser.add_argument("--docker-path", default="docker", help="Path to docker.")

    parser.add_argument("--manylinux-src", default="manylinux", help="Path to manylinux src folder")

    parser.add_argument(
        "--multiple_repos",
        action="store_true",
        help="used in packaging pipeline, which couldn't use get-docker-images-steps.yml",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log.debug(
        "Dockerfile: {}, context: {}, docker build args: '{}'".format(
            args.dockerfile, args.context, args.docker_build_args
        )
    )

    use_container_registry = args.container_registry is not None

    if not use_container_registry:
        log.info("No container registry will be used")

    full_image_name = (
        f"{args.container_registry}.azurecr.io/{args.repository}:latest"
        if use_container_registry
        else f"{args.repository}:latest"
    )

    log.info(f"Image: {full_image_name}")

    dst_deps_file = Path(args.context) / "scripts" / "deps.txt"
    # The docker file may provide a special deps.txt in its docker context dir and uses that one.
    # Otherwise, copy a generic one from this repo's cmake dir.
    if not dst_deps_file.exists():
        log.info(f"Copy deps.txt to : {dst_deps_file}")
        shutil.copyfile(Path(REPO_DIR) / "cmake" / "deps.txt", str(dst_deps_file))

    if "manylinux" in args.dockerfile and args.multiple_repos:
        manylinux_build_scripts_folder = Path(args.manylinux_src) / "docker" / "build_scripts"
        dest = Path(args.context) / "build_scripts"
        if dest.exists():
            log.info(f"Deleting: {dest!s}")
            shutil.rmtree(str(dest))
        shutil.copytree(str(manylinux_build_scripts_folder), str(dest))
        src_entrypoint_file = str(Path(args.manylinux_src) / "docker" / "manylinux-entrypoint")
        dst_entrypoint_file = str(Path(args.context) / "manylinux-entrypoint")
        shutil.copyfile(src_entrypoint_file, dst_entrypoint_file)
        shutil.copymode(src_entrypoint_file, dst_entrypoint_file)
        run(
            "patch",
            "-p1",
            "-i",
            str((Path(SCRIPT_DIR) / "github" / "linux" / "docker" / "manylinux.patch").resolve()),
            cwd=str(dest),
        )

    if use_container_registry:
        run(
            args.docker_path,
            "--log-level",
            "error",
            "buildx",
            "build",
            "--push",
            "--tag",
            full_image_name,
            "--cache-from",
            full_image_name,
            "--build-arg",
            "BUILDKIT_INLINE_CACHE=1",
            *shlex.split(args.docker_build_args),
            "-f",
            args.dockerfile,
            args.context,
        )
    elif args.use_imagecache:
        log.info("Building image with pipeline cache...")
        run(
            args.docker_path,
            "--log-level",
            "error",
            "buildx",
            "build",
            "--tag",
            full_image_name,
            "--cache-from",
            full_image_name,
            "--build-arg",
            "BUILDKIT_INLINE_CACHE=1",
            *shlex.split(args.docker_build_args),
            "-f",
            args.dockerfile,
            args.context,
        )
    else:
        log.info("Building image...")
        run(
            args.docker_path,
            "build",
            "--pull",
            *shlex.split(args.docker_build_args),
            "--tag",
            full_image_name,
            "--file",
            args.dockerfile,
            args.context,
        )

    # tag so we can refer to the image by repository name
    run(args.docker_path, "tag", full_image_name, args.repository)

    return 0


if __name__ == "__main__":
    sys.exit(main())
