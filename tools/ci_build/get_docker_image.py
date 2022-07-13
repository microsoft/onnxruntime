#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shlex
import sys

from logger import get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(os.path.join(REPO_DIR, "tools", "python"))


from util import run  # noqa: E402

log = get_logger("get_docker_image")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gets a docker image, either by pulling it from a "
        "container registry or building it locally and then pushing it. "
        "The uniqueness of the docker image is determined by a hash digest of "
        "the Dockerfile, the build context directory, and arguments to "
        "'docker build' affecting the image content. "
        "This digest value is used in the image tag. "
        "This script checks whether an image with that tag is initially "
        "present in the container registry to determine whether to pull or "
        "build the image. "
        "The user must be logged in to the container registry."
    )

    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to the Dockerfile.")
    parser.add_argument("--context", default=".", help="Path to the build context.")
    parser.add_argument(
        "--docker-build-args",
        default="",
        help="String of Docker build args which may affect the image content. "
        "These will be used in differentiating images from one another. "
        "For example, '--build-arg'.",
    )
    parser.add_argument(
        "--docker-build-args-not-affecting-image-content",
        default="",
        help="String of Docker build args which do not affect the image " "content.",
    )

    parser.add_argument(
        "--container-registry",
        help="The Azure container registry name. " "If not provided, no container registry will be used.",
    )
    parser.add_argument("--repository", required=True, help="The image repository name.")

    parser.add_argument("--docker-path", default="docker", help="Path to docker.")

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
        "{}.azurecr.io/{}:latest".format(args.container_registry, args.repository)
        if use_container_registry
        else "{}:latest".format(args.repository)
    )

    log.info("Image: {}".format(full_image_name))

    if use_container_registry:
        run("az", "login", "--identity", "--username", "d1c41439-486c-4110-9052-d97199c8aa78")
        run("az", "acr", "login", "-n", args.container_registry)
        run(
            args.docker_path,
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
    else:
        log.info("Building image...")
        run(
            args.docker_path,
            "build",
            "--pull",
            *shlex.split(args.docker_build_args),
            *shlex.split(args.docker_build_args_not_affecting_image_content),
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
