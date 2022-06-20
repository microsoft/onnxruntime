#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import collections
import hashlib
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


FileInfo = collections.namedtuple("FileInfo", ["path", "mode"])


def file_info_str(file_info: FileInfo):
    return "{} {}".format(file_info.path, file_info.mode)


def make_file_info_from_path(file_path: str):
    return FileInfo(file_path, os.stat(file_path).st_mode)


def update_hash_with_directory(dir_file_info: FileInfo, hash_obj):
    hash_obj.update(file_info_str(dir_file_info).encode())

    files, dirs = [], []
    for dir_entry in os.scandir(dir_file_info.path):
        file_info = FileInfo(dir_entry.path, dir_entry.stat().st_mode)
        if dir_entry.is_dir():
            dirs.append(file_info)
        elif dir_entry.is_file():
            files.append(file_info)

    def file_info_key(file_info: FileInfo):
        return file_info.path

    files.sort(key=file_info_key)
    dirs.sort(key=file_info_key)

    for file_info in files:
        update_hash_with_file(file_info, hash_obj)

    for file_info in dirs:
        update_hash_with_directory(file_info, hash_obj)


def update_hash_with_file(file_info: FileInfo, hash_obj):
    hash_obj.update(file_info_str(file_info).encode())

    read_bytes_length = 8192
    with open(file_info.path, mode="rb") as file_data:
        while True:
            read_bytes = file_data.read(read_bytes_length)
            if len(read_bytes) == 0:
                break
            hash_obj.update(read_bytes)


def generate_tag(dockerfile_path, context_path, docker_build_args_str):
    hash_obj = hashlib.sha256()
    hash_obj.update(docker_build_args_str.encode())
    update_hash_with_file(make_file_info_from_path(dockerfile_path), hash_obj)
    update_hash_with_directory(make_file_info_from_path(context_path), hash_obj)
    return "image_content_digest_{}".format(hash_obj.hexdigest())


def container_registry_has_image(full_image_name, docker_path):
    env = os.environ.copy()
    env["DOCKER_CLI_EXPERIMENTAL"] = "enabled"  # needed for "docker manifest"
    proc = run(docker_path, "manifest", "inspect", "--insecure", full_image_name, env=env, check=False, quiet=True)
    image_found = proc.returncode == 0
    log.debug("Image {} in registry".format("found" if image_found else "not found"))
    return image_found


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

    tag = generate_tag(args.dockerfile, args.context, args.docker_build_args)

    full_image_name = (
        "{}.azurecr.io/{}:{}".format(args.container_registry, args.repository, tag)
        if use_container_registry
        else "{}:{}".format(args.repository, tag)
    )

    log.info("Image: {}".format(full_image_name))

    if use_container_registry and container_registry_has_image(full_image_name, args.docker_path):
        log.info("Pulling image...")
        run(args.docker_path, "pull", full_image_name)
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

        if use_container_registry:
            # avoid pushing if an identically tagged image has been pushed since the last check
            # there is still a race condition, but this reduces the chance of a redundant push
            if not container_registry_has_image(full_image_name, args.docker_path):
                log.info("Pushing image...")
                run(args.docker_path, "push", full_image_name)
            else:
                log.info("Image now found, skipping push")

    # tag so we can refer to the image by repository name
    run(args.docker_path, "tag", full_image_name, args.repository)

    return 0


if __name__ == "__main__":
    sys.exit(main())
