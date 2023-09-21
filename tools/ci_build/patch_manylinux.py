#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil
import sys
from pathlib import Path

from logger import get_logger

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.join(REPO_DIR, "tools", "python"))
from util import run  # noqa: E402

log = get_logger("patch_manylinux")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a docker image and push it to a remote Azure Container Registry."
        "The content in the remote registry can be used as a cache when we need to build the thing again."
        "The user must be logged in to the container registry."
    )

    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to the Dockerfile.")
    parser.add_argument("--context", default=".", help="Path to the build context.")
    parser.add_argument("--manylinux-src", default="manylinux", help="Path to manylinux src folder")

    return parser.parse_args()


def main():
    args = parse_args()

    log.debug(f"Dockerfile: {args.dockerfile}, context: {args.context}")

    if "manylinux" in args.dockerfile:
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
