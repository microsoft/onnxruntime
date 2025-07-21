#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import csv
import logging
import platform
import re
import shutil
from pathlib import Path

from package_manager import DEFAULT_PACKAGE_CACHE_DIR, FileCache

REPO_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_CMAKE_CACHE_DIR = DEFAULT_PACKAGE_CACHE_DIR
DEFAULT_MIRROR_DIR = REPO_ROOT / "mirror"
DEPS_TXT = REPO_ROOT / "cmake" / "deps.txt"


def main(deps_dir: Path, mirror_dir: Path) -> None:
    deps = _parse_deps_txt()
    cache = FileCache(deps_dir)

    logging.info(f"Downloading dependencies into {deps_dir}")
    cached_paths: list[tuple[str, Path]] = []
    for name, url, sha1 in deps:
        cached_paths.append((url, cache.fetch(name, url, expected_sha1=sha1)))
        logging.debug(f"{name} --> {cached_paths[-1]}")

    logging.info(f"Making dependencies available in {mirror_dir}")
    if mirror_dir.exists():
        shutil.rmtree(mirror_dir)

    https = re.compile(r"^https://")
    for url, cached_path in cached_paths:
        dest = Path(https.sub(str(mirror_dir.as_posix()) + "/", url))
        logging.debug(f"{dest} --> {cached_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if platform.uname().system == "Windows":
            shutil.copyfile(cached_path, dest)
        else:
            dest.symlink_to(cached_path)


def _parse_deps_txt() -> list[tuple[str, str, str]]:
    with DEPS_TXT.open() as deps_file:
        return [(r[0], r[1], r[2]) for r in csv.reader(deps_file, delimiter=";") if not r[0].startswith("#")]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deps-dir",
        "-d",
        type=Path,
        default=DEFAULT_CMAKE_CACHE_DIR,
        help="Directory into which to download CMake dependencies",
    )
    parser.add_argument(
        "--mirror-dir",
        "-m",
        type=Path,
        default=DEFAULT_MIRROR_DIR,
        help="Directory into which to copy all dependencies.",
    )

    return parser


if __name__ == "__main__":
    log_format = "[%(asctime)s] [fetch_cmake_deps.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)

    parser = make_parser()
    args = parser.parse_args()

    main(args.deps_dir, args.mirror_dir)
