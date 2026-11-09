#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Resolve shared Docker base-image pins before building or calculating cache keys.

With --family alone, stdout is an immutable repository@sha256:digest reference.
With --dockerfile, stdout is one line of shell-quoted Docker build arguments,
preserving supplied arguments and adding the configured BASEIMAGE if needed.
Resolution errors go to stderr with a nonzero exit code and no result on stdout.
"""

import argparse
import json
import re
import shlex
from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent / "github" / "linux" / "docker"
MANIFEST_PATH = DOCKER_DIR / "base-images.json"


def load_manifest():
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    for family, image in manifest["images"].items():
        repository, digest, tag = (image.get(field) for field in ("repository", "digest", "tag"))
        if not isinstance(repository, str) or not re.fullmatch(r"[a-z0-9._/-]+", repository):
            raise ValueError(f"Invalid repository for image family {family!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid digest for image family {family!r}")
        if not isinstance(tag, str) or not re.fullmatch(r"[\w][\w.-]{0,127}", tag, flags=re.ASCII):
            raise ValueError(f"Invalid tag for image family {family!r}")
    for dockerfile, family in manifest["dockerfile_defaults"].items():
        if family not in manifest["images"]:
            raise ValueError(f"Unknown default image family {family!r} for {dockerfile}")
    return manifest


def resolve_family(family, manifest=None):
    if manifest is None:
        manifest = load_manifest()
    if family not in manifest["images"]:
        raise ValueError(f"Unknown Docker base-image family: {family!r}")
    image = manifest["images"][family]
    return f"{image['repository']}@{image['digest']}"


def resolve_build_args(dockerfile, build_args="", family=""):
    arguments = shlex.split(build_args)
    base_images = []
    for index, argument in enumerate(arguments):
        if argument == "--build-arg" and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith("--build-arg="):
            value = argument[len("--build-arg=") :]
        else:
            continue
        if value == "BASEIMAGE" or value.startswith("BASEIMAGE="):
            base_images.append(value)
    if len(base_images) > 1:
        raise ValueError("BASEIMAGE must not be specified more than once")
    if base_images:
        if family:
            raise ValueError("Specify either an image family or an explicit BASEIMAGE, not both")
        if not base_images[0].partition("=")[2]:
            raise ValueError("An explicit BASEIMAGE must have a nonempty value")
        return arguments

    manifest = load_manifest()
    if not family:
        try:
            relative = Path(dockerfile).resolve().relative_to(DOCKER_DIR).as_posix()
        except ValueError:
            relative = ""
        family = manifest["dockerfile_defaults"].get(relative, "")
    if family:
        arguments.extend(["--build-arg", f"BASEIMAGE={resolve_family(family, manifest)}"])
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="", help="Image family from base-images.json.")
    parser.add_argument("--dockerfile", help="Dockerfile whose default family should be used.")
    parser.add_argument("--build-args", default="", help="Additional Docker build arguments.")
    args = parser.parse_args()
    if not args.dockerfile and not args.family:
        parser.error("Specify --dockerfile or --family")
    try:
        if args.dockerfile:
            result = shlex.join(resolve_build_args(args.dockerfile, args.build_args, args.family))
        else:
            result = resolve_family(args.family)
        if "\n" in result or "\r" in result:
            raise ValueError("Resolved Docker arguments must be a single line")
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    else:
        print(result)


if __name__ == "__main__":
    main()
