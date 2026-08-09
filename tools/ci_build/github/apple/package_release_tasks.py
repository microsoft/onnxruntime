#!/usr/bin/env python3

# Note: This script is intended to be called from the CocoaPods package release pipeline or a similar context.

from __future__ import annotations

import argparse
import glob
import shlex
import subprocess
from enum import Enum
from pathlib import Path


class Task(Enum):
    upload_pod_archive = 1
    update_podspec = 2


def _run(command: list[str], **kwargs):
    print(f"Running command: {shlex.join(command)}", flush=True)
    kwargs.setdefault("check", True)
    return subprocess.run(command, **kwargs)  # noqa: PLW1510  # we add 'check' to kwargs if not present


def upload_pod_archive(pod_archive_path: Path):
    storage_account_name = "onnxruntimepackages"
    storage_account_container_name = "$web"
    dest_url = f"https://{storage_account_name}.blob.core.windows.net/{storage_account_container_name}/"

    upload_command = ["azcopy", "cp", str(pod_archive_path), dest_url, "--overwrite", "false"]

    _run(upload_command)


def update_podspec(pod_archive_path: Path, podspec_path: Path):
    storage_url = f"https://download.onnxruntime.ai/{pod_archive_path.name}"

    podspec_content = podspec_path.read_text()
    podspec_content = podspec_content.replace("file:///http_source_placeholder", storage_url)
    podspec_path.write_text(podspec_content)


def _resolve_single_path_from_pattern(path_pattern: str) -> Path:
    matches = glob.glob(path_pattern)
    if len(matches) != 1:
        raise argparse.ArgumentTypeError(
            f"Expected exactly 1 match for pattern '{path_pattern}' but got {len(matches)} matches."
        )
    return Path(matches[0]).resolve(strict=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Helper script to perform release tasks. Mostly useful for the CocoaPods package release pipeline.",
    )

    parser.add_argument(
        "--pod-archive-path",
        type=_resolve_single_path_from_pattern,
        help="Pod archive path. It may be a pattern, in which case it must match exactly one path.",
    )

    parser.add_argument(
        "--podspec-path",
        type=_resolve_single_path_from_pattern,
        help="Podspec path. It may be a pattern, in which case it must match exactly one path.",
    )

    parser.add_argument(
        "task",
        choices=[task.name for task in Task],
        help="Specify the task to run.",
    )

    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace, require_pod_archive_path: bool = False, require_podspec_path: bool = False
):
    if require_pod_archive_path:
        assert args.pod_archive_path is not None, "--pod-archive-path must be specified."

    if require_podspec_path:
        assert args.podspec_path is not None, "--podspec-path must be specified."


def main():
    args = _parse_args()

    task = Task[args.task]

    if task == Task.update_podspec:
        _validate_args(args, require_pod_archive_path=True, require_podspec_path=True)
        update_podspec(args.pod_archive_path, args.podspec_path)

    elif task == Task.upload_pod_archive:
        _validate_args(args, require_pod_archive_path=True)
        upload_pod_archive(args.pod_archive_path)


if __name__ == "__main__":
    main()
