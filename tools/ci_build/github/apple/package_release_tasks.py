#!/usr/bin/env python

# Note: This script is intended to be called from the CocoaPods package release pipeline or a similar context.

import argparse
import glob
import os
import shlex
import subprocess
from enum import Enum
from pathlib import Path


class Task(Enum):
    upload_pod = 1
    update_podspec = 2


def _run(command: list[str], **kwargs):
    print(f"Running command: {shlex.join(command)}")
    kwargs.setdefault("check", True)
    return subprocess.run(command, **kwargs)


def upload_pod(pod_archive_path: Path):
    env = os.environ.copy()
    env.update(
        {
            # configure azcopy to use managed identity
            "AZCOPY_AUTO_LOGIN_TYPE": "MSI",
            "AZCOPY_MSI_CLIENT_ID": "63b63039-6328-442f-954b-5a64d124e5b4",
        }
    )

    storage_account_name = "onnxruntimepackages"
    storage_account_container_name = "$web"
    dest_url = f"https://{storage_account_name}.blob.core.windows.net/{storage_account_container_name}/"

    upload_command = ["azcopy", "cp", str(pod_archive_path), dest_url]

    _run(upload_command, env=env)


def update_podspec(pod_archive_path: Path, podspec_path: Path):
    storage_url = f"https://download.onnxruntime.ai/{pod_archive_path.name}"

    podspec_content = podspec_path.read_text()
    podspec_content = podspec_content.replace("file:///http_source_placeholder", storage_url)
    podspec_path.write_text(podspec_content)


def _resolve_single_path_from_prefix(prefix: Path) -> Path:
    pattern = glob.escape(str(prefix)) + "*"
    matching_path_strs = glob.glob(pattern)
    num_matches = len(matching_path_strs)
    assert num_matches == 1, f"Expected exactly one match for pattern: '{pattern}'. Got {num_matches}."
    return Path(matching_path_strs[0]).resolve(strict=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Helper script to perform release tasks. "
        "Mostly useful for the CocoaPods package release pipeline.",
    )

    parser.add_argument(
        "--pod-archive-path-prefix",
        type=Path,
        help="Pod archive path prefix. There should be exactly one file with this prefix.",
    )

    parser.add_argument(
        "--podspec-path",
        type=Path,
        help="Podspec path.",
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
        assert args.pod_archive_path_prefix is not None, "--pod-archive-path-prefix must be specified."
        args.pod_archive_path = _resolve_single_path_from_prefix(args.pod_archive_path_prefix)

    if require_podspec_path:
        assert args.podspec_path is not None, "--podspec-path must be specified."
        args.podspec_path = args.podspec_path.resolve(strict=True)


def main():
    args = _parse_args()

    task = Task[args.task]

    if task == Task.update_podspec:
        _validate_args(args, require_pod_archive_path=True, require_podspec_path=True)
        update_podspec(args.pod_archive_path, args.podspec_path)

    elif task == Task.upload_pod:
        _validate_args(args, require_pod_archive_path=True)
        upload_pod(args.pod_archive_path)


if __name__ == "__main__":
    main()
