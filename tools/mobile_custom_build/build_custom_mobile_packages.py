#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Coordinate matching custom Android and iOS ONNX Runtime package builds."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
TOOLS_DIR = SCRIPT_DIR.parent


def run(command: list[str]) -> None:
    """Run a child build command."""

    string_command = [
        str(value)
        for value in command
    ]

    print(
        f"Running command:\n  "
        f"{shlex.join(string_command)}"
    )

    subprocess.run(
        string_command,
        check=True,
    )  # noqa: PLW1510


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matching custom ONNX Runtime packages "
            "for Android and iOS."
        )
    )

    parser.add_argument(
        "working_dir",
        type=pathlib.Path,
        help=(
            "Root working directory under which Android and iOS "
            "build outputs will be stored."
        ),
    )

    parser.add_argument(
        "--platform",
        choices=[
            "android",
            "ios",
            "all",
        ],
        default="all",
        help="Platform to build. Defaults to all.",
    )

    parser.add_argument(
        "--onnxruntime_branch_or_tag",
        default="main",
        help=(
            "ONNX Runtime branch or tag used by both Android and iOS."
        ),
    )

    parser.add_argument(
        "--onnxruntime_repo_url",
        help="Optional alternate ONNX Runtime Git repository.",
    )

    parser.add_argument(
        "--include_ops_by_config",
        type=pathlib.Path,
        help=(
            "Reduced operator configuration shared by Android and iOS."
        ),
    )

    parser.add_argument(
        "--android_build_settings",
        type=pathlib.Path,
        help="Android ONNX Runtime build settings JSON file.",
    )

    parser.add_argument(
        "--ios_build_settings",
        type=pathlib.Path,
        help="iOS ONNX Runtime build settings JSON file.",
    )

    parser.add_argument(
        "--config",
        choices=[
            "Debug",
            "MinSizeRel",
            "Release",
            "RelWithDebInfo",
        ],
        default="Release",
        help="Build configuration shared by both platforms.",
    )

    parser.add_argument(
        "--android_publish",
        action="store_true",
        help="Publish the Android AAR after it is built.",
    )

    parser.add_argument(
        "--android_publish_github_url",
        help="GitHub Packages Maven registry URL.",
    )

    parser.add_argument(
        "--android_publish_group",
        help="Android Maven group ID override.",
    )

    parser.add_argument(
        "--android_publish_artifact_id",
        help="Android Maven artifact ID override.",
    )

    parser.add_argument(
        "--package_version",
        help=(
            "Shared custom package version. Used as the Android Maven "
            "version and as part of the iOS package archive name."
        ),
    )

    parser.add_argument(
        "--ios_binary_url",
        help=(
            "Optional public URL for the generated iOS XCFramework ZIP."
        ),
    )

    parser.add_argument(
        "--ios_source_dir",
        type=pathlib.Path,
        help=(
            "Optional existing ONNX Runtime source checkout for the "
            "iOS build."
        ),
    )

    parser.add_argument(
        "--ios_path_to_protoc_exe",
        type=pathlib.Path,
        help="Optional host protoc executable for the iOS build.",
    )

    parser.add_argument(
        "--ios_dynamic_framework",
        action="store_true",
        help=(
            "Build a dynamic iOS framework instead of the default "
            "static framework."
        ),
    )

    parser.add_argument(
        "--clean_ios",
        action="store_true",
        help="Remove the previous iOS build before rebuilding.",
    )

    args = parser.parse_args()

    if args.platform in ("android", "all"):
        if args.android_build_settings is None:
            parser.error(
                "--android_build_settings is required for Android builds."
            )

        if not args.android_build_settings.is_file():
            parser.error(
                "--android_build_settings is not a file: "
                f"{args.android_build_settings}"
            )

    if args.platform in ("ios", "all"):
        if args.ios_build_settings is None:
            parser.error(
                "--ios_build_settings is required for iOS builds."
            )

        if not args.ios_build_settings.is_file():
            parser.error(
                "--ios_build_settings is not a file: "
                f"{args.ios_build_settings}"
            )

    if (
        args.include_ops_by_config is not None
        and not args.include_ops_by_config.is_file()
    ):
        parser.error(
            "--include_ops_by_config is not a file: "
            f"{args.include_ops_by_config}"
        )

    if args.platform == "all" and sys.platform != "darwin":
        parser.error(
            "--platform all must run on macOS because the iOS build "
            "requires Xcode."
        )

    if args.platform == "ios" and sys.platform != "darwin":
        parser.error(
            "The iOS build must run on macOS with Xcode installed."
        )

    if args.android_publish and not args.android_publish_github_url:
        parser.error(
            "--android_publish requires "
            "--android_publish_github_url."
        )

    return args


def common_args(
    args: argparse.Namespace,
) -> list[str]:
    """Return arguments shared between Android and iOS."""

    values = [
        "--onnxruntime_branch_or_tag",
        args.onnxruntime_branch_or_tag,
        "--config",
        args.config,
    ]

    if args.onnxruntime_repo_url:
        values += [
            "--onnxruntime_repo_url",
            args.onnxruntime_repo_url,
        ]

    if args.include_ops_by_config:
        values += [
            "--include_ops_by_config",
            str(
                args.include_ops_by_config.resolve()
            ),
        ]

    return values


def build_android(
    args: argparse.Namespace,
    working_dir: pathlib.Path,
) -> None:
    """Run the existing Android custom package builder."""

    android_script = (
        TOOLS_DIR
        / "android_custom_build"
        / "build_custom_android_package.py"
    )

    if not android_script.is_file():
        raise RuntimeError(
            "Unable to locate Android custom build script at "
            f"'{android_script}'."
        )

    command = [
        sys.executable,
        str(android_script),
        str(working_dir / "android"),
        "--build_settings",
        str(
            args.android_build_settings.resolve()
        ),
        *common_args(args),
    ]

    if args.android_publish:
        command.append(
            "--publish"
        )

        command += [
            "--publish_github_url",
            args.android_publish_github_url,
        ]

        if args.android_publish_group:
            command += [
                "--publish_group",
                args.android_publish_group,
            ]

        if args.android_publish_artifact_id:
            command += [
                "--publish_artifact_id",
                args.android_publish_artifact_id,
            ]

        if args.package_version:
            command += [
                "--publish_version",
                args.package_version,
            ]

    run(command)


def build_ios(
    args: argparse.Namespace,
    working_dir: pathlib.Path,
) -> None:
    """Run the iOS custom XCFramework builder."""

    ios_script = (
        TOOLS_DIR
        / "ios_custom_build"
        / "build_custom_ios_package.py"
    )

    if not ios_script.is_file():
        raise RuntimeError(
            "Unable to locate iOS custom build script at "
            f"'{ios_script}'."
        )

    package_name = "onnxruntime-ios-custom"

    if args.package_version:
        package_name += (
            f"-{args.package_version}"
        )

    command = [
        sys.executable,
        str(ios_script),
        str(working_dir / "ios"),
        "--build_settings",
        str(
            args.ios_build_settings.resolve()
        ),
        "--package_name",
        package_name,
        *common_args(args),
    ]

    if args.ios_binary_url:
        command += [
            "--binary_url",
            args.ios_binary_url,
        ]

    if args.ios_source_dir:
        command += [
            "--onnxruntime_source_dir",
            str(
                args.ios_source_dir.resolve()
            ),
        ]

    if args.ios_path_to_protoc_exe:
        command += [
            "--path_to_protoc_exe",
            str(
                args.ios_path_to_protoc_exe.resolve()
            ),
        ]

    if args.ios_dynamic_framework:
        command.append(
            "--dynamic_framework"
        )

    if args.clean_ios:
        command.append(
            "--clean"
        )

    run(command)


def write_mobile_metadata(
    args: argparse.Namespace,
    working_dir: pathlib.Path,
) -> None:
    """Write metadata describing the coordinated mobile build."""

    metadata = {
        "platform": args.platform,
        "onnxruntime_branch_or_tag": (
            args.onnxruntime_branch_or_tag
        ),
        "onnxruntime_repo_url": (
            args.onnxruntime_repo_url
        ),
        "configuration": args.config,
        "package_version": args.package_version,
        "operator_config": (
            str(
                args.include_ops_by_config.resolve()
            )
            if args.include_ops_by_config
            else None
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "outputs": {
            "android": (
                "android/output/aar_out"
                if args.platform in ("android", "all")
                else None
            ),
            "ios": (
                "ios/output/ios"
                if args.platform in ("ios", "all")
                else None
            ),
        },
    }

    metadata_path = (
        working_dir
        / "mobile-build-metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    working_dir = args.working_dir.resolve()

    working_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.platform in (
        "android",
        "all",
    ):
        build_android(
            args,
            working_dir,
        )

    if args.platform in (
        "ios",
        "all",
    ):
        build_ios(
            args,
            working_dir,
        )

    write_mobile_metadata(
        args,
        working_dir,
    )

    print(
        "Finished requested mobile builds under "
        f"'{working_dir}'."
    )


if __name__ == "__main__":
    main()
