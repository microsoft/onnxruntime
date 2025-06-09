#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import re
import zipfile
from pathlib import Path

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent


def _should_archive(path: Path, accept: re.Pattern | None = None, reject: re.Pattern | None = None) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    posix_path = str(path.as_posix())
    if accept is not None and accept.match(posix_path) is None:
        return False
    if reject is not None and reject.match(posix_path):
        return False
    return True


def archive_android(config: str, qairt_sdk_root: Path) -> None:
    build_root = REPO_ROOT / "build"
    archive_path = build_root / "onnxruntime-tests-android.zip"
    build_dir = build_root / "android"

    archive_path.unlink(missing_ok=True)

    always_reject_patterns = [r".*/__pycache__/.*"]

    ort_reject_patterns = [
        *always_reject_patterns,
        r".*\.(a|cc?|h|py)$",
        r".*/\.ninja_deps$",
        r".*/\.ninja_log$",
        r".*/CMakeCache.txt$",
        r".*/build.ninja$",
        r".*/compile_commands.json$",
        r".*/_deps/(?!onnx-src/onnx/backend/test/).*",
        r".*/CMakeFiles/.*",
        r".*/Google\.Protobuf\.Tools.*/.*",
        r".*/Microsoft\.AI\.DirectML.*/.*",
        r".*/Microsoft\.Windows\.CppWinRT.*/.*",
        r".*/Testing/.*",
        r".*/pkgconfig/.*",
    ]

    qairt_accept_patterns = [".*/aarch64-android/.*", r".*/hexagon-v.+/.*"]

    logging.info(f"Creating test archive in {archive_path}.")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        logging.debug(f"Adding items from ONNX Runtime build in {build_dir}.")
        for filename in build_dir.glob(f"{config}/**/*"):
            if _should_archive(filename, reject=re.compile("|".join(ort_reject_patterns))):
                arcname = filename.relative_to(build_dir)
                archive.write(filename, arcname)

        qdc_test_root = QCOM_ROOT / "scripts" / "linux" / "appium"
        logging.debug(f"Adding QDC test framework from {qdc_test_root}.")
        for filename in qdc_test_root.glob("**/*"):
            if _should_archive(filename, reject=re.compile("|".join(always_reject_patterns))):
                arcname = filename.relative_to(qdc_test_root)
                archive.write(filename, arcname)

        logging.debug(f"Adding QNN libraries from {qairt_sdk_root}.")
        for filename in (qairt_sdk_root / "lib").glob("**/*"):
            if _should_archive(filename, accept=re.compile("|".join(qairt_accept_patterns))):
                arcname = filename.relative_to(qairt_sdk_root)
                archive.write(filename, arcname)


def archive_windows(target_platform: str, config: str, cmake_bin_dir: Path) -> None:
    build_root = REPO_ROOT / "build"
    archive_path = build_root / f"onnxruntime-tests-{target_platform}.zip"
    build_dir = build_root / target_platform

    archive_path.unlink(missing_ok=True)

    logging.info(f"Creating test archive in {archive_path}.")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        logging.debug("Adding individual files.")
        archive.write(build_dir / config / "CTestTestfile.cmake", "CTestTestfile.cmake")
        archive.write(cmake_bin_dir / "ctest.exe", "ctest.exe")
        archive.write(QCOM_ROOT / "scripts" / "windows" / "run_tests.ps1", "run_tests.ps1")

        logging.debug(f"Adding ONNX build from {build_dir / config / config}")
        for filename in (build_dir / config / config).glob("**/*"):
            arcname = str(filename.relative_to(build_dir / config))
            archive.write(filename, arcname)


if __name__ == "__main__":
    log_format = "[%(asctime)s] [archive_tests.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cmake-bin-dir",
        help="The directory containing cmake executables.",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        help="The build configuration to archive",
        default="Release",
        choices=["Debug", "Release", "RelWithDebInfo"],
    )

    parser.add_argument(
        "--target-platform",
        help="The platform for which to package tests.",
        choices=["android", "windows-arm64", "windows-x86_64"],
        required=True,
    )

    parser.add_argument(
        "--qairt-sdk-root",
        help="Path to the QAIRT SDK.",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    if args.target_platform == "android":
        archive_android(args.config, args.qairt_sdk_root)
    elif args.target_platform.startswith("windows-"):
        archive_windows(args.target_platform, args.config, args.cmake_bin_dir)
    else:
        raise ValueError(f"Unknown platform {args.target_platform}.")
