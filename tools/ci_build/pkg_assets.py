#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import platform
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from util import get_logger, is_windows

log = get_logger("pkg_assets")


def parse_version_number(source_dir):
    """
    Parse the VERSION_NUMBER file to get the ORT version.

    Args:
        source_dir: Path to source directory containing VERSION_NUMBER file

    Returns:
        str: Version string (e.g., "1.23.0") or None if not found
    """
    version_file = os.path.join(source_dir, "VERSION_NUMBER")
    try:
        with open(version_file) as f:
            version_line = f.readline().strip()
            # Extract version using regex to handle any extra whitespace/content
            version_match = re.match(r"(\d+\.\d+\.\d+)", version_line)
            if version_match:
                return version_match.group(1)
            else:
                log.warning(f"Could not parse version from VERSION_NUMBER: {version_line}")
                return None
    except (OSError, FileNotFoundError) as e:
        log.warning(f"Could not read VERSION_NUMBER file: {e}")
        return None


def get_qnn_asset_file_list():
    """
    Returns the list of QNN asset files to include in the zip package.

    Args:
        is_windows_platform: If None, auto-detect from platform.system()

    Returns:
        list[str]: List of filenames to include
    """
    qnn_assets = {
        "windows": [
            "onnxruntime_providers_qnn_abi.dll",
            "Genie.dll",
            "QnnCpu.dll",
            "QnnGpu.dll",
            "QnnHtp.dll",
            "QnnIr.dll",
            "QnnSaver.dll",
            "QnnSystem.dll",
            "QnnHtpPrepare.dll",
            "QnnHtpV81Stub.dll",
            "libQnnHtpV81Skel.so",
            "libqnnhtpv81.cat",
            "QnnHtpV73Stub.dll",
            "libQnnHtpV73Skel.so",
            "libqnnhtpv73.cat",
            "QnnHtpV68Stub.dll",
            "libQnnHtpV68Skel.so",
        ],
        "others": [
            "libonnxruntime_providers_qnn_abi.so",
            "libGenie.so",
            "libQnnCpu.so",
            "libQnnGpu.so",
            "libQnnHtp.so",
            "libQnnHtpPrepare.so",
            "libQnnHtpV68Skel.so",
            "libQnnHtpV68Stub.so",
            "libQnnIr.so",
            "libQnnSaver.so",
            "libQnnSystem.so",
            "libHtpPrepare.so",
        ],
    }
    return qnn_assets["windows"] if is_windows() else qnn_assets["others"]


def build_zip_asset(
    source_dir,
    build_dir,
    configs,
    zip_name_suffix=None,
    use_ninja=False,
):
    """
    Build zip asset packages containing QNN EP and dependencies.

    Args:
        source_dir: Path to source directory
        build_dir: Path to build directory
        configs: List of build configurations (e.g., ['RelWithDebInfo'])
        zip_name_suffix: Optional suffix for zip filename
        use_ninja: Whether Ninja generator was used

    Returns:
        list[Path]: List of created zip file paths
    """
    created_zips = []

    for config in configs:
        log.info(f"Building zip asset for {config} configuration")

        # Determine working directory (matching build_python_wheel logic)
        config_build_dir = os.path.join(build_dir, config)
        if is_windows() and not use_ninja:
            cwd = os.path.join(config_build_dir, config)
        else:
            cwd = config_build_dir

        if not os.path.exists(cwd):
            raise FileNotFoundError(f"Build directory not found: {cwd}")

        # Create dist directory
        dist_dir = os.path.join(cwd, "dist")
        os.makedirs(dist_dir, exist_ok=True)

        # Generate zip filename
        platform_name = platform.system().lower()
        platform_abbr = {"windows": "win"}
        if platform_name in platform_abbr:
            platform_name = platform_abbr[platform_name]
        arch = platform.machine().lower()
        if arch == "amd64":
            arch = "x64"
        elif arch == "x86_64":
            arch = "x64"

        # Parse version from VERSION_NUMBER file
        version = parse_version_number(source_dir)

        zip_name = f"onnxruntime-qnn-{platform_name}-{arch}"
        if zip_name_suffix:
            zip_name += f"-{zip_name_suffix}"
        if version:
            zip_name += f"-{version}"
        zip_name += f"-{config}.zip"

        zip_path = Path(dist_dir) / zip_name

        # Get list of files to include
        asset_files = get_qnn_asset_file_list()

        # Collect and verify files exist
        missing_files = []
        found_files = []

        for filename in asset_files:
            file_path = os.path.join(cwd, filename)
            if os.path.exists(file_path):
                found_files.append((filename, file_path))
                log.debug(f"Found asset file: {file_path}")
            else:
                missing_files.append(filename)

        if missing_files:
            log.warning(f"Missing asset files in {cwd}:")
            for missing in missing_files:
                log.warning(f"  - {missing}")
            log.warning("Continuing with available files...")

        if not found_files:
            raise FileNotFoundError(f"No asset files found in {cwd}. Expected files: {asset_files}")

        # Create zip file
        log.info(f"Creating zip: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for filename, file_path in found_files:
                # TODO: we will remove this once the rename completed.
                zip_filename = filename
                if "onnxruntime_providers_qnn_abi" in filename:
                    zip_filename = filename.replace("_abi", "")

                zipf.write(file_path, zip_filename)
                log.debug(f"Added to zip: {zip_filename}")

        log.info(f"Created zip asset: {zip_path} ({len(found_files)} files)")
        created_zips.append(zip_path)

    return created_zips


def main():
    """
    Main entry point for standalone execution of pkg_assets.py
    """
    parser = argparse.ArgumentParser(
        description="Build QNN asset zip packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pkg_assets.py --source_dir . --build_dir build --config RelWithDebInfo
  python pkg_assets.py --source_dir . --build_dir build --config Debug --config Release --suffix custom
        """,
    )

    parser.add_argument("--source_dir", required=True, help="Path to source directory containing VERSION_NUMBER file")

    parser.add_argument("--build_dir", required=True, help="Path to build directory containing compiled assets")

    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Build configuration(s) to package (e.g., RelWithDebInfo, Debug). Can be specified multiple times.",
    )

    parser.add_argument("--suffix", help="Optional suffix for zip filename")

    parser.add_argument("--use_ninja", action="store_true", help="Whether Ninja generator was used for build")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Default to RelWithDebInfo if no configs specified
    if not args.config:
        args.config = ["RelWithDebInfo"]

    try:
        created_zips = build_zip_asset(
            source_dir=args.source_dir,
            build_dir=args.build_dir,
            configs=args.config,
            zip_name_suffix=args.suffix,
            use_ninja=args.use_ninja,
        )

        print(f"Successfully created {len(created_zips)} zip package(s):")
        for zip_path in created_zips:
            print(f"  {zip_path}")

    except Exception as e:
        log.error(f"Failed to create zip packages: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
