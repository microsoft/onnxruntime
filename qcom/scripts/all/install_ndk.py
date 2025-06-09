#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import platform
import subprocess
from pathlib import Path

BUILD_TOOLS_VERSION = "34.0.0"
NDK_VERSION = "26.2.11394342"
PLATFORM_VERSION = "29"


def main(cli_tools_root: Path) -> None:
    sdk_root = cli_tools_root.parent
    ndk_root = sdk_root / "ndk" / NDK_VERSION

    if ndk_root.exists():
        logging.info(f"NDK found in {ndk_root}.")
    else:
        logging.info(f"Installing NDK into {ndk_root}.")
        sdkmanager_exe = "sdkmanager.bat" if platform.uname().system == "Windows" else "sdkmanager"
        sdkmanager = cli_tools_root / "bin" / sdkmanager_exe
        say_yes(
            [
                str(sdkmanager),
                f"--sdk_root={sdk_root}",
                "--install",
                f"build-tools;{BUILD_TOOLS_VERSION}",
                f"ndk;{NDK_VERSION}",
                f"platforms;android-{PLATFORM_VERSION}",
            ]
        )
        say_yes(
            [
                str(sdkmanager),
                f"--sdk_root={sdk_root}",
                "--licenses",
            ]
        )
    print(str(ndk_root))


def say_yes(cmd: list[str]) -> None:
    logging.debug(f"Saying yes to {cmd}")
    subprocess.run(cmd, input=("y\n" * 64).encode("utf-8"), capture_output=True, check=True)


if __name__ == "__main__":
    log_format = "[%(asctime)s] [install_ndk.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli-tools-root",
        required=True,
        help="Location of Android commandlinetools install.",
        type=Path,
    )

    args = parser.parse_args()
    main(args.cli_tools_root)
