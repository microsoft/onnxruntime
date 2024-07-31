#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import sys
import typing

from util import run
from util.android import get_sdk_tool_paths


def run_adb(android_sdk_root: str, args: typing.List[str]):
    sdk_tool_paths = get_sdk_tool_paths(android_sdk_root)
    if is_emulator_running(sdk_tool_paths.adb):
        run(sdk_tool_paths.adb, *args)
    else:
        print("No emulator is running.")

def is_emulator_running(adb_path) -> bool:
    result = run(adb_path, "devices", capture_stdout=True)
    output = result.stdout
    lines = output.strip().split("\n")
    if len(lines) > 1:
        for line in lines[1:]:
            if "emulator" in line:
                return True
    return False


def main():
    logging.getLogger().setLevel(logging.WARNING)

    adb_args = sys.argv[1:]

    android_sdk_root = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if android_sdk_root is None:
        raise RuntimeError(
            "Please provide the Android SDK root with environment variable 'ANDROID_HOME' or "
            "environment variable 'ANDROID_SDK_ROOT'."
        )

    run_adb(android_sdk_root, adb_args)


if __name__ == "__main__":
    main()
