#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys

from util import get_android_sdk_tool_paths, running_android_emulator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sets up and runs an Android emulator.")
    parser.add_argument(
        "--android-sdk-root", required=True, help="Path to the Android SDK root.")
    parser.add_argument(
        "--system-image", default="system-images;android-29;google_apis;x86_64",
        help="The Android system image package name.")
    args = parser.parse_args()

    sdk_paths = get_android_sdk_tool_paths(args.android_sdk_root)

    with running_android_emulator(
            sdk_tool_paths=sdk_paths,
            system_image_package_name=args.system_image):
        try:
            print("** Press Enter or CTRL+C to close. **")
            sys.stdin.readline()
        except KeyboardInterrupt:
            pass
