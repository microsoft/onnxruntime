#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import contextlib
import sys

from util import get_logger
import util.android as android


log = get_logger("run_android_emulator")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manages the running of an Android emulator. "
        "Supported modes are to start and stop (default), only start, or only "
        "stop the emulator.")
    parser.add_argument(
        "--android-sdk-root", required=True, help="Path to the Android SDK root.")
    parser.add_argument(
        "--system-image", default="system-images;android-29;google_apis;x86_64",
        help="The Android system image package name.")
    parser.add_argument(
        "--avd-name", default="ort_android",
        help="The Android virtual device name.")

    parser.add_argument(
        "--create-avd", action="store_true",
        help="Whether to create the Android virtual device.")

    parser.add_argument(
        "--start", action="store_true", help="Start the emulator.")
    parser.add_argument(
        "--stop", action="store_true", help="Stop the emulator.")

    parser.add_argument(
        "--pid-file",
        help="Output/input file containing the PID of the emulator process. "
        "This is only required if exactly one of --start or --stop is given.")

    args = parser.parse_args()

    if not args.start and not args.stop:
        # unspecified means start and stop
        args.start = args.stop = True

    if args.start != args.stop and args.pid_file is None:
        raise ValueError("PID file must be specified if only starting or stopping.")

    return args


def main():
    args = parse_args()

    sdk_tool_paths = android.get_sdk_tool_paths(args.android_sdk_root)

    if args.create_avd:
        android.create_virtual_device(sdk_tool_paths, args.system_image, args.avd_name)

    if args.start and args.stop:
        with contextlib.ExitStack() as context_stack:
            emulator_proc = android.start_emulator(sdk_tool_paths, args.avd_name)
            context_stack.enter_context(emulator_proc)
            context_stack.callback(android.stop_emulator, emulator_proc)

            log.info("Press Enter to close.")
            sys.stdin.readline()

    elif args.start:
        emulator_proc = android.start_emulator(sdk_tool_paths, args.avd_name)

        with open(args.pid_file, mode="w") as pid_file:
            print("{}".format(emulator_proc.pid), file=pid_file)

    elif args.stop:
        with open(args.pid_file, mode="r") as pid_file:
            emulator_pid = int(pid_file.readline().strip())

        android.stop_emulator(emulator_pid)


if __name__ == "__main__":
    sys.exit(main())
