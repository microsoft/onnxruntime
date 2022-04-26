#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import argparse

from _test_commons import run_subprocess

import logging

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmd_line_with_args",
        required=True,
        help="command line with arguments to be executed in a subprocess. \
        it expects a single string containing arguments separated by spaces.",
    )
    parser.add_argument("--cwd", help="working directory")
    # parser.add_argument("--env", help="env variables.")
    parser.add_argument("--env", help="env variables", nargs=2, action="append", default=[])

    return parser.parse_args()


launch_args = parse_arguments()

print("sys.executable: ", sys.executable)
cmd_line_with_args = launch_args.cmd_line_with_args.split()
for n, arg in enumerate(cmd_line_with_args):
    if arg == "python":
        cmd_line_with_args[n] = sys.executable

run_subprocess(cmd_line_with_args, cwd=launch_args.cwd, env=dict(launch_args.env), log=log)
