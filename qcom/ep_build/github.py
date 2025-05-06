# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os

from .util import Colors, echo, get_env_bool


def on_github():
    return "GITHUB_ACTION" in os.environ or get_env_bool("EP_BUILD_FORCE_ON_CI", False)


def start_group(group_name):
    if on_github():
        print(f"::group::{group_name}", flush=True)
    else:
        echo(f"{Colors.GREEN}{group_name}{Colors.OFF}")


def end_group():
    if on_github():
        print("::endgroup::", flush=True)


def set_github_output(key, value):
    if on_github():
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            print(f"{key}={value}", file=fh)
