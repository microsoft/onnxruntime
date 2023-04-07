# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys


def is_windows():
    return sys.platform.startswith("win")


def is_macOS():  # noqa: N802
    return sys.platform.startswith("darwin")


def is_linux():
    return sys.platform.startswith("linux")
