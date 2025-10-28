# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

from typing import Literal

BuildConfigT = Literal["Debug", "Release", "RelWithDebInfo"]
TargetArchLinuxT = Literal["aarch64", "aarch64_manylinux_2_34", "aarch64_oe_gcc11.2", "x86_64"]
TargetArchWindowsT = Literal["arm64", "arm64ec", "x86_64"]
TargetPyVersionT = Literal["3.10", "3.11", "3.12", "3.13"]
