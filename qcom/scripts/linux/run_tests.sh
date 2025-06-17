#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

cd "$(dirname ${BASH_SOURCE[0]})"

# CTestTestfile.cmake files aren't relocatable. Rewrite it to find the build in this directory.

orig_build_dir=$(sed -n "s@# Build directory: @@p" CTestTestfile.cmake)
new_build_dir="${PWD}"

sed --in-place=".bak" "s@${orig_build_dir}@${new_build_dir}@g" CTestTestfile.cmake

./ctest --verbose --timeout 10800
