#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Requires nuget and personal access tokens to be setup
nuget restore -PackagesDirectory nuget_root

#requires python3.6 or higher
python3 $DIR/tools/ci_build/build.py --use_brainslice --brain_slice_package_path $DIR/nuget_root --enable_msinternal --brain_slice_package_name CatapultFpga.Linux.5.1.3.40 --brain_slice_client_package_name BrainSlice.v3.Client.3.0.0 --build_dir $DIR/build/Linux "$@"
