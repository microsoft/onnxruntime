#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#requires python3.6 or higher
python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/Linux "$@"
