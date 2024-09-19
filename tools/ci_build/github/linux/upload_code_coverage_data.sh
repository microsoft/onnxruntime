#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -x -e
/usr/bin/env python3 -m pip install --user -r $BUILD_SOURCESDIRECTORY/tools/ci_build/github/windows/post_to_dashboard/requirements.txt
$BUILD_SOURCESDIRECTORY/tools/ci_build/github/windows/post_code_coverage_to_dashboard.py --commit_hash=$BUILD_SOURCEVERSION --report_file $1 --report_url $2 --branch $BUILD_SOURCEBRANCHNAME --arch $3 --os $4 --build_config $5