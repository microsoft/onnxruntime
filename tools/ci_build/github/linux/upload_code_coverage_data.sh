#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
$BUILD_SOURCESDIRECTORY/tools/ci_build/github/windows/post_code_coverage_to_dashboard.py --commit_hash=$BUILD_SOURCEVERSION --report_file="$BUILD_BINARIESDIRECTORY/report.json" --report_url=$1 --branch $BUILD_SOURCEBRANCHNAME --arch $2 --os $3 --build_config default