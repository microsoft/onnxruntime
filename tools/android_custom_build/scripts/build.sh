#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Note: This script is intended to be run by ../build_custom_android_package.py in a Docker container.

set -e -x

USAGE_TEXT="Usage: ${0} <build config> <ops config file> <build settings file> <output directory>"

BUILD_CONFIG=${1:?${USAGE_TEXT}}
OPS_CONFIG_FILE=${2:?${USAGE_TEXT}}
BUILD_SETTINGS_FILE=${3:?${USAGE_TEXT}}
OUTPUT_DIR=${4:?${USAGE_TEXT}}

BUILD_DIR=/workspace/build

# build ORT AAR
python3 /workspace/onnxruntime/tools/ci_build/github/android/build_aar_package.py \
  --build_dir="${BUILD_DIR}" \
  --config="${BUILD_CONFIG}" \
  --include_ops_by_config="${OPS_CONFIG_FILE}" \
  "${BUILD_SETTINGS_FILE}"

# copy AAR to output directory
cp -r "${BUILD_DIR}/aar_out" "${OUTPUT_DIR}"
