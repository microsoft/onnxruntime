#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

python_exe=python3

for i in "$@"; do
  case $i in
    --python=*)
      python_exe="${i#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: ${i}"
      exit 1
  esac
done

cd "$(dirname ${BASH_SOURCE[0]})"
build_dir="${PWD}"

onnx_models_root="$(get_onnx_models_dir)"

# CTestTestfile.cmake files aren't relocatable. Rewrite it to find the build in this directory.
orig_build_dir=$(sed -n "s@# Build directory: @@p" CTestTestfile.cmake)
sed --in-place=".bak" "s@${orig_build_dir}@${build_dir}@g" CTestTestfile.cmake

log_info "-=-=-=- Running ctests -=-=-=-"
# TODO: [AISW-164203] ORT test failures on Rubik Pi
./ctest --verbose --timeout 10800 --stop-on-failure --exclude-regex "onnxruntime_provider_test"

log_info "-=-=-=- Running Python tests -=-=-=-"
mapfile -t PYTHON_TEST_FILES < "python_test_files.txt"

for python_file in "${PYTHON_TEST_FILES[@]}"; do
    if [ -f "${python_file}" ]; then
        # TODO: [AISW-164203] ORT test failures on Rubik Pi
        if [[ "${python_file}" =~ ^(onnxruntime_test_python(_compile_api|_mlops)?.py)$ ]]; then
            log_warn "Skipping ${python_file} due to known failures."
        else
            log_debug "Running ${python_file}..."
            "${python_exe}" ${python_file}
        fi
    else
        log_warn "Failed to find ${python_file} - may be OK on platforms which do not support Python."
    fi
done

if [ -d "quantization" ]; then
    # Quantization tests ran calling unittest directly in MSFT build.py
    "${python_exe}" -m unittest discover -s quantization
else
    log_warn "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
fi

log_info "-=-=-=- Running ONNX model tests -=-=-=-=-"
"${build_dir}/onnx_test_runner" \
    -j 1 \
    -e qnn \
    -i "backend_type|cpu" \
    "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node"

log_info "-=-=-=- Running onnx/models float32 tests -=-=-=-=-"
cd "${onnx_models_root}"
"${build_dir}/onnx_test_runner" \
    -j 1 \
    -e qnn \
    -i "backend_type|cpu" \
    "testdata/float32"

if [ "$(uname -m)" != "aarch64" ]; then  # TODO: [AISW-164203] ORT test failures on Rubik Pi
  log_info "-=-=-=- Running onnx/models qdq tests -=-=-=-=-"
  "${build_dir}/onnx_test_runner" \
      -j 1 \
      -e qnn \
      -i "backend_type|htp" \
      "testdata/qdq"
fi

log_info "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-=-"
log_debug "Scrubbing old context caches"
find "testdata/qdq-with-context-cache" -name "*_ctx.onnx" -print -delete
"${build_dir}/onnx_test_runner" \
    -j 1 \
    -e qnn \
    -f -i "backend_type|htp" \
    "testdata/qdq-with-context-cache"
