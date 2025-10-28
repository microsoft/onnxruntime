#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

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

# CTestTestfile.cmake files aren't relocatable. Rewrite it to find the build in this directory.

orig_build_dir=$(sed -n "s@# Build directory: @@p" CTestTestfile.cmake)
new_build_dir="${PWD}"

sed --in-place=".bak" "s@${orig_build_dir}@${new_build_dir}@g" CTestTestfile.cmake

echo "Running ctests..."
./ctest --verbose --timeout 10800

echo "Running Python tests..."
mapfile -t PYTHON_TEST_FILES < "python_test_files.txt"

for python_file in "${PYTHON_TEST_FILES[@]}"; do
    if [ -f "${python_file}" ]; then
        echo "Running ${python_file}..."
        "${python_exe}" ${python_file}
    else
        echo "Failed to find ${python_file} - may be OK on platforms which do not support Python."
    fi
done

if [ -d "quantization" ]; then
    # Quantization tests ran calling unittest directly in MSFT build.py
    "${python_exe}" -m unittest discover -s quantization
else
    echo "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
fi
