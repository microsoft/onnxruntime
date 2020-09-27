#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# and the exclude ops config file, which will be used in the build_minimal_ort_and_run_tests.sh

set -e

# Run a full build of ORT
# Since we need the ORT python package to generate the ORT format files and the include ops config files
# Will not run tests since those are covered by other CIs
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --build_wheel \
    --skip_tests \
    --enable_pybind

# Install the ORT python wheel
python3 -m pip install --user /build/Debug/dist/*

# Copy the test data to a separated folder
cp -Rf /onnxruntime_src/onnxruntime/test/testdata/ort_minimal_e2e_test_data /home/onnxruntimedev/.test_data

# Convert all the onnx models in the $HOME/.test_data/ort_minimal_e2e_test_data to ort model
# and generate the included ops config file as $HOME/.test_data/ort_minimal_e2e_test_data/required_operators.config
python3 /onnxruntime_src/tools/python/convert_onnx_models_to_ort.py \
    /home/onnxruntimedev/.test_data/ort_minimal_e2e_test_data

# Delete all the .onnx files, because the minimal build tests will not work on onnx files
find /home/onnxruntimedev/.test_data/ort_minimal_e2e_test_data -type f -name "*.onnx" -delete

# Uninstall the ORT python wheel
python3 -m pip uninstall -y onnxruntime

# Clear the build
rm -rf /build/Debug
