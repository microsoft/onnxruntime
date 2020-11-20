#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# and the exclude ops config file, which will be used in the build_minimal_ort_and_run_tests.sh

set -e

# Build python package requires numpy
python3 -m pip install --user numpy

ORT_ROOT=$1
FULL_BUILD_DIR=$ORT_ROOT/full_build

# Run a full build of ORT
# Since we need the ORT python package to generate the ORT format files and the include ops config files
# Will not run tests since those are covered by other CIs
python3 $ORT_ROOT/tools/ci_build/build.py \
    --build_dir $FULL_BUILD_DIR \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --cmake_generator=Ninja \
    --build_wheel \
    --enable_pybind \
    --skip_tests \
    --use_nnapi

# Install the ORT python wheel
pip3 install -U $FULL_BUILD_DIR/Debug/dist/*

# Copy the test data to a separated folder
cp -Rf $ORT_ROOT/onnxruntime/test/testdata/ort_minimal_e2e_test_data $TMPDIR/.test_data

# Convert onnx file to ort requires onnx
pip3 install -U onnx

# Convert all the onnx models in the $HOME/.test_data/ort_minimal_e2e_test_data to ort model
# and generate the included ops config file as $HOME/.test_data/ort_minimal_e2e_test_data/required_operators.config
python3 $ORT_ROOT/tools/python/convert_onnx_models_to_ort.py --use_nnapi \
    $ORT_ROOT/.test_data/ort_minimal_e2e_test_data

# Uninstall the ORT python wheel
pip3 uninstall -y onnxruntime_noopenmp

MIN_BUILD_DIR=$ORT_ROOT/min_build

# build minimal package for Android x86_64 Emulator
# Since this is a minimal build with reduced ops, we will only run e2e test using onnx_test_runner
python3 $ORT_ROOT/tools/ci_build/build.py \
    --build_dir $MIN_BUILD_DIR \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --cmake_generator=Ninja \
    --skip_tests \
    --use_nnapi \
    --android \
    --android_sdk_path $ANDROID_HOME \
    --android_ndk_path $ANDROID_HOME/ndk-bundle \
    --android_abi=x86_64 \
    --android_api=29 \
    --build_java \
    --include_ops_by_config $TMPDIR/.test_data/ort_minimal_e2e_test_data/required_operators.config

