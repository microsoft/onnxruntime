#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# and the exclude ops config file, which will be used in the build_minimal_ort_and_run_tests.sh

# This script takes one command line argument, the root of the current Onnx Runtime project

set -e
set -x

ORT_ROOT=$1
MIN_BUILD_DIR=$ORT_ROOT/build_nnapi_minimal

# Remove builds from previous CPU and NNAPI full Android ORT build to free up disk space
rm -rf $ORT_ROOT/build
rm -rf $ORT_ROOT/build_nnapi

# Build with reduced ops requires onnx
python3 -m pip install -U --user onnx

# Copy all the models containing the required ops to pass the UT
mkdir -p $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/ort_github_issue_4031.onnx $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/mnist.onnx $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/ort_minimal_test_models/*.onnx $TMPDIR/.test_data/models_to_include

# Build minimal package for Android x86_64 Emulator
# No test will be triggered in the build process
# UT will be triggered separately below
python3 $ORT_ROOT/tools/ci_build/build.py \
    --build_dir $MIN_BUILD_DIR \
    --config Debug \
    --skip_submodule_sync \
    --parallel \
    --cmake_generator=Ninja \
    --use_nnapi \
    --android \
    --android_sdk_path $ANDROID_HOME \
    --android_ndk_path $ANDROID_HOME/ndk-bundle \
    --android_abi=x86_64 \
    --android_api=29 \
    --build_java \
    --minimal_build extended \
    --disable_rtti \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_model $TMPDIR/.test_data/models_to_include/ \
    --include_ops_by_config $ORT_ROOT/onnxruntime/test/testdata/reduced_ops_via_config.config \
    --skip_tests

# Start the Android Emulator
/bin/bash $ORT_ROOT/tools/ci_build/github/android/start_android_emulator.sh

# Push onnxruntime_test_all and testdata to emulator
adb push $MIN_BUILD_DIR/Debug/onnxruntime_test_all /data/local/tmp/
adb push $MIN_BUILD_DIR/Debug/testdata /data/local/tmp/

# Perform the UT
adb shell 'cd /data/local/tmp/ && ./onnxruntime_test_all'
