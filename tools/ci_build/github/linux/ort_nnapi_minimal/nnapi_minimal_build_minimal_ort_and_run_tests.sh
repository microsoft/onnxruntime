#!/bin/bash

# This script will run a full ORT build and use the python package built to generate ort format test files,
# and the exclude ops config file, which will be used in the build_minimal_ort_and_run_tests.sh

set -e
set -x

ORT_ROOT=$1

MIN_BUILD_DIR=$ORT_ROOT/min_build

mkdir -p $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/ort_github_issue_4031.onnx $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/mnist.onnx $TMPDIR/.test_data/models_to_include
cp $ORT_ROOT/onnxruntime/test/testdata/ort_minimal_test_models/*.onnx $TMPDIR/.test_data/models_to_include

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
    --minimal_build extended \
    --disable_rtti \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_model $TMPDIR/.test_data/models_to_include/ \
    --include_ops_by_config $TMPDIR/.test_data/ort_minimal_e2e_test_data/required_operators.config

# Start the Android Emulator
/bin/bash $ORT_ROOT/tools/ci_build/github/android/start_android_emulator.sh

# Push onnx_test_runner to emulator
adb push $MIN_BUILD_DIR/Debug/onnx_test_runner /data/local/tmp/

# Push test data to device
adb push $ORT_ROOT/.test_data/ort_minimal_e2e_test_data /data/local/tmp/

# Perform the e2e tests
adb shell 'cd /data/local/tmp/ && ./onnx_test_runner -e nnapi ./ort_minimal_e2e_test_data'

# Push onnxruntime_test_all and testdata to emulator
adb push $MIN_BUILD_DIR/Debug/onnxruntime_test_all /data/local/tmp/
adb push $MIN_BUILD_DIR/Debug/testdata /data/local/tmp/

# Perform the ut
adb shell 'cd /data/local/tmp/ && ./onnxruntime_test_all'
