#!/bin/bash

# This script will run ORT build for Android with code coverage option

set -e
set -x

if [ $# -ne 1 ]; then
    echo "One command line argument, the ROOT root directory, is expected"
fi

ORT_ROOT=$1
# Build and run onnxruntime using NNAPI execution provider targeting android emulator
python3 ${ORT_ROOT}/tools/ci_build/build.py \
	--android \
	--build_dir build_nnapi \
	--android_sdk_path $ANDROID_HOME \
	--android_ndk_path $ANDROID_NDK_HOME \
	--android_abi=x86_64 \
	--android_api=29 \
	--skip_submodule_sync \
	--parallel \
	--use_nnapi \
	--cmake_generator=Ninja \
	--build_java \
	--code_coverage

# Install gcovr
python3 -m pip install gcovr

# Retrieve runtime code coverage files from the emulator and analyze
python3 ${ORT_ROOT}/tools/ci_build/coverage.py \
  --build_dir build_nnapi \
  --android_sdk_path $ANDROID_HOME

