#!/bin/bash

# This script will run a baseline minimal ort build for android arm64-v8a ABI
# and report the binary size to the ort mysql DB

set -e
set -x
export PATH=/opt/python/cp37-cp37m/bin:$PATH
# Create an empty file to be used with build --include_ops_by_config, which will include no operators at all
echo -n > /home/onnxruntimedev/.test_data/include_no_operators.config

# Run a baseline minimal build of ORT Android arm64-v8a
# Generate binary size as /build/MinSizeRel/binary_size_data.txt
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config MinSizeRel \
    --skip_submodule_sync \
    --parallel \
    --android \
    --android_sdk_path /android_home \
    --android_ndk_path /ndk_home \
    --android_abi=arm64-v8a \
    --android_api=29 \
    --minimal_build \
    --build_shared_lib \
    --build_java \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_config /home/onnxruntimedev/.test_data/include_no_operators.config

# set current size limit to BINARY_SIZE_LIMIT_IN_BYTES.
BINARY_SIZE_LIMIT_IN_BYTES=1256000
echo "The current preset binary size limit is $BINARY_SIZE_LIMIT_IN_BYTES"
python3 /onnxruntime_src/tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py \
    --threshold=$BINARY_SIZE_LIMIT_IN_BYTES \
    /build/MinSizeRel/libonnxruntime.so

echo "The content of binary_size_data.txt"
cat /build/MinSizeRel/binary_size_data.txt
