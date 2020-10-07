#!/bin/bash

# This script will run a baseline minimal ort build for android arm64-v8a ABI
# and report the binary size to the ort mysql DB

set -e

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
    --android_ndk_path /android_home/ndk-bundle \
    --android_abi=arm64-v8a \
    --android_api=29 \
    --minimal_build \
    --build_shared_lib \
    --build_java \
    --disable_ml_ops \
    --disable_exceptions \
    --test_binary_size \
    --include_ops_by_config /home/onnxruntimedev/.test_data/include_no_operators.config

# Install the mysql connector
python3 -m pip install --user mysql-connector-python

# Post the binary size info to ort mysql DB
# The report script's DB connection failure will not fail the pipeline
python3 /onnxruntime_src/tools/ci_build/github/windows/post_binary_sizes_to_dashboard.py \
    --ignore_db_error \
    --commit_hash=$BUILD_SOURCEVERSION \
    --size_data_file=/build/MinSizeRel/binary_size_data.txt \
    --build_project=onnxruntime \
    --build_id=$BUILD_ID

# Clear the build
rm -rf /build/MinSizeRel
