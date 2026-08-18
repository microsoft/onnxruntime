#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -e -x

BUILD_DEVICE="CPU"
BUILD_CONFIG="Release"
while getopts "d:c:" parameter_Option
do case "${parameter_Option}"
in
#GPU or CPU.
d) BUILD_DEVICE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
esac
done

export PATH=/opt/python/cp312-cp312/bin:$PATH
cd /build
files=(whl/*.whl)
FILE_NAME="${files[0]}"
FILE_NAME=$(basename $FILE_NAME)
PYTHON_PACKAGE_NAME=$(echo "$FILE_NAME" | cut -f 1 -d '-')

echo "Package name:$PYTHON_PACKAGE_NAME"

BUILD_ARGS="--build_dir /build --config $BUILD_CONFIG --test --skip_submodule_sync --parallel --enable_lto --build_wheel "

if [[ "$PYTHON_PACKAGE_NAME" == *"training"* ]]; then
  BUILD_ARGS="$BUILD_ARGS --enable_training"
fi

ARCH=$(uname -m)

if [ $ARCH == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS="$BUILD_ARGS --enable_onnx_tests"
fi
if [ $BUILD_DEVICE == "GPU" ]; then
    SHORT_CUDA_VERSION=$(echo $CUDA_VERSION | sed   's/\([[:digit:]]\+\.[[:digit:]]\+\)\.[[:digit:]]\+/\1/')
    CUDA_HOME=/usr/local/cuda-$SHORT_CUDA_VERSION
    if [ ! -d "$CUDA_HOME" ] && [ -d /usr/local/cuda ]; then
        # Allow the cu13 packaging flow to run on images that expose a newer CUDA minor version via /usr/local/cuda.
        CUDA_HOME=/usr/local/cuda
    fi

    BUILD_ARGS="$BUILD_ARGS --use_cuda --cuda_version=$SHORT_CUDA_VERSION --cuda_home=$CUDA_HOME --cudnn_home=$CUDA_HOME"
    # Enable TRT EP only if TensorRT is installed.
    if [ -f /usr/include/NvInfer.h ]; then
        BUILD_ARGS="$BUILD_ARGS --use_tensorrt --tensorrt_home=/usr"
    elif [ "$ARCH" != "aarch64" ] && [ -f /opt/tensorrt/include/NvInfer.h ]; then
        # The aarch64 TensorRT tarball is not compatible with the packaging image's glibc baseline.
        BUILD_ARGS="$BUILD_ARGS --use_tensorrt --tensorrt_home=/opt/tensorrt"
    fi
fi

python3 -m pip install --upgrade pip
# Install the packages that are needed for installing the onnxruntime python package
python3 -m pip install -r /build/$BUILD_CONFIG/requirements.txt
# Install the packages that are needed for running test scripts
python3 -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/python/requirements.txt
# The "--no-index" flag is crucial. The local whl folder is just an additional source. Pypi's doc says "there is no
# ordering in the locations that are searched" if we don't disable the default one with "--no-index"
python3 -m pip install --no-index --find-links /build/whl $PYTHON_PACKAGE_NAME
cd /build/$BUILD_CONFIG
# Restore file permissions
xargs -a perms.txt chmod a+x
python3 /onnxruntime_src/tools/ci_build/build.py $BUILD_ARGS --ctest_path ''
