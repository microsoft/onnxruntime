#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -e -x

BUILD_DEVICE="CPU"
BUILD_CONFIG="Release"
while getopts "d:c:u:" parameter_Option
do case "${parameter_Option}"
in
#GPU or CPU. 
d) BUILD_DEVICE=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
u) CUDA_VERSION=${OPTARG:"11.8"};;
*) echo "Usage: $0 -d <GPU|CPU> [-c <build_config>] [-u <cuda_version>] "
   exit 1;;
esac
done

export PATH=/opt/python/cp38-cp38/bin:$PATH
cd /build
files=(whl/*.whl)
FILE_NAME="${files[0]}"
FILE_NAME=$(basename $FILE_NAME)
PYTHON_PACKAGE_NAME=$(echo "$FILE_NAME" | cut -f 1 -d '-')

echo "Package name:$PYTHON_PACKAGE_NAME"

BUILD_ARGS="--build_dir /build --config $BUILD_CONFIG --test --skip_submodule_sync --parallel --enable_lto --build_wheel "

ARCH=$(uname -m)

if [ $ARCH == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS="$BUILD_ARGS --enable_onnx_tests"
fi
if [ $BUILD_DEVICE == "GPU" ]; then
    BUILD_ARGS="$BUILD_ARGS --use_cuda --use_tensorrt --tensorrt_home=/usr --cuda_version=$CUDA_VERSION  --cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr/local/cuda-$CUDA_VERSION"
fi
# We assume the machine doesn't have gcc and python development header files, so we don't build onnxruntime from source
python3 -m pip install --upgrade pip
# Install the packages that are needed for installing the onnxruntime python package
python3 -m pip install -r /build/$BUILD_CONFIG/requirements.txt
# Install the packages that are needed for running test scripts
python3 -m pip install pytest
# The "--no-index" flag is crucial. The local whl folder is just an additional source. Pypi's doc says "there is no 
# ordering in the locations that are searched" if we don't disable the default one with "--no-index"
python3 -m pip install --no-index --find-links /build/whl $PYTHON_PACKAGE_NAME
cd /build/$BUILD_CONFIG
# Restore file permissions
xargs -a perms.txt chmod a+x
python3 /onnxruntime_src/tools/ci_build/build.py $BUILD_ARGS --ctest_path ''
