#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -e -x

BUILD_DEVICE="CPU"
BUILD_CONFIG="Release"
while getopts "d:" parameter_Option
do case "${parameter_Option}"
in
#GPU or CPU. 
d) BUILD_DEVICE=${OPTARG};;
esac
done

cd $BUILD_BINARIESDIRECTORY
files=(whl/*.whl)
FILE_NAME="${files[0]}"
FILE_NAME=$(basename $FILE_NAME)
PYTHON_PACKAGE_NAME=$(echo "$FILE_NAME" | cut -f 1 -d '-')

echo "Package name:$PYTHON_PACKAGE_NAME"

BUILD_ARGS="--build_dir $BUILD_BINARIESDIRECTORY --config $BUILD_CONFIG --test --skip_submodule_sync --parallel --enable_lto --build_wheel "

if [ $ARCH == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS="$BUILD_ARGS --enable_onnx_tests"
fi
if [ $BUILD_DEVICE == "GPU" ]; then
    BUILD_ARGS="$BUILD_ARGS --use_cuda --use_tensorrt --cuda_version=11.8 --tensorrt_home=/usr --cuda_home=/usr/local/cuda-11.8 --cudnn_home=/usr/local/cuda-11.8"
fi
# We assume the machine doesn't have gcc and python development header files, so we don't build onnxruntime from source
sudo rm -rf /build /onnxruntime_src
sudo ln -s $BUILD_SOURCESDIRECTORY /onnxruntime_src
python3 -m pip uninstall -y $PYTHON_PACKAGE_NAME ort-nightly-gpu ort-nightly onnxruntime onnxruntime-gpu onnxruntime-training onnxruntime-directml ort-nightly-directml onnx -qq
cp $BUILD_SOURCESDIRECTORY/tools/ci_build/github/linux/docker/scripts/manylinux/requirements.txt $BUILD_BINARIESDIRECTORY/requirements.txt
# Test ORT with the latest ONNX release.
sed -i "s/git+http:\/\/github\.com\/onnx\/onnx.*/onnx/" $BUILD_BINARIESDIRECTORY/requirements.txt
python3 -m pip install -r $BUILD_BINARIESDIRECTORY/requirements.txt
python3 -m pip install --find-links $BUILD_BINARIESDIRECTORY/whl $PYTHON_PACKAGE_NAME
ln -s /data/models $BUILD_BINARIESDIRECTORY
cd $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG
# Restore file permissions
xargs -a $BUILD_BINARIESDIRECTORY/$BUILD_CONFIG/perms.txt chmod a+x
python3 $BUILD_SOURCESDIRECTORY/tools/ci_build/build.py $BUILD_ARGS --ctest_path ''
