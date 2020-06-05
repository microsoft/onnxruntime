#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x -e

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
CurrentOnnxRuntimeVersion=$4
PackageName=${PACKAGENAME:-Microsoft.ML.OnnxRuntime.Gpu}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}

PYTHON_VER=3.5
IMAGE="ubuntu16.04-$CUDA_VER"
OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker

sudo docker build -t onnxruntime-ubuntu16.04-cuda10.1-cudnn7.6 --build-arg OS_VERSION=16.04 --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER}  -f Dockerfile.ubuntu_gpu .
sudo docker rm -f "onnxruntime-gpu-container" || true


sudo --preserve-env docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-gpu-container" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume /data/models:/home/onnxruntimedev/models:ro \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$ISRELEASEBUILD" \
        -e "PackageName=$PackageName" \
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        onnxruntime-ubuntu16.04-cuda10.1-cudnn7.6 \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $CurrentOnnxRuntimeVersion
