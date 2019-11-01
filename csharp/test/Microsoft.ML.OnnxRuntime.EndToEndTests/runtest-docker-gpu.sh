#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
CurrentOnnxRuntimeVersion=$4
PackageName=${PACKAGENAME:-Microsoft.ML.OnnxRuntime.Gpu}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}
#CUDA_VER=cuda10.0-cudnn7.3, cuda9.1-cudnn7.1, cuda10.0-cudnn7.3
CUDA_VER=${5:-cuda10.0-cudnn7.3}

PYTHON_VER=3.5
IMAGE="ubuntu16.04-$CUDA_VER"
OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker

DOCKER_FILE=Dockerfile.ubuntu_gpu_cuda9
if [ $CUDA_VER = "cuda10.0-cudnn7.3" ]; then
DOCKER_FILE=Dockerfile.ubuntu_gpu
fi

docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .

docker rm -f "onnxruntime-gpu-container" || true

set +e

docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-gpu-container" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$ISRELEASEBUILD" \
        -e "PackageName=$PackageName" \
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $CurrentOnnxRuntimeVersion &

wait -n

EXIT_CODE=$?

set -e
cd $OldDir
exit $EXIT_CODE
