#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
#CUDA_VER=cuda10.0-cudnn7.3, cuda9.1-cudnn7.1
CUDA_VER=${4:-cuda9.1-cudnn7.1}

IMAGE="ubuntu16.04-$CUDA_VER"
PYTHON_VER=3.5
OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker

DOCKER_FILE=Dockerfile.ubuntu_gpu_cuda9
if [ $CUDA_VER = "cuda10.0-cudnn7.3" ]; then
DOCKER_FILE=Dockerfile.ubuntu_gpu_cuda
fi

docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f $DOCKER_FILE .

docker rm -f "onnxruntime-gpu-container" || true

set +e

docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-gpu-container" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$IsReleaseBuild" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest-gpu.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev &

wait -n

EXIT_CODE=$?

set -e
cd $OldDir
exit $EXIT_CODE
