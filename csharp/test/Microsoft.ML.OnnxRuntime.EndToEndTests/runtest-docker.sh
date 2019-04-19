#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

#TODO: Get this working, not tested yet
set -x



SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
IMAGE="ubuntu16.04"
PYTHON_VER=3.5
OldDir=$(pwd)
cd $SOURCE_ROOT/tools/ci_build/github/linux/docker
docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu .


docker rm -f "onnxruntime-cpu" || true

set +e

docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-cpu" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume "$HOME/.cache/onnxruntime:/home/onnxruntimedev/.cache/onnxruntime" \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$IsReleaseBuild" \
        -e "PackageName=$PackageName" \
        -e "DisableContribOps=$DisableContribOps" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $TestDataUrl $TestDataChecksum &

wait -n

EXIT_CODE=$?

set -e
exit $EXIT_CODE


cd $OldDir
