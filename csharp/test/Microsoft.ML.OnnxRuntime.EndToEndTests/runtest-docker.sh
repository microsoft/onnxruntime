#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
Arch=${4:-x64}          # x32, x64
PackageName=${PackageName:-Microsoft.ML.OnnxRuntime}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}
PYTHON_VER=3.5
IMAGE="ubuntu16.04_$Arch"

OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker
if [ $Arch = "x86" ]; then
   docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu_x86 .
else
   docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu .
fi

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
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev &

wait -n

EXIT_CODE=$?

set -e
exit $EXIT_CODE
cd $OldDir
