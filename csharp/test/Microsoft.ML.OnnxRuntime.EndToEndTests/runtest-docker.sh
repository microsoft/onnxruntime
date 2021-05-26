#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x -e

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
CurrentOnnxRuntimeVersion=$4
DockerImage=$5
UseCentos7=${6:-false}
Arch=${7:-x64}          # x32, x64
PACKAGENAME=${PACKAGENAME:-Microsoft.ML.OnnxRuntime}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}
PYTHON_VER=3.5
IMAGE="ubuntu16.04_$Arch"

OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker


docker run --rm \
        --name "onnxruntime-cpu" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume /data/models:/home/onnxruntimedev/models:ro \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$ISRELEASEBUILD" \
        -e "PACKAGENAME=$PACKAGENAME" \
        -e "DisableContribOps=$DISABLECONTRIBOPS" \
        -e "DisableMlOps=$DISABLEMLOPS" \
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        -e "BUILD_BINARIESDIRECTORY=/home/onnxruntimedev" \
        -e "BUILD_SOURCESDIRECTORY=/onnxruntime_src" \
        "$DockerImage" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $CurrentOnnxRuntimeVersion
