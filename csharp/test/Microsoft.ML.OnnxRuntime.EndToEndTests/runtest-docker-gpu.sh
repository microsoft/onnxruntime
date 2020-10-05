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

OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker

docker run --gpus all --rm \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume /data/models:/home/onnxruntimedev/models:ro \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$ISRELEASEBUILD" \
        -e "PackageName=$PackageName" \
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        onnxruntimeregistry.azurecr.io/internal/azureml/onnxruntimecentosgpubuild:ch5h \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $CurrentOnnxRuntimeVersion
