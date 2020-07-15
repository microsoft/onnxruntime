#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# build docker image for CPU

set -x -e

SOURCE_ROOT=$1
BUILD_DIR=$2
NUGET_REPO_DIRNAME=$3   # path relative to BUILD_DIR
CurrentOnnxRuntimeVersion=$4
UseCentos7=${5:-false}
Arch=${6:-x64}          # x32, x64
PackageName=${PACKAGENAME:-Microsoft.ML.OnnxRuntime}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}
PYTHON_VER=3.5
IMAGE="ubuntu16.04_$Arch"

OldDir=$(pwd)

cd $SOURCE_ROOT/tools/ci_build/github/linux/docker

if [ $UseCentos7 = "false" ]; then
  echo "Image used for testing is onnxruntime-$IMAGE"
  if [ $Arch = "x86" ]; then
    sudo docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu_x86 .
  else
    sudo docker build -t "onnxruntime-$IMAGE" --build-arg OS_VERSION=16.04 --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.ubuntu .
  fi
else
  IMAGE="centos7"
  PYTHON_VER=3.6
  echo "Image used for testing is onnxruntime-$IMAGE"

  sudo docker build -t "onnxruntime-$IMAGE" --build-arg BUILD_USER=onnxruntimedev --build-arg BUILD_UID=$(id -u) --build-arg PYTHON_VERSION=${PYTHON_VER} -f Dockerfile.centos .
fi

sudo docker rm -f "onnxruntime-cpu" || true


sudo --preserve-env docker run -h $HOSTNAME \
        --rm \
        --name "onnxruntime-cpu" \
        --volume "$SOURCE_ROOT:/onnxruntime_src" \
        --volume "$BUILD_DIR:/home/onnxruntimedev" \
        --volume /data/models:/home/onnxruntimedev/models:ro \
        -e "OnnxRuntimeBuildDirectory=/home/onnxruntimedev" \
        -e "IsReleaseBuild=$ISRELEASEBUILD" \
        -e "PackageName=$PackageName" \
        -e "DisableContribOps=$DISABLECONTRIBOPS" \
        -e "DisableMlOps=$DISABLEMLOPS" \
        -e "RunTestCsharp=$RunTestCsharp" \
        -e "RunTestNative=$RunTestNative" \
        "onnxruntime-$IMAGE" \
        /bin/bash /onnxruntime_src/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/runtest.sh \
        /home/onnxruntimedev/$NUGET_REPO_DIRNAME /onnxruntime_src /home/onnxruntimedev $CurrentOnnxRuntimeVersion
