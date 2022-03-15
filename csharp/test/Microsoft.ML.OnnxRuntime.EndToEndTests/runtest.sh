#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

LocalNuGetRepo=$1
export CurrentOnnxRuntimeVersion=$2
IsMacOS=${3:-false}
PACKAGENAME=${PACKAGENAME:-Microsoft.ML.OnnxRuntime}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}

set -x -e

pushd .
cd $BUILD_SOURCESDIRECTORY

echo "Current NuGet package version is $CurrentOnnxRuntimeVersion"

if [ $RunTestCsharp = "true" ]; then
  if [[ $IsMacOS == "True" || $IsMacOS == "true" ]]; then
    mkdir -p $BUILD_BINARIESDIRECTORY/models
    ln -s $BUILD_SOURCESDIRECTORY/cmake/external/onnx/onnx/backend/test/data/node $BUILD_BINARIESDIRECTORY/models/opset16
  fi
  # Run C# tests
  dotnet restore $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s $LocalNuGetRepo -s https://api.nuget.org/v3/index.json
  if [ $? -ne 0 ]; then
    echo "Failed to restore nuget packages for the test project"
    exit 1
  fi

  if [ $PACKAGENAME = "Microsoft.ML.OnnxRuntime.Gpu" ]; then
    export TESTONGPU=ON 
    dotnet test -p:DefineConstants=USE_CUDA $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore --verbosity detailed
    if [ $? -ne 0 ]; then
      echo "Failed to build or execute the end-to-end test"
      exit 1
    fi
    dotnet test -p:DefineConstants=USE_TENSORRT $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore --verbosity detailed
  else
    dotnet test $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore --verbosity detailed
  fi
  if [ $? -ne 0 ]; then
    echo "Failed to build or execute the end-to-end test"
    exit 1
  fi
fi

cd $OldDir
popd
