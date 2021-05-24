#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

LocalNuGetRepo=$1
BuildDir=$3
export CurrentOnnxRuntimeVersion=$4
IsMacOS=${5:-false}
PACKAGENAME=${PACKAGENAME:-Microsoft.ML.OnnxRuntime}
RunTestCsharp=${RunTestCsharp:-true}
RunTestNative=${RunTestNative:-true}

set -x -e

OldDir=`pwd`
cd $BUILD_SOURCESDIRECTORY

echo "Current NuGet package version is $CurrentOnnxRuntimeVersion"

if [ $RunTestCsharp = "true" ]; then
  if [[ $IsMacOS == "True" || $IsMacOS == "true" ]]; then
    mkdir -p $BUILD_BINARIESDIRECTORY/models
    ln -s $BUILD_SOURCESDIRECTORY/cmake/external/onnx/onnx/backend/test/data/node $BUILD_BINARIESDIRECTORY/models/opset14
  fi
  # Run C# tests
  dotnet restore $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s $LocalNuGetRepo -s https://api.nuget.org/v3/index.json
  if [ $? -ne 0 ]; then
    echo "Failed to restore nuget packages for the test project"
    exit 1
  fi

  dotnet test $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore --verbosity detailed
  if [ $? -ne 0 ]; then
    echo "Failed to build or execute the end-to-end test"
    exit 1
  fi
fi

if [ $RunTestNative = "true" ]; then
  # Run Native shared object test
  # PACKAGENAME is passed in environment (e.g. Microsoft.ML.OnnxRuntime)
  PACKAGENAME="$PACKAGENAME.$CurrentOnnxRuntimeVersion.nupkg"
  cd $LocalNuGetRepo
  TempDir=_tmp
  mkdir -p $TempDir && pushd $TempDir
  unzip ../$PACKAGENAME

  inc="-I build/native/include"

  if [[ $IsMacOS == "True" || $IsMacOS == "true" ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH=$LocalNuGetRepo/_tmp:${DYLD_FALLBACK_LIBRARY_PATH}
    libs="-L runtimes/osx-x64/native -l onnxruntime"
    g++ -std=c++11 $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp $inc $libs -Wunused-result -Wformat=0 -o sampletest
    libName=$(otool -L ./sampletest | grep onnxruntime | xargs | cut -d' ' -f1 | cut -d'/' -f2)
    ln -sf runtimes/osx-x64/native/libonnxruntime.dylib $libName
  else
    export LD_LIBRARY_PATH=$LocalNuGetRepo/_tmp:${LD_LIBRARY_PATH}
    libs="-L runtimes/linux-x86/native -L runtimes/linux-x64/native -l onnxruntime"
    g++ -std=c++11 $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp $inc $libs -Wunused-result -o sampletest
    # Create link to versioned shared object required at runtime
    libname=`ldd sampletest | grep onnxruntime | xargs | cut -d" " -f1`
    ln -sf runtimes/linux-x64/native/libonnxruntime.so $libname
  fi

  # Copy Sample Model
  cp $BUILD_SOURCESDIRECTORY/csharp/testdata/squeezenet.onnx .

  # Run the sample model
  ./sampletest
  popd
  rm -rf $TempDir
fi
cd $OldDir

