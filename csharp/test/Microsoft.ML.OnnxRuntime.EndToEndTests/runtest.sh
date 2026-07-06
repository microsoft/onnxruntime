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

cd $BUILD_SOURCESDIRECTORY

echo "Current NuGet package version is $CurrentOnnxRuntimeVersion"

if [ $RunTestCsharp = "true" ]; then
  if [[ $IsMacOS == "True" || $IsMacOS == "true" ]]; then
    # TODO(#12040): The test should figure out the opset version from the model file. Remove it from the path.
    ONNX_DIR="${BUILD_SOURCESDIRECTORY}/cmake/external/onnx"
    ONNX_VERSION_NUMBER=$(cat "${ONNX_DIR}/VERSION_NUMBER" | sed -E 's/([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
    OPSET_VERSION=$(grep "${ONNX_VERSION_NUMBER}" "${ONNX_DIR}/docs/Versioning.md" | sed -E "s/${ONNX_VERSION_NUMBER}\|[^|]+\|([0-9]+)\|.*/\1/")
    mkdir -p "${BUILD_BINARIESDIRECTORY}/models"
    ln -s "${ONNX_DIR}/onnx/backend/test/data/node" "${BUILD_BINARIESDIRECTORY}/models/opset${OPSET_VERSION}"

    # Fail-loud tripwire for onnx/onnx#7959. The symlink above points the C# backend test at the
    # onnx submodule's on-disk node corpus (onnx/backend/test/data/node). #7959 DELETES that corpus
    # on onnx master + all future releases. This leg deliberately pins the submodule to the
    # IMMUTABLE v1.13.1 tag (see test_macos.yml: "float8 not supported in nuget"), which predates
    # #7959 and still ships the corpus (1097 node case dirs), so the symlink is SAFE today. But if
    # that pin is ever bumped past #7959 the symlink DANGLES, and the test would enumerate ZERO
    # opset cases (InferenceTest.netcore.cs globs opset*/) and pass SILENT-GREEN. Assert a
    # non-empty, non-truncated corpus so that regression is LOUD RED instead.
    #
    # The floor is LEG-LOCAL by design and is intentionally NOT the C++ ORT_ONNX_NODE_MIN_CASES
    # (1500, calibrated for the onnx 1.22 corpus): v1.13.1 (opset 18) ships only 1097 node cases,
    # so 1500 would false-fail this older/smaller corpus on every run. 1000 sits comfortably below
    # the immutable 1097 (never shrinks) while still catching a dangling/empty/truncated tree.
    # See onnx-opset-bump-checklist gotcha (p).
    NODE_MODELS_DIR="${BUILD_BINARIESDIRECTORY}/models/opset${OPSET_VERSION}"
    ONNX_NODE_MIN_CASES=1000
    NODE_CASE_COUNT=$(find -L "${NODE_MODELS_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'test_*' 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ "${NODE_CASE_COUNT}" -lt "${ONNX_NODE_MIN_CASES}" ]; then
      echo "ERROR: node test corpus collapsed -- '${NODE_MODELS_DIR}' resolved to only ${NODE_CASE_COUNT} test_* case dir(s) (floor ${ONNX_NODE_MIN_CASES})."
      echo "       The onnx submodule symlink target is empty, dangling, or truncated. Most likely cmake/external/onnx"
      echo "       was bumped past onnx/onnx#7959, which deletes onnx/backend/test/data/node."
      echo "       See the onnx-opset-bump-checklist gotcha (p). Refusing to run a silent-green 0-case C# backend suite."
      exit 1
    fi
  fi
  # Run C# tests
  dotnet restore $BUILD_SOURCESDIRECTORY/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s $LocalNuGetRepo -s https://api.nuget.org/v3/index.json
  if [ $? -ne 0 ]; then
    echo "Failed to restore nuget packages for the test project"
    exit 1
  fi

  if [ $PACKAGENAME = "Microsoft.ML.OnnxRuntime.Gpu" ] || [ $PACKAGENAME = "Microsoft.ML.OnnxRuntime.Gpu.Linux" ]; then
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

