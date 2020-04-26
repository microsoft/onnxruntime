// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "RawApiTests.h"

#include "microsoft.ai.machinelearning.h"
#include "microsoft.ai.machinelearning.native.h"

#include "winml/microsoft.ai.machinelearning.h"
#include "winml/microsoft.ai.machinelearning.gpu.h"
#include <d3d11.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <dxgi.h>
#include <dxgi1_6.h>
#include <d3d11on12.h>
#include <d3d11_3.h>
#include <VersionHelpers.h>

namespace ml = Microsoft::AI::MachineLearning;

static void GpuMethodSetup() {
  GPUTEST;
}

static void CreateModelFromFilePath() {
  std::wstring model_path = L"squeezenet_modifiedforruntimestests.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));
  WINML_EXPECT_NO_THROW(model.reset());
}

template <>
const RawApiTestsApi& getapi<RawApiTestsApi>() {
  static constexpr RawApiTestsApi api = {
      CreateModelFromFilePath,
  };
  return api;
}
template <>
const RawApiTestsGpuApi& getapi<RawApiTestsGpuApi>() {
  static constexpr RawApiTestsGpuApi api = {
      GpuMethodSetup,
  };
  return api;
}