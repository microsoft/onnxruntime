// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "RawApiTests.h"
#include "RawApiHelpers.h"

#include <fstream>

#include <roapi.h>

namespace ml = Microsoft::AI::MachineLearning;

auto CreateModelAsBuffer(const wchar_t* model_path)
{
    std::ifstream input_stream(model_path, std::ios::binary | std::ios::ate);
    std::streamsize size = input_stream.tellg();
    input_stream.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<std::vector<char>::size_type>(size));
    input_stream.read(buffer.data(), size);

    return std::make_pair(buffer, size);
}

static void RawApiTestsApiTestsClassSetup() {
  WINML_EXPECT_HRESULT_SUCCEEDED(RoInitialize(RO_INIT_TYPE::RO_INIT_MULTITHREADED));
}

static void CreateModelFromFilePath() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));
  WINML_EXPECT_NO_THROW(model.reset());
}

static void CreateCpuDevice() {
  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>());
}

static void Evaluate() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>());

  RunOnDevice(*model.get(), *device.get(), InputStrategy::CopyInputs);

  WINML_EXPECT_NO_THROW(model.reset());
}

static void EvaluateNoInputCopy() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>());

  RunOnDevice(*model.get(), *device.get(), InputStrategy::BindAsReference);

  WINML_EXPECT_NO_THROW(model.reset());
}

static void EvaluateManyBuffers() {
  std::wstring model_path = L"model.onnx";
  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(model_path.c_str(), model_path.size()));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>());

  RunOnDevice(*model.get(), *device.get(), InputStrategy::BindWithMultipleReferences);

  WINML_EXPECT_NO_THROW(model.reset());
}

static void EvaluateFromModelFromBuffer() {
  std::wstring model_path = L"model.onnx";

  auto [buffer, size] = CreateModelAsBuffer(model_path.c_str());

  std::unique_ptr<ml::learning_model> model = nullptr;
  WINML_EXPECT_NO_THROW(model = std::make_unique<ml::learning_model>(buffer.data(), static_cast<size_t>(size)));

  std::unique_ptr<ml::learning_model_device> device = nullptr;
  WINML_EXPECT_NO_THROW(device = std::make_unique<ml::learning_model_device>());

  RunOnDevice(*model.get(), *device.get(), InputStrategy::CopyInputs);

  WINML_EXPECT_NO_THROW(model.reset());
}

const RawApiTestsApi& getapi() {
  static constexpr RawApiTestsApi api = {
      RawApiTestsApiTestsClassSetup,
      CreateModelFromFilePath,
      CreateCpuDevice,
      Evaluate,
      EvaluateNoInputCopy,
      EvaluateManyBuffers,
      EvaluateFromModelFromBuffer,
  };
  return api;
}
