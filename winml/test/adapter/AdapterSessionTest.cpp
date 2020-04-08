// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include "lib/Api.Ort/pch.h"

#include "AdapterSessionTest.h"
#include "OnnxruntimeEngine.h"

// #include <winrt/Windows.Graphics.Imaging.h>
// #include <winrt/Windows.Media.h>
// #include <winrt/Windows.Storage.h>
// #include <winrt/Windows.Storage.Streams.h>

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

namespace {
WinML::OnnxruntimeEngineFactory engine_factory;
const OrtApi *ort_api;
const WinmlAdapterApi *winml_adapter_api;
OrtEnv* ort_env;

static void AdapterSessionTestSetup() {
  init_apartment();
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory.RuntimeClassInitialize());
  WINML_EXPECT_NOT_EQUAL(nullptr, winml_adapter_api = engine_factory.UseWinmlAdapterApi());
  WINML_EXPECT_NOT_EQUAL(nullptr, ort_api = engine_factory.UseOrtApi());
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory.GetOrtEnvironment(&ort_env));
}

static void AppendExecutionProvider_CPU() {
  // LearningModel learningModel = nullptr;
  // WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
}

static void AppendExecutionProvider_DML() {

}

static void CreateWithoutModel() {
  OrtSession* ort_session_raw;
  WINML_EXPECT_NOT_EQUAL(nullptr, winml_adapter_api->CreateSessionWithoutModel(ort_env, {}, &ort_session_raw));
}

static void GetExecutionProvider() {

}

static void Initialize() {

}

static void RegisterGraphTransformers() {

}

static void RegisterCustomRegistry() {

}

static void LoadAndPurloinModel() {

}

static void StartProfiling() {

}

static void EndProfiling() {

}

static void CopyInputAcrossDevices() {

}
}

const AdapterSessionTestAPi& getapi() {
  static constexpr AdapterSessionTestAPi api =
  {
    AdapterSessionTestSetup,
    AppendExecutionProvider_CPU,
    AppendExecutionProvider_DML,
    CreateWithoutModel,
    GetExecutionProvider,
    Initialize,
    RegisterGraphTransformers,
    RegisterCustomRegistry,
    LoadAndPurloinModel,
    StartProfiling,
    EndProfiling,
    CopyInputAcrossDevices
  };
  return api;
}
