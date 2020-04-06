// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "AdapterSessionTest.h"
// #include "OnnxruntimeSessionBuilder.h"
#include "OnnxruntimeEngine.h"

// #include <winrt/Windows.Graphics.Imaging.h>
// #include <winrt/Windows.Media.h>
// #include <winrt/Windows.Storage.h>
// #include <winrt/Windows.Storage.Streams.h>

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
// using namespace winrt::Windows::Foundation::Collections;
// using namespace winrt::Windows::Graphics::Imaging;
// using namespace winrt::Windows::Media;
// using namespace winrt::Windows::Storage;
// using namespace winrt::Windows::Storage::Streams;

namespace {
OnnxruntimeEngineFactory engine_factory;
// const auto ort_api = engine_factory->UseOrtApi();
// const auto winml_adapter_api = engine_factory->UseWinmlAdapterApi();
// OrtEnv* ort_env;

static void AdapterSessionTestSetup() {
  init_apartment();
  // WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory->GetOrtEnvironment(&ort_env));
}

static void AppendExecutionProvider_CPU() {
  // LearningModel learningModel = nullptr;
  // WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
}

static void AppendExecutionProvider_DML() {

}

static void CreateWithoutModel() {
  // RETURN_HR_IF_NULL(E_POINTER, session);

  // auto ort_api = engine_factory->UseOrtApi();
  // auto winml_adapter_api = engine_factory->UseWinmlAdapterApi();

  // OrtEnv* ort_env;
  // RETURN_IF_FAILED(engine_factory->GetOrtEnvironment(&ort_env));

  // OrtSession* ort_session_raw;
  // RETURN_HR_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, options, &ort_session_raw),
  //                         engine_factory->UseOrtApi());

  // auto ort_session = UniqueOrtSession(ort_session_raw, ort_api->ReleaseSession);

  // *session = ort_session.release();

  // return S_OK;
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

static void CopyOneInputAcrossDevices() {

}
}

const AdapterSessionTestAPi& getapi() {
  static constexpr AdapterSessionTestAPi api =
  {
    AdapterSessionTestSetup,
    AppendExecutionProvider_DML,
    CreateWithoutModel,
    GetExecutionProvider,
    Initialize,
    RegisterGraphTransformers,
    RegisterCustomRegistry,
    LoadAndPurloinModel,
    StartProfiling,
    EndProfiling,
    CopyOneInputAcrossDevices
  };
  return api;
}
