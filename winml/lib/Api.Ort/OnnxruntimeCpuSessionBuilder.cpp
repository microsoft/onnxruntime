// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include "OnnxruntimeCpuSessionBuilder.h"
#include "OnnxruntimeEngine.h"

using namespace Windows::AI::MachineLearning;

HRESULT OnnxruntimeCpuSessionBuilder::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory) {
  engine_factory_ = engine_factory;
  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::CreateSessionOptions(
    OrtSessionOptions** options) {
  RETURN_HR_IF_NULL(E_POINTER, options);

  auto ort_api = engine_factory_->UseOrtApi();

  OrtSessionOptions* ort_options;
  ort_api->CreateSessionOptions(&ort_options);

  auto session_options = UniqueOrtSessionOptions(ort_options, ort_api->ReleaseSessionOptions);

  // set the graph optimization level to all (used to be called level 3)
  ort_api->SetSessionGraphOptimizationLevel(session_options.get(), GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Onnxruntime will use half the number of concurrent threads supported on the system
  // by default. This causes MLAS to not exercise every logical core.
  // We force the thread pool size to be maxxed out to ensure that WinML always
  // runs the fastest.
  ort_api->SetIntraOpNumThreads(session_options.get(), std::thread::hardware_concurrency());

  // call release() so the underlying OrtSessionOptions object isn't freed
  *options = session_options.release();

    //winml_adapter_api->sessionoptionsappendexecutionprovider_cpu
  //#ifndef _WIN64
  //  xpInfo.create_arena = false;
  //#endif

  return S_OK;
}


HRESULT
OnnxruntimeCpuSessionBuilder::CreateSession(
    OrtSessionOptions* options,
    OrtSession** session,
    OrtExecutionProvider** provider) {
  RETURN_HR_IF_NULL(E_POINTER, session);
  RETURN_HR_IF_NULL(E_POINTER, provider);
  RETURN_HR_IF(E_POINTER, *provider != nullptr);

  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  RETURN_IF_FAILED(engine_factory_->GetOrtEnvironment(&ort_env));

  OrtSession* ort_session_raw;
  winml_adapter_api->CreateSessionWihtoutModel(ort_env, options, &ort_session_raw);
  auto ort_session = UniqueOrtSession(ort_session_raw, ort_api->ReleaseSession);

  //winml_adapter_api->SessionGetExecutionProvidersCount()
  //winml_adapter_api->SessionGetExecutionProvider(i)

  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::Initialize(
    OrtSession* session,
    OrtExecutionProvider* /*p_provider*/
    ) {
  //winml_adapter_api->SessionInitialize(i)
//    ORT_THROW_IF_ERROR(session->get()->Initialize());
  return S_OK;
}
