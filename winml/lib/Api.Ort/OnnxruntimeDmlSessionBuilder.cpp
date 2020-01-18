// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#ifdef USE_DML

#include "OnnxruntimeDmlSessionBuilder.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"
#include "LearningModelDevice.h"

using namespace Windows::AI::MachineLearning;

HRESULT OnnxruntimeDmlSessionBuilder::RuntimeClassInitialize(OnnxruntimeEngineFactory* engine_factory, ID3D12Device* device, ID3D12CommandQueue* queue) {
  engine_factory_ = engine_factory;
  device_.copy_from(device);
  queue_.copy_from(queue);
  return S_OK;
}

HRESULT
OnnxruntimeDmlSessionBuilder::CreateSessionOptions(
    OrtSessionOptions** options) {
  RETURN_HR_IF_NULL(E_POINTER, options);

  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtSessionOptions* ort_options;
  ort_api->CreateSessionOptions(&ort_options);

  auto session_options = UniqueOrtSessionOptions(ort_options, ort_api->ReleaseSessionOptions);

  // set the graph optimization level to all (used to be called level 3)
  ort_api->SetSessionGraphOptimizationLevel(session_options.get(), GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Disable the mem pattern session option for DML. It will cause problems with how memory is allocated.
  ort_api->DisableMemPattern(session_options.get());

  // Request the dml ep
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device_.get(), queue_.get()),
                                   ort_api);

#ifndef _WIN64
  auto use_arena = false;
#else
  auto use_arena = true;
#endif
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), use_arena),
                                   ort_api);

  // call release() so the underlying OrtSessionOptions object isn't freed
  *options = session_options.release();

  return S_OK;
}

HRESULT OnnxruntimeDmlSessionBuilder::CreateSession(
    OrtSessionOptions* options,
    OrtSession** session) {
  RETURN_HR_IF_NULL(E_POINTER, session);

  auto ort_api = engine_factory_->UseOrtApi();
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();

  OrtEnv* ort_env;
  RETURN_IF_FAILED(engine_factory_->GetOrtEnvironment(&ort_env));

  OrtSession* ort_session_raw;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, options, &ort_session_raw),
                                   engine_factory_->UseOrtApi());
  auto ort_session = UniqueOrtSession(ort_session_raw, ort_api->ReleaseSession);

  *session = ort_session.release();

  return S_OK;
}

HRESULT OnnxruntimeDmlSessionBuilder::Initialize(
    OrtSession* session) {
  RETURN_HR_IF_NULL(E_INVALIDARG, session);
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionInitialize(session),
                                   engine_factory_->UseOrtApi());

  OrtExecutionProvider* ort_provider;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvider(session, 0, &ort_provider),
                                   engine_factory_->UseOrtApi());

 
  size_t num_providers;
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->SessionGetExecutionProvidersCount(session, &num_providers),
                                   engine_factory_->UseOrtApi());
  RETURN_HR_IF(E_UNEXPECTED, num_providers != 2);

  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlExecutionProviderSetDefaultRoundingMode(ort_provider, true),
                                   engine_factory_->UseOrtApi());

  // Flush the D3D12 work from the DML execution provider
  RETURN_HR_IF_WINMLA_API_FAIL_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider),
                                   engine_factory_->UseOrtApi());

  return S_OK;
}

#endif USE_DML