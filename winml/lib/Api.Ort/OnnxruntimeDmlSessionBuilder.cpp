// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#ifdef USE_DML

#include "OnnxruntimeDmlSessionBuilder.h"
#include "OnnxruntimeEngine.h"
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
  winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device_.get(), queue_.get());

#ifndef _WIN64
  auto use_arena = false;
#else
  auto use_arena = true;
#endif
  winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), use_arena);
  
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
  winml_adapter_api->CreateSessionWithoutModel(ort_env, options, &ort_session_raw);
  auto ort_session = UniqueOrtSession(ort_session_raw, ort_api->ReleaseSession);

  *session = ort_session.release();

  return S_OK;
}

HRESULT OnnxruntimeDmlSessionBuilder::Initialize(
    OrtSession* session) {
  RETURN_HR_IF_NULL(E_INVALIDARG, session);
  auto winml_adapter_api = engine_factory_->UseWinmlAdapterApi();
  
  if (auto status = winml_adapter_api->SessionInitialize(session)) {
    return E_FAIL;
  }

  OrtExecutionProvider* ort_provider;
  winml_adapter_api->SessionGetExecutionProvider(session, 0, &ort_provider);
 
  size_t num_providers;
  winml_adapter_api->SessionGetExecutionProvidersCount(session, &num_providers);
  RETURN_HR_IF(E_UNEXPECTED, num_providers != 2);

  winml_adapter_api->DmlExecutionProviderSetDefaultRoundingMode(ort_provider, true);

  // Flush the D3D12 work from the DML execution provider
  winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider);

  return S_OK;
}

#endif USE_DML