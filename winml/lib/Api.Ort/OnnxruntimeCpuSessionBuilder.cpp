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
//
//  // Create the inference session
//  auto session = std::make_unique<onnxruntime::InferenceSession>(options->value);
//
//  // Create the cpu execution provider
//  onnxruntime::CPUExecutionProviderInfo xpInfo;
//#ifndef _WIN64
//  xpInfo.create_arena = false;
//#endif
//  auto cpu_provider = std::make_unique<onnxruntime::CPUExecutionProvider>(xpInfo);
//
//  // Cache the provider's raw pointer
//  *pp_provider = cpu_provider.get();
//
//  // Register the cpu xp
//  ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(cpu_provider)));
//
//  // assign the session to the out parameter
//  auto sessionptr = wil::MakeOrThrow<winmla::InferenceSession>(session.release());
//  RETURN_IF_FAILED(sessionptr.CopyTo(_uuidof(winmla::IInferenceSession), (void**)p_session));
//
  return S_OK;
}

HRESULT
OnnxruntimeCpuSessionBuilder::Initialize(
    OrtSession* session,
    OrtExecutionProvider* /*p_provider*/
    ) {
//    ORT_THROW_IF_ERROR(session->get()->Initialize());
  return S_OK;
}
