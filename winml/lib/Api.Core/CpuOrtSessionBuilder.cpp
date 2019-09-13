// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#include "CpuOrtSessionBuilder.h"

// winml includes
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"

// ort includes
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"

using namespace Windows::AI::MachineLearning;

HRESULT
CpuOrtSessionBuilder::CreateSessionOptions(
    onnxruntime::SessionOptions* p_options) {
  RETURN_HR_IF_NULL(E_POINTER, p_options);

  *p_options = onnxruntime::SessionOptions();
  p_options->graph_optimization_level = onnxruntime::TransformerLevel::Level3;

  // Onnxruntime will use half the number of concurrent threads supported on the system
  // by default. This causes MLAS to not exercise every logical core.
  // We force the thread pool size to be maxxed out to ensure that WinML always
  // runs the fastest.
  p_options->session_thread_pool_size = std::thread::hardware_concurrency();

  return S_OK;
}

HRESULT
CpuOrtSessionBuilder::CreateSession(
    const onnxruntime::SessionOptions& options,
    std::unique_ptr<onnxruntime::InferenceSession>* p_session,
    onnxruntime::IExecutionProvider** pp_provider) {
  RETURN_HR_IF_NULL(E_POINTER, p_session);
  RETURN_HR_IF_NULL(E_POINTER, pp_provider);
  RETURN_HR_IF(E_POINTER, *pp_provider != nullptr);

  // Create the inference session
  auto session = std::make_unique<onnxruntime::InferenceSession>(options);

  // Create the cpu execution provider
  onnxruntime::CPUExecutionProviderInfo xpInfo;
#ifndef _WIN64
  xpInfo.create_arena = false;
#endif
  auto cpu_provider = std::make_unique<onnxruntime::CPUExecutionProvider>(xpInfo);

  // Cache the provider's raw pointer
  *pp_provider = cpu_provider.get();

  // Register the cpu xp
  WINML_THROW_IF_NOT_OK(session->RegisterExecutionProvider(std::move(cpu_provider)));

  // assign the session to the out parameter
  *p_session = std::move(session);

  return S_OK;
}

HRESULT
CpuOrtSessionBuilder::Initialize(
    onnxruntime::InferenceSession* p_session,
    onnxruntime::IExecutionProvider* /*p_provider*/
) {
  WINML_THROW_IF_NOT_OK(p_session->Initialize());
  return S_OK;
}