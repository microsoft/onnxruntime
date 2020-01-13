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
#include "WinMLAdapter.h"
#include "WinMLAdapterErrors.h"

// winml includes
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"

// ort includes
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/session/abi_session_options_impl.h"

using namespace Windows::AI::MachineLearning;

namespace Windows::AI::MachineLearning::Adapter {

OnnxruntimeCpuSessionBuilder::OnnxruntimeCpuSessionBuilder() {

}

HRESULT
OnnxruntimeCpuSessionBuilder::CreateSessionOptions(
    OrtSessionOptions** options) try {
  RETURN_HR_IF_NULL(E_POINTER, options);

  Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(options));
  Ort::SessionOptions session_options(*options);

  // set the graph optimization level to all (used to be called level 3)
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Onnxruntime will use half the number of concurrent threads supported on the system
  // by default. This causes MLAS to not exercise every logical core.
  // We force the thread pool size to be maxxed out to ensure that WinML always
  // runs the fastest.
  session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());

  // call release() so the underlying OrtSessionOptions object isn't freed
  session_options.release();

  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT
OnnxruntimeCpuSessionBuilder::CreateSession(
    OrtSessionOptions* options,
    winmla::IInferenceSession** p_session,
    onnxruntime::IExecutionProvider** pp_provider) try {
  RETURN_HR_IF_NULL(E_POINTER, p_session);
  RETURN_HR_IF_NULL(E_POINTER, pp_provider);
  RETURN_HR_IF(E_POINTER, *pp_provider != nullptr);

  // Create the inference session
  auto session = std::make_unique<onnxruntime::InferenceSession>(options->value);

  // Create the cpu execution provider
  onnxruntime::CPUExecutionProviderInfo xpInfo;
#ifndef _WIN64
  xpInfo.create_arena = false;
#endif
  auto cpu_provider = std::make_unique<onnxruntime::CPUExecutionProvider>(xpInfo);

  // Cache the provider's raw pointer
  *pp_provider = cpu_provider.get();

  // Register the cpu xp
  ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(cpu_provider)));

  // assign the session to the out parameter
  auto sessionptr = wil::MakeOrThrow<winmla::InferenceSession>(session.release());
  RETURN_IF_FAILED(sessionptr.CopyTo(_uuidof(winmla::IInferenceSession), (void**)p_session));

  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT
OnnxruntimeCpuSessionBuilder::Initialize(
    winmla::IInferenceSession* p_session,
    onnxruntime::IExecutionProvider* /*p_provider*/
) try {
    ORT_THROW_IF_ERROR(p_session->get()->Initialize());
  return S_OK;
}
WINMLA_CATCH_ALL_COM

} // Windows::AI::MachineLearning::Adapter