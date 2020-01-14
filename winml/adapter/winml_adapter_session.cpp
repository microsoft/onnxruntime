// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"

#include "core/session/inference_session.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_env.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

// ORT intentionally requires callers derive from their session class to access
// the protected methods used below.
class InferenceSessionProtectedLoadAccessor : public onnxruntime::InferenceSession {
 public:
  onnxruntime::common::Status
  Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) {
    return onnxruntime::InferenceSession::Load(std::move(p_model_proto));
  }
  const onnxruntime::SessionState& GetSessionState() {
    return *session_state_;
  }
};

ORT_API_STATUS_IMPL(winmla::CreateSessionWihtoutModel, _In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session) {
  API_IMPL_BEGIN
  try {
    // Create the inference session
    *session = reinterpret_cast<OrtSession*>(new onnxruntime::InferenceSession(options->value, env->GetLoggingManager()));
  } catch (const std::exception& e) {
    return OrtApis::CreateStatus(ORT_FAIL, e.what());
  }
  return nullptr;
  API_IMPL_END
}