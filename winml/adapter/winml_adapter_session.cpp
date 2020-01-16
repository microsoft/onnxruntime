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

#include "winml_adapter_model.h"
#include "core/framework/utils.h"


#include "core/providers/dml/DmlExecutionProvider/src/AbiCustomRegistry.h"

#ifdef USE_DML
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#endif USE_DML

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

ORT_API_STATUS_IMPL(winmla::CreateSessionWithoutModel, _In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session) {
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

ORT_API_STATUS_IMPL(winmla::SessionGetExecutionProvidersCount, _In_ OrtSession* session, _Out_ size_t* count) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);
  auto& session_state = session_protected_load_accessor->GetSessionState();
  *count = session_state.GetExecutionProviders().NumProviders();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionGetExecutionProvider, _In_ OrtSession* session, _In_ size_t index, _Out_ OrtExecutionProvider** ort_provider) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);
  const auto& session_state = session_protected_load_accessor->GetSessionState();
  auto& provider_id = session_state.GetExecutionProviders().GetIds().at(index);
  const auto& provider = session_state.GetExecutionProviders().Get(provider_id);

  *ort_provider = const_cast<OrtExecutionProvider*>(reinterpret_cast<const OrtExecutionProvider*>(provider));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionInitialize, _In_ OrtSession* session) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto status = inference_session->Initialize();
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionLoadAndPurloinModel, _In_ OrtSession* session, _In_ OrtModel* model) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);

  auto status = session_protected_load_accessor->Load(model->DetachModelProto());

  ReleaseModel(model);

  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionStartProfiling, _In_ OrtEnv* env, _In_ OrtSession* session) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  inference_session->StartProfiling(&env->GetLoggingManager()->DefaultLogger());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionEndProfiling, _In_ OrtSession* session) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  inference_session->EndProfiling();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionRegisterGraphTransformers, _In_ OrtSession* session) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);

  // Bug 22973884 : Fix issues with BatchNorm + Add and BatchNorm + Mul handling implicit inputs, and move from Winml to ORT
  GraphTransformerHelpers::RegisterGraphTransformers(inference_session);
#endif USE_DML
  return nullptr;
  API_IMPL_END
}

inline std::list<std::shared_ptr<onnxruntime::CustomRegistry>>
GetLotusCustomRegistries(IMLOperatorRegistry* registry) {
  if (registry != nullptr) {
    // Down-cast to the concrete type.
    // The only supported input is the AbiCustomRegistry type.
    // Other implementations of IMLOperatorRegistry are forbidden.
    auto abi_custom_registry =
        static_cast<winmla::AbiCustomRegistry*>(registry);

    // Get the ORT registry
    return abi_custom_registry->GetRegistries();
  }
  return {};
}

ORT_API_STATUS_IMPL(winmla::SessionRegisterCustomRegistry, _In_ OrtSession* session, _In_ IMLOperatorRegistry* registry) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto custom_registries = GetLotusCustomRegistries(registry);

  // Register
  for (auto& custom_registry : custom_registries) {
    ORT_THROW_IF_ERROR(inference_session->RegisterCustomRegistry(custom_registry));
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const char* const input_name,
    _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);
    const onnxruntime::SessionState& sessionState = session_protected_load_accessor->GetSessionState();
  auto ort_value = std::make_unique<OrtValue>();
  auto status = onnxruntime::utils::CopyOneInputAcrossDevices(sessionState, input_name, *orig_value, *ort_value.get());
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }

  *new_value = ort_value.release();

  return nullptr;
  API_IMPL_END
}