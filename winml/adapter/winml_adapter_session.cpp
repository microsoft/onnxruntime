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
#include "core/session/ort_env.h"

#include "winml_adapter_model.h"
#include "core/framework/utils.h"

#ifdef USE_DML
#include "core/providers/dml/DmlExecutionProvider/src/AbiCustomRegistry.h"
#include "abi_custom_registry_impl.h"
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
    return onnxruntime::InferenceSession::GetSessionState();
  }
};

ORT_API_STATUS_IMPL(winmla::CreateSessionWithoutModel, _In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> inference_session;
  try {
    // Create the inference session
    inference_session = std::make_unique<onnxruntime::InferenceSession>(options->value, env->GetEnvironment());
  } catch (const std::exception& e) {
    return OrtApis::CreateStatus(ORT_FAIL, e.what());
  }

  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>> provider_list;
  if (options) {
    for (auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      if (provider->Type() == onnxruntime::kDmlExecutionProvider) {
        if (options->value.enable_mem_pattern) {
          // TODO Instead of returning an error, should we set mem pattern to false here and log a warning saying so?
          // Doing so would be inconsistent with the Python API that doesn't go through this code path.
          return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Mem pattern should be disabled when using DML execution provider.");
        }
        if (options->value.execution_mode != ExecutionMode::ORT_SEQUENTIAL) {
          return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Sequential execution should be enabled when using DML execution provider.");
        }
      }
      provider_list.push_back(std::move(provider));
    }
  }

  Status status;
  if (options) {
    if (!options->custom_op_domains_.empty()) {
      status = inference_session->AddCustomOpDomains(options->custom_op_domains_);
      if (!status.IsOK())
        return onnxruntime::ToOrtStatus(status);
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      inference_session->RegisterExecutionProvider(std::move(provider));
    }
  }

  *session = reinterpret_cast<OrtSession*>(inference_session.release());

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
#ifdef USE_DML
    // Down-cast to the concrete type.
    // The only supported input is the AbiCustomRegistry type.
    // Other implementations of IMLOperatorRegistry are forbidden.
    auto abi_custom_registry =
        static_cast<winmla::AbiCustomRegistry*>(registry);

    // Get the ORT registry
    return abi_custom_registry->GetRegistries();
#endif  // USE_DML
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

ORT_API_STATUS_IMPL(winmla::CreateCustomRegistry, _Out_ IMLOperatorRegistry** registry) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto impl = wil::MakeOrThrow<winmla::AbiCustomRegistryImpl>();
  *registry = impl.Detach();
#else
  *registry = nullptr;
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

static OrtDevice GetSessionGetInputDevice(_In_ OrtSession* session, _In_ const char* const input_name) {
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);
  const onnxruntime::SessionState& session_state = session_protected_load_accessor->GetSessionState();

  std::vector<onnxruntime::SessionState::NodeInfo> node_info_vec;
  session_state.GetInputNodeInfo(input_name, node_info_vec);
  const auto& node_info = node_info_vec.front();  // all consumers of a feed have the same device so first entry is fine
  return *node_info.device;
}

ORT_API_STATUS_IMPL(winmla::SessionGetInputRequiredDeviceId, _In_ OrtSession* session, _In_ const char* const input_name, _Out_ int16_t* device_id) {
  auto device = GetSessionGetInputDevice(session, input_name);
  *device_id = device.Id();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::ValueGetDeviceId, _In_ OrtValue* ort_value, _Out_ int16_t* device_id) {
  auto device = ort_value->Get<onnxruntime::Tensor>().Location().device;
  *device_id = device.Id();
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const char* const input_name,
                    _In_ OrtValue* orig_value, _Outptr_ OrtValue** new_value) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_protected_load_accessor =
      static_cast<InferenceSessionProtectedLoadAccessor*>(inference_session);
  const onnxruntime::SessionState& session_state = session_protected_load_accessor->GetSessionState();

  auto ort_value = std::make_unique<OrtValue>();
  auto status = onnxruntime::utils::CopyOneInputAcrossDevices(session_state, input_name, *orig_value, *ort_value.get());
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }

  *new_value = ort_value.release();

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionGetNumberOfIntraOpThreads, _In_ OrtSession* session, _Out_ uint32_t* num_threads) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_options = inference_session->GetSessionOptions();
  *num_threads = session_options.intra_op_param.thread_pool_size;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionGetIntraOpThreadSpinning, _In_ OrtSession* session, _Out_ bool* allow_spinning) {
    API_IMPL_BEGIN
    auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
    auto session_options = inference_session->GetSessionOptions();
    auto iter = session_options.session_configurations.find("session.intra_op.allow_spinning");
    *allow_spinning = iter != session_options.session_configurations.cend() && iter->second == "0";
    return nullptr;
    API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::SessionGetNamedDimensionsOverrides, _In_ OrtSession* session, _Out_ winrt::Windows::Foundation::Collections::IMapView<winrt::hstring, uint32_t>& named_dimension_overrides) {
  API_IMPL_BEGIN
  auto inference_session = reinterpret_cast<::onnxruntime::InferenceSession*>(session);
  auto session_options = inference_session->GetSessionOptions();
  winrt::Windows::Foundation::Collections::IMap<winrt::hstring, uint32_t> override_map = winrt::single_threaded_map<winrt::hstring, uint32_t>();
  for (auto freeDimOverride : session_options.free_dimension_overrides)
  {
    if (freeDimOverride.dim_identifer_type == onnxruntime::FreeDimensionOverrideType::Name) 
    {
      override_map.Insert(winrt::to_hstring(freeDimOverride.dim_identifier), static_cast<uint32_t>(freeDimOverride.dim_value));
    }
  }
  named_dimension_overrides = override_map.GetView();
  return nullptr;
  API_IMPL_END
}