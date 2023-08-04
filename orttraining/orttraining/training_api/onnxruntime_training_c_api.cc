// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/onnxruntime_training_c_api.h"
#include "core/common/string_helper.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/random_seed.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/ort_training_apis.h"
#include "orttraining/training_api/training_session.h"

namespace {

std::vector<std::shared_ptr<onnxruntime::IExecutionProvider>> CreateProviders(
    const std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>>& provider_factories) {
  std::vector<std::shared_ptr<onnxruntime::IExecutionProvider>> execution_providers;
  execution_providers.reserve(provider_factories.size());
  for (const auto& factory : provider_factories) {
    execution_providers.emplace_back(factory->CreateProvider());
  }

  return execution_providers;
}

}  // namespace

ORT_API_STATUS_IMPL(OrtTrainingApis::CreateTrainingSession, _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const ORTCHAR_T* train_model_path, _In_ const ORTCHAR_T* eval_model_path,
                    _In_ const ORTCHAR_T* optimizer_model_path, _Outptr_ OrtTrainingSession** out) {
  API_IMPL_BEGIN
  if (options != nullptr && options->value.config_options.GetConfigOrDefault(kOrtSessionOptionsConfigUseEnvAllocators, "0") == "1") {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Use Env Allocators is not supported for on device training.");
  }
  std::unique_ptr<onnxruntime::training::api::TrainingSession> train_sess;
  auto chkpt_state = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state);
  OrtStatus* status = nullptr;
  *out = nullptr;

  ORT_TRY {
    using ProvidersType = std::vector<std::shared_ptr<onnxruntime::IExecutionProvider>>;
    train_sess = std::make_unique<onnxruntime::training::api::TrainingSession>(
        env->GetEnvironment(),
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        options == nullptr ? ProvidersType() : CreateProviders(options->provider_factories),
        chkpt_state,
        onnxruntime::training::api::ModelIdentifiers(
            onnxruntime::ToUTF8String(train_model_path),
            eval_model_path ? std::optional<std::string>(onnxruntime::ToUTF8String(eval_model_path))
                            : std::nullopt,
            optimizer_model_path ? std::optional<std::string>(onnxruntime::ToUTF8String(optimizer_model_path))
                                 : std::nullopt),
        options == nullptr ? gsl::span<OrtCustomOpDomain* const>() : options->custom_op_domains_);

    *out = reinterpret_cast<OrtTrainingSession*>(train_sess.release());
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainingModelOutputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetTrainingModelOutputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModelOutputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetEvalModelOutputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainingModelOutputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetTrainingModelOutputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModelOutputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetEvalModelOutputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::LazyResetGrad, _Inout_ OrtTrainingSession* session) {
  API_IMPL_BEGIN
  auto train_session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(session);
  ORT_API_RETURN_IF_STATUS_NOT_OK(train_session->LazyResetGrad());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainStep, _Inout_ OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options, _In_ size_t inputs_len,
                    _In_reads_(inputs_len) const OrtValue* const* inputs, _In_ size_t outputs_len,
                    _Inout_updates_all_(outputs_len) OrtValue** outputs) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);

  std::vector<OrtValue> feeds(inputs_len);

  for (size_t i = 0; i != inputs_len; ++i) {
    feeds[i] = *reinterpret_cast<const ::OrtValue*>(inputs[i]);
  }

  // Create output feed
  std::vector<OrtValue> fetches(outputs_len);
  for (size_t i = 0; i != outputs_len; ++i) {
    if (outputs[i] != nullptr) {
      ::OrtValue& value = *(outputs[i]);
      fetches[i] = value;
    }
  }
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions op;
    status = session->TrainStep(op, feeds, fetches);
  } else {
    status = session->TrainStep(*run_options, feeds, fetches);
  }

  if (!status.IsOK())
    return onnxruntime::ToOrtStatus(status);
  for (size_t i = 0; i != outputs_len; ++i) {
    ::OrtValue& value = fetches[i];
    if (outputs[i] == nullptr) {
      outputs[i] = std::make_unique<OrtValue>(value).release();
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::EvalStep, _In_ const OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options, _In_ size_t inputs_len,
                    _In_reads_(inputs_len) const OrtValue* const* inputs, _In_ size_t outputs_len,
                    _Inout_updates_all_(outputs_len) OrtValue** outputs) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);

  std::vector<OrtValue> feeds(inputs_len);

  for (size_t i = 0; i != inputs_len; ++i) {
    feeds[i] = *reinterpret_cast<const ::OrtValue*>(inputs[i]);
  }

  // Create output feed
  std::vector<OrtValue> fetches(outputs_len);
  for (size_t i = 0; i != outputs_len; ++i) {
    if (outputs[i] != nullptr) {
      ::OrtValue& value = *(outputs[i]);
      fetches[i] = value;
    }
  }
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions op;
    status = session->EvalStep(op, feeds, fetches);
  } else {
    status = session->EvalStep(*run_options, feeds, fetches);
  }

  if (!status.IsOK())
    return onnxruntime::ToOrtStatus(status);
  for (size_t i = 0; i != outputs_len; ++i) {
    ::OrtValue& value = fetches[i];
    if (outputs[i] == nullptr) {
      outputs[i] = std::make_unique<OrtValue>(value).release();
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetLearningRate, _Inout_ OrtTrainingSession* sess,
                    _In_ float learning_rate) {
  API_IMPL_BEGIN

  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->SetLearningRate(learning_rate));

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetLearningRate, _Inout_ OrtTrainingSession* sess,
                    _Out_ float* learning_rate) {
  API_IMPL_BEGIN

  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  *learning_rate = session->GetLearningRate();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::OptimizerStep, _Inout_ OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  if (run_options == nullptr) {
    OrtRunOptions op;
    ORT_API_RETURN_IF_STATUS_NOT_OK(session->OptimizerStep(op));
  } else {
    ORT_API_RETURN_IF_STATUS_NOT_OK(session->OptimizerStep(*run_options));
  }

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::RegisterLinearLRScheduler, _Inout_ OrtTrainingSession* sess,
                    _In_ const int64_t warmup_step_count,
                    _In_ const int64_t total_step_count,
                    _In_ const float initial_lr) {
  API_IMPL_BEGIN

  OrtStatus* status = nullptr;

  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      session->RegisterScheduler([=](auto optimizer) {
        return std::make_unique<onnxruntime::training::api::LinearLRScheduler>(
            optimizer, warmup_step_count, total_step_count);
      },
                                 initial_lr));

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SchedulerStep, _Inout_ OrtTrainingSession* sess) {
  API_IMPL_BEGIN

  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->SchedulerStep());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path,
                    _Outptr_ OrtCheckpointState** checkpoint_state) {
  API_IMPL_BEGIN
  *checkpoint_state = nullptr;
  auto chkpt_state = std::make_unique<onnxruntime::training::api::CheckpointState>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(onnxruntime::training::api::LoadCheckpoint(checkpoint_path, *chkpt_state));
  *checkpoint_state = reinterpret_cast<OrtCheckpointState*>(chkpt_state.release());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SaveCheckpoint, _In_ OrtCheckpointState* checkpoint_state,
                    _In_ const ORTCHAR_T* checkpoint_path, const bool include_optimizer_state) {
  API_IMPL_BEGIN
  auto chkpt_state = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state);
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      onnxruntime::training::api::SaveCheckpoint(*chkpt_state, checkpoint_path, include_optimizer_state));

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::LoadCheckpointFromBuffer, _In_ const void* checkpoint_buffer,
                    _In_ const size_t num_bytes, _Outptr_ OrtCheckpointState** checkpoint_state) {
  API_IMPL_BEGIN

  *checkpoint_state = nullptr;
  auto chkpt_state = std::make_unique<onnxruntime::training::api::CheckpointState>();
  const auto* checkpoint_bytes = reinterpret_cast<const uint8_t*>(checkpoint_buffer);
  gsl::span checkpoint_span(checkpoint_bytes, num_bytes);
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      onnxruntime::training::api::LoadCheckpointFromBuffer(checkpoint_span, *chkpt_state));
  *checkpoint_state = reinterpret_cast<OrtCheckpointState*>(chkpt_state.release());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetParametersSize, _Inout_ OrtTrainingSession* sess,
                    _Out_ size_t* out, bool trainable_only) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetParametersSize(trainable_only);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::CopyParametersToBuffer, _Inout_ OrtTrainingSession* sess,
                    _Inout_ OrtValue* parameters_buffer, bool trainable_only) {
  API_IMPL_BEGIN
  if (parameters_buffer == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "parameters_buffer is null.");
  }
  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->CopyParametersToBuffer(*parameters_buffer, trainable_only));

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::CopyBufferToParameters, _Inout_ OrtTrainingSession* sess,
                    _Inout_ OrtValue* parameters_buffer, bool trainable_only) {
  API_IMPL_BEGIN
  if (parameters_buffer == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "parameters_buffer is null.");
  }
  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->CopyBufferToParameters(*parameters_buffer, trainable_only));

  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtTrainingApis::ReleaseTrainingSession, _Frees_ptr_opt_ OrtTrainingSession* session) {
  delete reinterpret_cast<onnxruntime::training::api::TrainingSession*>(session);
}

ORT_API(void, OrtTrainingApis::ReleaseCheckpointState, _Frees_ptr_opt_ OrtCheckpointState* checkpoint_state) {
  delete reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::ExportModelForInferencing, _Inout_ OrtTrainingSession* sess,
                    _In_ const ORTCHAR_T* inference_model_path, size_t graph_outputs_len,
                    _In_reads_(graph_outputs_len) const char* const* graph_output_names) {
  API_IMPL_BEGIN

  OrtStatus* status = nullptr;

#if !defined(ORT_MINIMAL_BUILD)
  if (graph_outputs_len == 0U) {
    return OrtApis::CreateStatus(
        ORT_INVALID_ARGUMENT,
        "Empty array of graph output names is not valid. Please provide valid graph output names");
  }

  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);

  onnxruntime::InlinedVector<std::string> output_names(graph_outputs_len);

  for (size_t i = 0; i != graph_outputs_len; ++i) {
    if (graph_output_names[i] == nullptr || graph_output_names[i][0] == '\0') {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "Name of graph output cannot be empty. Please provide valid graph names");
    }

    output_names[i] = graph_output_names[i];
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(
      session->ExportModelForInferencing(onnxruntime::ToUTF8String(inference_model_path), output_names));
#else
  ORT_UNUSED_PARAMETER(sess);
  ORT_UNUSED_PARAMETER(inference_model_path);
  ORT_UNUSED_PARAMETER(graph_outputs_len);
  ORT_UNUSED_PARAMETER(graph_output_names);
  status = OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "Inference model cannot be exported in a minimal training build.");
#endif

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetSeed, _In_ const int64_t seed) {
  API_IMPL_BEGIN

  onnxruntime::utils::SetRandomSeed(seed);
  return nullptr;

  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainingModelInputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetTrainingModelInputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModelInputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetEvalModelInputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainingModelInputName, _In_ const OrtTrainingSession* sess,
                    size_t index, _In_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetTrainingModelInputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModelInputName, _In_ const OrtTrainingSession* sess,
                    size_t index, _In_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetEvalModelInputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::AddProperty, _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const char* property_name, _In_ enum OrtPropertyType property_type,
                    _In_ void* property_value) {
  API_IMPL_BEGIN

  OrtStatus* status = nullptr;

  auto chkpt_state = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state);

  switch (property_type) {
    case OrtPropertyType::OrtIntProperty: {
      int64_t* value = reinterpret_cast<int64_t*>(property_value);
      chkpt_state->property_bag.AddProperty(property_name, *value);
      break;
    }
    case OrtPropertyType::OrtFloatProperty: {
      float* value = reinterpret_cast<float*>(property_value);
      chkpt_state->property_bag.AddProperty(property_name, *value);
      break;
    }
    case OrtPropertyType::OrtStringProperty: {
      char* value = reinterpret_cast<char*>(property_value);
      chkpt_state->property_bag.AddProperty(property_name, value);
      break;
    }
    default: {
      std::ostringstream stream;
      stream << "Given property type: " << property_type << " is not supported.";
      status = OrtApis::CreateStatus(ORT_FAIL, stream.str().c_str());
      break;
    }
  }

  return status;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetProperty, _In_ const OrtCheckpointState* checkpoint_state,
                    _In_ const char* property_name, _Inout_ OrtAllocator* allocator,
                    _Out_ enum OrtPropertyType* property_type, _Outptr_ void** property_value) {
  API_IMPL_BEGIN

  OrtStatus* status = nullptr;

  auto chkpt_state = reinterpret_cast<const onnxruntime::training::api::CheckpointState*>(checkpoint_state);
  const auto value = chkpt_state->property_bag.GetProperty<
      onnxruntime::training::api::PropertyDataType>(property_name);

  if (std::holds_alternative<int64_t>(value)) {
    int64_t* value_p = reinterpret_cast<int64_t*>(allocator->Alloc(allocator, sizeof(int64_t)));
    if (!value_p) {
      return OrtApis::CreateStatus(ORT_FAIL, "Int property value buffer allocation failed.");
    }
    *value_p = std::get<int64_t>(value);
    *(reinterpret_cast<int64_t**>(property_value)) = value_p;
    *property_type = OrtPropertyType::OrtIntProperty;
  } else if (std::holds_alternative<float>(value)) {
    float* value_p = reinterpret_cast<float*>(allocator->Alloc(allocator, sizeof(float)));
    if (!value_p) {
      return OrtApis::CreateStatus(ORT_FAIL, "Float property value buffer allocation failed.");
    }
    *value_p = std::get<float>(value);
    *(reinterpret_cast<float**>(property_value)) = value_p;
    *property_type = OrtPropertyType::OrtFloatProperty;
  } else if (std::holds_alternative<std::string>(value)) {
    auto property_value_str = std::get<std::string>(value);
    // property_value_str.length() + 1 for null termination of c strings
    auto buffer = reinterpret_cast<char*>(allocator->Alloc(allocator, property_value_str.length() + 1));
    memcpy(buffer, property_value_str.c_str(), property_value_str.length());
    buffer[property_value_str.length()] = '\0';
    *(reinterpret_cast<char**>(property_value)) = buffer;
    *property_type = OrtPropertyType::OrtStringProperty;
  } else {
    std::ostringstream stream;
    stream << "Unknown type for property: " << property_name;
    status = OrtApis::CreateStatus(ORT_FAIL, stream.str().c_str());
  }

  return status;
  API_IMPL_END
}

static constexpr OrtTrainingApi ort_training_api = {
    // NOTE: The C# bindings depend on the API order within this struct. Since Training APIs are not officially
    // released, it is OK to change the order here, however a corresponding matching change should also be done in the
    // "OrtTrainingApi" struct in NativeTrainingMethods.shared.cs
    &OrtTrainingApis::LoadCheckpoint,
    &OrtTrainingApis::SaveCheckpoint,
    &OrtTrainingApis::CreateTrainingSession,
    &OrtTrainingApis::TrainingSessionGetTrainingModelOutputCount,
    &OrtTrainingApis::TrainingSessionGetEvalModelOutputCount,
    &OrtTrainingApis::TrainingSessionGetTrainingModelOutputName,
    &OrtTrainingApis::TrainingSessionGetEvalModelOutputName,
    &OrtTrainingApis::LazyResetGrad,
    &OrtTrainingApis::TrainStep,
    &OrtTrainingApis::EvalStep,
    &OrtTrainingApis::SetLearningRate,
    &OrtTrainingApis::GetLearningRate,
    &OrtTrainingApis::OptimizerStep,
    &OrtTrainingApis::RegisterLinearLRScheduler,
    &OrtTrainingApis::SchedulerStep,
    &OrtTrainingApis::GetParametersSize,
    &OrtTrainingApis::CopyParametersToBuffer,
    &OrtTrainingApis::CopyBufferToParameters,
    &OrtTrainingApis::ReleaseTrainingSession,
    &OrtTrainingApis::ReleaseCheckpointState,
    &OrtTrainingApis::ExportModelForInferencing,
    &OrtTrainingApis::SetSeed,
    &OrtTrainingApis::TrainingSessionGetTrainingModelInputCount,
    &OrtTrainingApis::TrainingSessionGetEvalModelInputCount,
    &OrtTrainingApis::TrainingSessionGetTrainingModelInputName,
    &OrtTrainingApis::TrainingSessionGetEvalModelInputName,
    &OrtTrainingApis::AddProperty,
    &OrtTrainingApis::GetProperty,
    &OrtTrainingApis::LoadCheckpointFromBuffer};

ORT_API(const OrtTrainingApi*, OrtTrainingApis::GetTrainingApi, uint32_t) {
  // No constraints on the API version yet.
  return &ort_training_api;
}
