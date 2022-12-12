// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/onnxruntime_training_c_api.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/abi_session_options_impl.h"
#include "orttraining/training_api/include/checkpoint.h"
#include "orttraining/training_api/include/training_session.h"
#include "orttraining/training_api/include/ort_training_apis.h"
#include "core/common/string_helper.h"

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
        chkpt_state->module_checkpoint_state.named_parameters,
        onnxruntime::training::api::ModelIdentifiers(
            onnxruntime::ToUTF8String(train_model_path),
            eval_model_path ? std::optional<std::string>(onnxruntime::ToUTF8String(eval_model_path))
                            : std::nullopt,
            optimizer_model_path ? std::optional<std::string>(onnxruntime::ToUTF8String(optimizer_model_path))
                                 : std::nullopt));

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

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainingModelOutputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetTrainingModelOutputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModelOutputName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  std::string name = session->GetEvalModelOutputName(index);
  *output = onnxruntime::StrDup(name, allocator);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::ResetGrad, _Inout_ OrtTrainingSession* session) {
  API_IMPL_BEGIN
  auto train_session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(session);
  ORT_API_RETURN_IF_STATUS_NOT_OK(train_session->ResetGrad());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainStep, _Inout_ OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options, size_t inputs_len,
                    _In_reads_(inputs_len) const OrtValue* const* inputs, size_t outputs_len,
                    _Inout_updates_all_(outputs_len) OrtValue** outputs) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<onnxruntime::training::api::TrainingSession*>(sess);
  constexpr int queue_id = 0;

  std::vector<OrtValue> feeds(inputs_len);

  for (size_t i = 0; i != inputs_len; ++i) {
    auto& ort_value = feeds[i] = *reinterpret_cast<const ::OrtValue*>(inputs[i]);
    if (ort_value.Fence()) {
      ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    }
  }

  // Create output feed
  std::vector<OrtValue> fetches(outputs_len);
  for (size_t i = 0; i != outputs_len; ++i) {
    if (outputs[i] != nullptr) {
      ::OrtValue& value = *(outputs[i]);
      if (value.Fence())
        value.Fence()->BeforeUsingAsOutput(onnxruntime::kCpuExecutionProvider, queue_id);
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
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (outputs[i] == nullptr) {
      outputs[i] = new OrtValue(value);
    }
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::EvalStep, _In_ const OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options, size_t inputs_len,
                    _In_reads_(inputs_len) const OrtValue* const* inputs, size_t outputs_len,
                    _Inout_updates_all_(outputs_len) OrtValue** outputs) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  constexpr int queue_id = 0;

  std::vector<OrtValue> feeds(inputs_len);

  for (size_t i = 0; i != inputs_len; ++i) {
    auto& ort_value = feeds[i] = *reinterpret_cast<const ::OrtValue*>(inputs[i]);

    if (ort_value.Fence()) ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
  }

  // Create output feed
  std::vector<OrtValue> fetches(outputs_len);
  for (size_t i = 0; i != outputs_len; ++i) {
    if (outputs[i] != nullptr) {
      ::OrtValue& value = *(outputs[i]);
      if (value.Fence())
        value.Fence()->BeforeUsingAsOutput(onnxruntime::kCpuExecutionProvider, queue_id);
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
    if (value.Fence())
      value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    if (outputs[i] == nullptr) {
      outputs[i] = new OrtValue(value);
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
                                 std::optional<float>(initial_lr)));

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

ORT_API_STATUS_IMPL(OrtTrainingApis::SaveCheckpoint, _In_ const ORTCHAR_T* checkpoint_path,
                    _In_ const OrtTrainingSession* sess, bool save_optimizer_state) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  onnxruntime::training::api::CheckpointState chkpt_state;
  ORT_API_RETURN_IF_STATUS_NOT_OK(session->CreateCheckpointState(chkpt_state, save_optimizer_state));
  ORT_API_RETURN_IF_STATUS_NOT_OK(onnxruntime::training::api::SaveCheckpoint(chkpt_state, checkpoint_path));

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

  return nullptr;
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
    &OrtTrainingApis::ResetGrad,
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
};

ORT_API(const OrtTrainingApi*, OrtTrainingApis::GetTrainingApi, uint32_t) {
  // No constraints on the API version yet.
  return &ort_training_api;
}
