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

namespace {

std::vector<std::shared_ptr<onnxruntime::IExecutionProvider>> CreateProviders(
    const std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>>& provider_factories) {
  std::vector<std::shared_ptr<onnxruntime::IExecutionProvider>> execution_providers;
  for (const auto& factory : provider_factories) {
    execution_providers.emplace_back(std::move(factory->CreateProvider()));
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

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetTrainModeOutputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetTrainModeOutputCount();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::TrainingSessionGetEvalModeOutputCount, _In_ const OrtTrainingSession* sess,
                    _Out_ size_t* out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<const onnxruntime::training::api::TrainingSession*>(sess);
  *out = session->GetEvalModeOutputCount();
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

ORT_API(void, OrtTrainingApis::ReleaseTrainingSession, _Frees_ptr_opt_ OrtTrainingSession* session) {
  delete reinterpret_cast<onnxruntime::training::api::TrainingSession*>(session);
}

ORT_API(void, OrtTrainingApis::ReleaseCheckpointState, _Frees_ptr_opt_ OrtCheckpointState* checkpoint_state) {
  delete reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state);
}

static constexpr OrtTrainingApi ort_training_api = {
    &OrtTrainingApis::LoadCheckpoint,
    &OrtTrainingApis::SaveCheckpoint,
    &OrtTrainingApis::CreateTrainingSession,
    &OrtTrainingApis::TrainingSessionGetTrainModeOutputCount,
    &OrtTrainingApis::TrainingSessionGetEvalModeOutputCount,
    &OrtTrainingApis::ResetGrad,
    &OrtTrainingApis::TrainStep,
    &OrtTrainingApis::EvalStep,
    &OrtTrainingApis::OptimizerStep,
    &OrtTrainingApis::ReleaseTrainingSession,
    &OrtTrainingApis::ReleaseCheckpointState,
};

ORT_API(const OrtTrainingApi*, OrtTrainingApis::GetTrainingApi, uint32_t) {
  // No constraints on the API version yet.
  return &ort_training_api;
}
