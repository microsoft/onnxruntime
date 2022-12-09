// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_cxx_api.h"

namespace Ort {

inline TrainingSession::TrainingSession(const SessionOptions& session_options,
                                        CheckpointState& checkpoint_state,
                                        const std::basic_string<ORTCHAR_T>& train_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path) {
  Env env = Env();
  ThrowOnError(GetTrainingApi().CreateTrainingSession(
      env, session_options, checkpoint_state,
      train_model_path.c_str(),
      eval_model_path.has_value() ? eval_model_path.value().c_str() : nullptr,
      optimizer_model_path.has_value() ? optimizer_model_path.value().c_str() : nullptr,
      &p_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetTrainingModelOutputCount(p_, &training_model_output_count_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetEvalModelOutputCount(p_, &eval_model_output_count_));
}

inline std::vector<Value> TrainingSession::TrainStep(const std::vector<Value>& input_values) {
  std::vector<Value> output_values;
  output_values.reserve(training_model_output_count_);
  for (size_t i = 0; i < training_model_output_count_; i++) output_values.emplace_back(nullptr);
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values.data());
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values.data());
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().TrainStep(
      p_, run_options, input_values.size(), ort_input_values,
      training_model_output_count_, ort_output_values));

  return output_values;
}

inline void TrainingSession::LazyResetGrad() {
  ThrowOnError(GetTrainingApi().LazyResetGrad(p_));
}

inline std::vector<Value> TrainingSession::EvalStep(const std::vector<Value>& input_values) {
  std::vector<Value> output_values;
  output_values.reserve(eval_model_output_count_);
  for (size_t i = 0; i < eval_model_output_count_; i++) output_values.emplace_back(nullptr);
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values.data());
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values.data());
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().EvalStep(
      p_, run_options, input_values.size(), ort_input_values,
      training_model_output_count_, ort_output_values));

  return output_values;
}

inline void TrainingSession::SetLearningRate(float learning_rate) {
  ThrowOnError(GetTrainingApi().SetLearningRate(p_, learning_rate));
}

inline float TrainingSession::GetLearningRate() const {
  float learning_rate = 0;
  ThrowOnError(GetTrainingApi().GetLearningRate(p_, &learning_rate));
  return learning_rate;
}

inline void TrainingSession::RegisterLinearLRScheduler(int64_t warmup_step_count, int64_t total_step_count,
                                                       float initial_lr) {
  ThrowOnError(GetTrainingApi().RegisterLinearLRScheduler(p_, warmup_step_count, total_step_count,
                                                          initial_lr));
}

inline void TrainingSession::SchedulerStep() {
  ThrowOnError(GetTrainingApi().SchedulerStep(p_));
}

inline void TrainingSession::OptimizerStep() {
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().OptimizerStep(p_, run_options));
}

inline CheckpointState CheckpointState::LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint) {
  OrtCheckpointState* checkpoint_state;
  ThrowOnError(GetTrainingApi().LoadCheckpoint(path_to_checkpoint.c_str(), &checkpoint_state));
  return CheckpointState(checkpoint_state);
}

inline void CheckpointState::SaveCheckpoint(const TrainingSession& session,
                                            const std::basic_string<ORTCHAR_T>& path_to_checkpoint,
                                            bool include_optimizer_states) {
  ThrowOnError(GetTrainingApi().SaveCheckpoint(path_to_checkpoint.c_str(), session, include_optimizer_states));
}

inline void TrainingSession::ExportModelForInferencing(const std::basic_string<ORTCHAR_T>& inference_model_path,
                                                       const std::vector<std::string>& graph_output_names) {
  std::vector<const char*> output_names;
  output_names.reserve(graph_output_names.size());
  for (const auto& output_name : graph_output_names) {
    output_names.push_back(output_name.c_str());
  }
  ThrowOnError(GetTrainingApi().ExportModelForInferencing(
      p_, inference_model_path.c_str(), graph_output_names.size(), output_names.data()));
}

inline void SetSeed(const int64_t seed) {
  ThrowOnError(GetTrainingApi().SetSeed(seed));
}

}  // namespace Ort
