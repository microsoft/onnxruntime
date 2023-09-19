// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_cxx_api.h"

namespace Ort {

inline TrainingSession::TrainingSession(const Env& env, const SessionOptions& session_options,
                                        CheckpointState& checkpoint_state,
                                        const std::basic_string<ORTCHAR_T>& train_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path) {
  ThrowOnError(GetTrainingApi().CreateTrainingSession(
      env, session_options, checkpoint_state,
      train_model_path.c_str(),
      eval_model_path.has_value() ? eval_model_path.value().c_str() : nullptr,
      optimizer_model_path.has_value() ? optimizer_model_path.value().c_str() : nullptr,
      &p_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetTrainingModelOutputCount(p_, &training_model_output_count_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetEvalModelOutputCount(p_, &eval_model_output_count_));
}

inline TrainingSession::TrainingSession(const Env& env, const SessionOptions& session_options,
                                        CheckpointState& checkpoint_state,
                                        const std::vector<uint8_t>& train_model_data,
                                        const std::vector<uint8_t>& eval_model_data,
                                        const std::vector<uint8_t>& optim_model_data) {
  ThrowOnError(GetTrainingApi().CreateTrainingSessionFromBuffer(
      env, session_options, checkpoint_state,
      train_model_data.data(), train_model_data.size(),
      eval_model_data.data(), eval_model_data.size(),
      optim_model_data.data(), optim_model_data.size(),
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

inline std::vector<std::string> TrainingSession::InputNames(const bool training) {
  auto& input_count_function = training ? GetTrainingApi().TrainingSessionGetTrainingModelInputCount
                                        : GetTrainingApi().TrainingSessionGetEvalModelInputCount;
  auto& input_name_function = training ? GetTrainingApi().TrainingSessionGetTrainingModelInputName
                                       : GetTrainingApi().TrainingSessionGetEvalModelInputName;

  size_t input_count = 0;
  ThrowOnError(input_count_function(p_, &input_count));
  std::vector<std::string> input_names(input_count);
  AllocatorWithDefaultOptions allocator;
  for (size_t index = 0; index < input_count; ++index) {
    char* input_name;
    ThrowOnError(input_name_function(p_, index, allocator, &input_name));
    input_names[index] = std::string(input_name);
    allocator.Free(input_name);
  }

  return input_names;
}

inline std::vector<std::string> TrainingSession::OutputNames(const bool training) {
  auto& output_count_function = training ? GetTrainingApi().TrainingSessionGetTrainingModelOutputCount
                                         : GetTrainingApi().TrainingSessionGetEvalModelOutputCount;
  auto& output_name_function = training ? GetTrainingApi().TrainingSessionGetTrainingModelOutputName
                                        : GetTrainingApi().TrainingSessionGetEvalModelOutputName;

  size_t output_count = 0;
  ThrowOnError(output_count_function(p_, &output_count));
  std::vector<std::string> output_names(output_count);
  AllocatorWithDefaultOptions allocator;
  for (size_t index = 0; index < output_count; ++index) {
    char* output_name;
    ThrowOnError(output_name_function(p_, index, allocator, &output_name));
    output_names[index] = std::string(output_name);
    allocator.Free(output_name);
  }

  return output_names;
}

inline Value TrainingSession::ToBuffer(const bool only_trainable) {
  size_t buffer_size = 0U;
  ThrowOnError(GetTrainingApi().GetParametersSize(p_, &buffer_size, only_trainable));

  std::array<int64_t, 1> buffer_shape{static_cast<int64_t>(buffer_size)};

  AllocatorWithDefaultOptions allocator;
  Value buffer = Value::CreateTensor(allocator, buffer_shape.data(), 1U,
                                     ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  ThrowOnError(GetTrainingApi().CopyParametersToBuffer(p_, buffer, only_trainable));

  return buffer;
}

inline void TrainingSession::FromBuffer(Value& buffer) {
  if (!buffer.IsTensor()) {
    ThrowStatus(Status("Incorrect buffer received. Expected a tensor buffer.", OrtErrorCode::ORT_INVALID_ARGUMENT));
  }

  auto tensor_info = buffer.GetTensorTypeAndShapeInfo();
  auto buffer_shape = tensor_info.GetShape();

  if (buffer_shape.size() != 1U) {
    ThrowStatus(Status("Incorrect buffer received. Expected a contiguous tensor buffer.",
                       OrtErrorCode::ORT_INVALID_ARGUMENT));
  }

  auto buffer_size = buffer_shape.front();

  size_t session_buffer_size_trainable_only = 0U;
  ThrowOnError(GetTrainingApi().GetParametersSize(p_, &session_buffer_size_trainable_only, true));

  if (buffer_size == static_cast<int64_t>(session_buffer_size_trainable_only)) {
    ThrowOnError(GetTrainingApi().CopyBufferToParameters(p_, buffer, true));
    return;
  }

  size_t session_buffer_size = 0U;
  ThrowOnError(GetTrainingApi().GetParametersSize(p_, &session_buffer_size, false));

  if (buffer_size != static_cast<int64_t>(session_buffer_size)) {
    ThrowStatus(Status("Incorrect buffer size received.", OrtErrorCode::ORT_INVALID_ARGUMENT));
  }

  ThrowOnError(GetTrainingApi().CopyBufferToParameters(p_, buffer, false));
}

inline CheckpointState CheckpointState::LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint) {
  OrtCheckpointState* checkpoint_state;
  ThrowOnError(GetTrainingApi().LoadCheckpoint(path_to_checkpoint.c_str(), &checkpoint_state));
  return CheckpointState(checkpoint_state);
}

inline CheckpointState CheckpointState::LoadCheckpointFromBuffer(const std::vector<uint8_t>& buffer) {
  OrtCheckpointState* checkpoint_state;
  ThrowOnError(GetTrainingApi().LoadCheckpointFromBuffer(buffer.data(), buffer.size(), &checkpoint_state));
  return CheckpointState(checkpoint_state);
}

inline void CheckpointState::SaveCheckpoint(const CheckpointState& checkpoint_states,
                                            const std::basic_string<ORTCHAR_T>& path_to_checkpoint,
                                            const bool include_optimizer_state) {
  ThrowOnError(GetTrainingApi().SaveCheckpoint(checkpoint_states, path_to_checkpoint.c_str(),
                                               include_optimizer_state));
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

inline void CheckpointState::AddProperty(const std::string& property_name, const Property& property_value) {
  if (std::holds_alternative<int64_t>(property_value)) {
    int64_t value = std::get<int64_t>(property_value);
    void* value_p = &value;
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtIntProperty, value_p));
  } else if (std::holds_alternative<float>(property_value)) {
    float value = std::get<float>(property_value);
    void* value_p = &value;
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtFloatProperty, value_p));
  } else if (std::holds_alternative<std::string>(property_value)) {
    std::string value = std::get<std::string>(property_value);
    auto buffer = std::make_unique<char[]>(value.length() + 1);
    memcpy(buffer.get(), value.c_str(), value.length());
    // AddProperty takes a char* and calls PropertyBag::AddProperty which takes a std::string. The data will be
    // copied at that point so buffer can free the local allocation once the call is made.
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtStringProperty,
                                              buffer.get()));
  } else {
    ThrowStatus(Status("Unknown property type received.", OrtErrorCode::ORT_INVALID_ARGUMENT));
  }
}

inline Property CheckpointState::GetProperty(const std::string& property_name) {
  void* property_value = nullptr;
  OrtPropertyType property_type;

  AllocatorWithDefaultOptions allocator;
  ThrowOnError(GetTrainingApi().GetProperty(p_, property_name.c_str(), allocator, &property_type, &property_value));

  Property property;

  switch (property_type) {
    case OrtPropertyType::OrtIntProperty: {
      auto value_p = reinterpret_cast<int64_t*>(property_value);
      property = *value_p;
      allocator.Free(property_value);
      break;
    }
    case OrtPropertyType::OrtFloatProperty: {
      auto value_p = reinterpret_cast<float*>(property_value);
      property = *value_p;
      allocator.Free(property_value);
      break;
    }
    case OrtPropertyType::OrtStringProperty: {
      auto value_p = reinterpret_cast<char*>(property_value);
      property = std::string(value_p);
      allocator.Free(property_value);
      break;
    }
    default: {
      ThrowStatus(Status("Unknown property type received.", OrtErrorCode::ORT_INVALID_ARGUMENT));
      break;
    }
  }

  return property;
}

inline void CheckpointState::UpdateParameter(const std::string& parameter_name, const Value& parameter) {
  ThrowOnError(GetTrainingApi().UpdateParameter(p_, parameter_name.c_str(), parameter));
}

inline Value CheckpointState::GetParameter(const std::string& parameter_name) {
  AllocatorWithDefaultOptions allocator;
  OrtValue* parameter;
  ThrowOnError(GetTrainingApi().GetParameter(p_, parameter_name.c_str(), allocator, &parameter));

  return Value{parameter};
}

}  // namespace Ort
