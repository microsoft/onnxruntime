// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/inference_session.h"

namespace onnxruntime {
namespace training {
namespace api {

struct Parameter {
 public:
  Parameter(const std::string& name, const OrtValue& data, const bool requires_grad)
      : name_(name), data_(data), requires_grad_(requires_grad) {
    ORT_ENFORCE(data_.IsAllocated());
    ORT_ENFORCE(!name_.empty(), "Parameter must have a non-empty name.");
  }

  // Return the mutable data.
  OrtValue& Data() { return data_; }
  const std::string& Name() const { return name_; }

  // Return parameter trainable or not. The trainable property of a param
  // cannot change over the lifetime of the on-device training
  // session since the gradient graph is prebuilt for this setting.
  bool RequiresGrad() const { return requires_grad_; }

  // Return the mutable gradient for trainable parameter.
  OrtValue& Gradient() { return gradient_; }
  const std::string& GradientName() const { return gradient_name_; }

  // Reset and release the gradient buffer of this Parameter greedily.
  Status ResetGrad();

 protected:
  Status SetGrad(const std::string& gradient_name, const OrtValue& param_grad);

 private:
  std::string name_;
  OrtValue data_;

  OrtValue gradient_;
  std::string gradient_name_;

  bool requires_grad_{true};
  friend struct Module;
};

struct ModuleCheckpointState {
 public:
  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters;
  const DataTransferManager* train_session_data_transfer_mgr;
};

struct Module {
 public:
  // Initialize a module from an ORT inference session with loaded
  // training ONNX model and load parameters
  Module(const std::string& train_model_path_or_bytes,
         const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
         const onnxruntime::SessionOptions& session_options,
         const Environment& env,
         const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
         const std::optional<std::string>& eval_model_path_or_bytes = std::nullopt);

  // Return the trainable/nontrainable parameters
  std::vector<std::shared_ptr<Parameter>> Parameters() const;

  std::unordered_map<std::string, std::shared_ptr<Parameter>> NamedParameters() const {
    return named_parameters_;
  }

  // Reset and release the gradient buffer of all trainable params lazily.
  Status ResetGrad();

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs.
  // Gradients will be accumulated within the Parameter object
  Status TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs);

  // Eval Step – does forward computation. This will use a separate inference session
  // and take in a separate inference graph, while sharing the parameters
  Status EvalStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs);

  // Return the states of the module as a map.
  Status GetStateDict(ModuleCheckpointState& module_checkpoint_states);

  // Returns the output count for training graph
  size_t GetTrainModeOutputCount() const noexcept;

  // Returns the output count for eval graph
  size_t GetEvalModeOutputCount() const noexcept;

 private:
  std::unique_ptr<onnxruntime::InferenceSession> train_sess_{nullptr};
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_{nullptr};
  std::vector<std::string> train_input_names_;
  std::vector<std::string> train_output_names_;
  std::vector<std::string> eval_input_names_;
  std::vector<std::string> eval_output_names_;
  std::vector<OrtValue> weights_;
  std::vector<OrtValue> gradients_;
  bool accumulate_gradient_ = false;
  const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
