// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace training {
namespace api {

struct Parameter {
 public:
  // Create parameter
  Parameter(std::string name, const OrtValue& data)
      : name_(name), data_(data) {
  }

  // Return the mutable data.
  OrtValue& Data() { return data_; }
  std::string Name() const { return name_; }

  // Return if trainable. The trainable property of a param
  // cannot change over the lifetime of the on-device training
  // session since the gradient graph is prebuilt for this setting.
  bool RequiresGrad() const { return requires_grad_; }

  // Return the mutable gradient for trainable parameter.
  OrtValue& Gradient() { return gradient_; }
  std::string GradientName() const { return gradient_name_; }

  // Reset and release the gradient buffer of this Parameter.
  Status ResetGrad() {
    return Status::OK();
  }

  Status SetRequiresGrad(bool requires_grad) {
    requires_grad_ = requires_grad;
    return Status::OK();
  }

  // need to set grad but not public api
 private:
  std::string name_;
  OrtValue data_;

  OrtValue gradient_;
  std::string gradient_name_;

  // Whether the param is trainable. The optimizer state is
  // only created for a trainable param
  bool requires_grad_{true};
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
  Module(const std::string& /*train_model_path_or_bytes*/,
         const std::unordered_map<std::string, std::shared_ptr<Parameter>>& /*parameters*/,
         const std::optional<std::string>& /*eval_model_path_or_bytes*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  // Return the trainable/nontrainable parameters
  std::vector<std::shared_ptr<Parameter>> parameters() const {
    return parameters_;
  }
  std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters() const {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return {};
  }

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs.
  // Gradients will be accumulated within the Parameter object
  Status TrainStep(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Eval Step – does forward computation. This will use a separate inference session
  // and take in a separate inference graph, while sharing the parameters
  Status EvalStep(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Return the states of the module as a map.
  Status GetStateDict(ModuleCheckpointState& module_checkpoint_states);

 private:
  std::unique_ptr<onnxruntime::InferenceSession> train_sess_;
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_;
  std::vector<std::shared_ptr<Parameter>> parameters_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
