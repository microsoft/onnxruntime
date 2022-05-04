// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#pragma once
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
namespace onnxruntime {
namespace training {
namespace api_test {

class Parameter {
 public:
  // create parameter
  Parameter(const std::string& name, const OrtValue& data) : name_(name), data_(data) {
    ORT_ENFORCE(data.IsAllocated());
  }

  // Return the mutable data
  OrtValue& data() { return data_; }
  std::string name() const { return name_; }

  // Return if trainable. The trainable property of a param
  // cannot change over the lifetime of the on-device training
  // session since the gradient graph is prebuilt for this setting.
  bool requires_grad() const { return requires_grad_; }

  // Return the mutable gradient for trainable parameter
  OrtValue& gradient() { return gradient_; }
  std::string gradient_name() const { return gradient_name_; }

  // Reset and release the gradient buffer of this Parameter
  Status ResetGrad();
  // need to set grad but not public api

 protected:
  Status AllocateGrad(const std::string& gradient_name, const SessionState& allocator);

 private:
  std::string name_;
  OrtValue data_;

  OrtValue gradient_;
  std::string gradient_name_;

  // Whether the param is trainable. The optimizer state is
  // only created for a trainable param
  bool requires_grad_{true};
  friend class Module;
};

class Module {
 public:
  // Initialize a module from an ORT inference session with loaded
  // training ONNX model and load parameters
  Module(const std::string& train_model_path_or_bytes,
         std::map<std::string, std::shared_ptr<Parameter>>& parameters,
         const std::optional<std::string>& eval_model_path_or_bytes = std::nullopt);

  // Return the trainable/nontrainable parameters
  std::vector<std::shared_ptr<Parameter>> parameters() const {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    // return parameters_;
  }
  std::map<std::string, std::shared_ptr<Parameter>> named_parameters() const {
    return parameters_;
  }

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs. Gradients will be accumulated within the Parameter object
  Status TrainStep(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/);

  // Eval Step – does forward computation. This will use a separate inference session
  // and take in a separate inference graph, while sharing the parameters
  Status EvalStep(const std::vector<OrtValue>& /*inputs*/, std::vector<OrtValue>& /*outputs*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Return the states of the module as a map.
  Status GetStateDict(const std::map<std::string, std::shared_ptr<Parameter>>& /*module_state_dict*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 private:
  std::unique_ptr<onnxruntime::InferenceSession> train_sess_;
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_;
  std::map<std::string, std::shared_ptr<Parameter>> parameters_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OrtValue> weights_;
  std::vector<OrtValue> gradients_;
};

// Internal state
struct ParameterOptimizerState {
  int64_t step_;
  float learning_rate_;
  // Per param optimizer state. E.g. For Adam and param_0, this would contain
  // {“Moment_1_param_0”:<value>, …},
  // It should be noted that the names should only be maintained to correlate with
  // the graph inputs for the optimizer graph
  std::map<std::string, OrtValue> states_;
};

struct OptimizerState {
  // overall state related to optimizer
  int64_t step_;
  float learning_rate_;
  std::unordered_map<std::string, ParameterOptimizerState> optimizer_states_;
};

class Optimizer {
 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input’s ValueInfoProto if the parameter doesn’t have it already.
  Optimizer(const std::string& /*optim_path_or_bytes*/,
            std::map<std::string, std::shared_ptr<Parameter>>& /*parameters*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  // Reset and release the gradient buffer of all trainable params
  Status ResetGrad() {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Optimizer Step.
  Status Step() {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Return the states of the optimizer as a map.
  Status GetStateDict(const OptimizerState& /*optimizer_state_dict*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 protected:
  int64_t GetStep() const {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return 0;
  }
  Status SetLearningRate(float /*lr*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 private:
  std::unique_ptr<onnxruntime::InferenceSession> optim_sess_;
  std::vector<std::shared_ptr<Parameter>> parameters_;
  OptimizerState optimizer_state_;
};

class LearningRateScheduler {
 public:
  LearningRateScheduler(const Optimizer& optim)
      : optim_(optim) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  virtual ~LearningRateScheduler() = default;

  // Modify the current learning rate based on current step
  virtual Status Step(/*int64_t step*/) = 0;

  const Optimizer& optim_;
};

class LinearScheduler : public LearningRateScheduler {
 public:
  explicit LinearScheduler(const Optimizer& optim, float start_factor, float end_factor, int64_t total_iters)
      : LearningRateScheduler(optim),
        start_factor_(start_factor),
        end_factor_(end_factor),
        total_iters_(total_iters) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  // Fetch the step, calculate next value and set lr in optimizer
  Status Step(/*int64_t step*/) override {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 private:
  float start_factor_;
  float end_factor_;
  int64_t total_iters_;
};

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
