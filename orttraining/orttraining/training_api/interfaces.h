// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

#pragma once

namespace onnxruntime {
namespace training {
namespace api_test {

class Parameter {
 public:
  // create parameter
  Parameter(std::string /*name*/, const OrtValue& /*data*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
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
  Status ResetGrad() {
    return Status::OK();
  }
  // need to set grad but not public api
 private:
  OrtValue data_;
  std::string name_;

  OrtValue gradient_;
  std::string gradient_name_;

  // Whether the param is trainable. The optimizer state is
  // only created for a trainable param
  bool requires_grad_{true};
};

class Module {
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

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs. Gradients will be accumulated within the Parameter object
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
  Status GetStateDict(std::unordered_map<std::string, std::shared_ptr<Parameter>>& module_states) {
    module_states = named_parameters();
    return Status::OK();
  }

 private:
  std::unique_ptr<onnxruntime::InferenceSession> train_sess_;
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_;
  std::vector<std::shared_ptr<Parameter>> parameters_;
};

// Internal state
struct ParameterOptimizerState {
  // Per param optimizer state. E.g. For Adam and param_0, this would contain
  // {“Moment_1_param_0”:<value>, …},
  // It should be noted that the names should only be maintained to correlate with
  // the graph inputs for the optimizer graph
  std::unordered_map<std::string, std::shared_ptr<OrtValue>> states_;
};

struct OptimizerState {
  // overall state related to optimizer
  int64_t step_;
  float learning_rate_;
  std::unordered_map<std::string, ParameterOptimizerState> param_optimizer_states_;
};

class Optimizer {
 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input’s ValueInfoProto if the parameter doesn’t have it already.
  Optimizer(const std::string& /*optim_path_or_bytes*/,
            const std::unordered_map<std::string, std::shared_ptr<Parameter>>& /*parameters*/) {
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
  Status GetStateDict(std::unordered_map<std::string, std::shared_ptr<OptimizerState>>& grouped_optimizer_states) {
    const std::string group_zero_name = "group_0";
    grouped_optimizer_states.insert({group_zero_name, std::make_shared<OptimizerState>(optimizer_state_)});
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

namespace utils {
/*
  module.train_sess.RegisterExecutionProvider(provider);
  module.eval_sess.RegisterExecutionProvider(provider);
  optimizer.optim_sess.RegisterExecutionProvider(provider);
*/
void SetExecutionProvider(const Module& /*module*/, const Optimizer& /*optimizer*/, IExecutionProvider* /*provider*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
}
}  // namespace utils

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
