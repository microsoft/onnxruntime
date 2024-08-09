// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "core/session/inference_session.h"
#include "orttraining/training_api/utils.h"

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
  Status CopyTo(const DataTransferManager* data_transfer_manager, OrtValue& data) const;
  Status CopyFrom(const DataTransferManager* data_transfer_manager, const OrtValue& data);
  const std::string& Name() const { return name_; }

  // Returns whether this parameter is trainable or not.
  // The trainable property of a param is immutable since the gradient graph is prebuilt for this setting.
  bool RequiresGrad() const { return requires_grad_; }

  // Return the mutable gradient for trainable parameter.
  OrtValue& Gradient() { return gradient_; }
  const std::string& GradientName() const { return gradient_name_; }

  // Reset and release the gradient buffer of this Parameter greedily.
  Status ResetGrad();

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
  bool is_nominal_state = false;
};

struct CheckpointState;

/**
 * @brief Module class for running training forward and backward.
 *
 * This class is responsible for running forward and backward.
 * It does NOT own the parameters but only holds a weak reference to the passed
 * 'CheckpointState' in the constructor.
 *
 * During initialization, if the Parameter (stored in `CheckpointState`)'s
 * device does not match the target device, it will re-create the tensor on the
 * target device and update the Parameter's data in place. The 'target device'
 * is extracted from node placement.
 *
 * Currently, we only support load checkpoints from the constructor;
 * no public API to load state dict after Module instance is created.
 */
struct Module {
 public:
  // Initialize a module from an ORT inference session with loaded
  // training ONNX model and load parameters
  // The model and checkpoint state can be provided as a file path or a byte array
  Module(const ModelIdentifiers& model_identifiers,
         CheckpointState* state,
         const onnxruntime::SessionOptions& session_options,
         const Environment& env,
         const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
         gsl::span<OrtCustomOpDomain* const> op_domains = gsl::span<OrtCustomOpDomain* const>());

  ~Module();

  // Return the trainable/nontrainable parameters
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will raise an exception.
  std::vector<std::shared_ptr<Parameter>> Parameters() const;

  // Return the trainable/nontrainable parameters as a map
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will raise an exception.
  std::unordered_map<std::string, std::shared_ptr<Parameter>> NamedParameters() const;

  // Reset and release the gradient buffer of all trainable params lazily.
  Status LazyResetGrad();

  // Train Step – does forward and backward computation. The outputs will be the forward’s outputs.
  // Gradients will be accumulated within the Parameter object.
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will return an error.
  Status TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs);

  // Eval Step – does forward computation. This will use a separate inference session
  // and take in a separate inference graph, while sharing the parameters
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will return an error.
  Status EvalStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs);

  // Returns the output count for training graph
  size_t GetTrainingModelOutputCount() const noexcept;

  // Returns the output count for eval graph
  size_t GetEvalModelOutputCount() const noexcept;

  // Returns the output names for train graph
  std::string GetTrainingModelOutputName(size_t index) const;

  // Returns the output names for eval graph
  std::string GetEvalModelOutputName(size_t index) const;

  // Return size of all parameters
  size_t GetParametersSize(const bool trainable_only = true) const;

  // Copy parameters onto contiguous buffer held by parameters_buffer
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will return an error.
  Status CopyParametersToBuffer(OrtValue& parameters_buffer, const bool trainable_only = true);

  // Copy parameter values from contiguous buffer held by parameters_buffer onto parameters
  // This function is responsible for completing the nominal checkpoint state. The checkpoint
  // state will no longer be nominal after the successful completion of this function.
  Status CopyBufferToParameters(OrtValue& parameters_buffer, const bool trainable_only = true);

#if !defined(ORT_MINIMAL_BUILD)
  // Load the eval model from eval_model_path_or_bytes and transform it for the purpose of
  // inferencing, and serialize to given path.
  // If the parameter state is not available; i.e. the module was created using the nominal checkpoint,
  // and the state has not been loaded yet, then this function will return an error.
  Status ExportModelForInferencing(const std::string& inference_model_path,
                                   gsl::span<const std::string> graph_output_names) const;
#endif

  // Returns the user input count for training graph
  size_t GetTrainingModelInputCount() const noexcept;

  // Returns the user input count for eval graph
  size_t GetEvalModelInputCount() const noexcept;

  // Returns the user input name for train graph at given index
  std::string GetTrainingModelInputName(size_t index) const;

  // Returns the user input name for eval graph at given index
  std::string GetEvalModelInputName(size_t index) const;

  // Returns the input definitions of the Training model
  std::pair<common::Status, const InputDefList*> GetTrainingModelInputs() const noexcept;

  // Returns the input definitions of the Eval model
  std::pair<common::Status, const InputDefList*> GetEvalModelInputs() const noexcept;

 private:
  std::unique_ptr<onnxruntime::InferenceSession> train_sess_{nullptr};
  std::unique_ptr<onnxruntime::InferenceSession> eval_sess_{nullptr};

  struct TrainInputNames {
   private:
    InlinedVector<std::string> train_input_names_;
    InlinedVector<size_t> train_input_index_offsets_;  // offset range[[0], [1]) = user input names
                                                       // offset range[[1], [2]) = weights input names
                                                       // offset range[[2], [3]) = gradient input names
   public:
    TrainInputNames() = default;
    TrainInputNames(gsl::span<const std::string> user_input_names,
                    gsl::span<const std::string> weights_input_names,
                    gsl::span<const std::string> gradient_input_names);

    gsl::span<const std::string> AllInputNames() const;
    gsl::span<const std::string> UserInputNames() const;
    gsl::span<const std::string> WeightsInputNames() const;
    gsl::span<const std::string> GradientInputNames() const;
  };

  TrainInputNames train_input_names_;
  InlinedVector<std::string> train_output_names_;
  InlinedVector<std::string> eval_input_names_;
  InlinedVector<std::string> eval_output_names_;

  InlinedVector<OrtValue> weights_;
  InlinedVector<OrtValue> gradients_;

  CheckpointState* state_;  // Non owning pointer to the state.

  bool accumulate_gradient_ = false;
  std::optional<std::string> eval_model_path_;
  size_t eval_user_input_count_{0U};
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
