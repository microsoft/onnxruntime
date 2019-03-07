// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/inference_session_impl.h"
#include "core/framework/ml_value.h"
#include "loss_function_registry.h"
#include "core/training/loss_func/loss_func_common.h"

namespace onnxruntime {  // forward declarations
struct SessionOptions;

namespace logging {
class LoggingManager;
}

namespace training {

// Although being used as pimpl, TrainingSessionImpl has to be outside of TrainingSession in order to inherit from InferenceSession::Impl.
// Because:
// 1. it needs to be declared friend of InferenceSession, otherwise it cannot access InferenceSession::Impl;
// 2. InferenceSession doesn't want to see the definition of TrainingSession.
class TrainingSessionImpl;

class TrainingSession {
 public:
  explicit TrainingSession(const SessionOptions& session_options,
                           logging::LoggingManager* logging_manager = nullptr);

  ~TrainingSession();

  common::Status Load(const std::string& model_uri);

  /** Register a custom loss function before calling AddLossFuncion, when user wants to customize the loss function.
  @param loss_func_name The op name to be used as a loss function.
  @remarks When using a custom/standard op as loss function, 2 ops must have been registered:
             1. an op for loss function, schema:
                 Inputs:
                     OUT
                     LABEL
                 Outputs:
                     LOSS
             2. an op to calculate gradients, schema:
                 Inputs:
                     GRADIENT_OF_OUTPUT
                     OUT
                     LABEL
                 Outputs:
                     GRADIENT_OF_OUT
                     GRADIENT_OF_LABEL
           And also in gradient_builder.cc, the gradient builder must have been registered.
  */
  common::Status RegisterCustomLossFunction(const std::string& loss_func_name);

  /** Add a system provided or a customized loss function to the model.
  After the call, the model have one more input named as label_name and one more output named as loss_func_output_name.
  @param loss_func_info The loss function info.
  @returns Status indicating success or providing an error message.
  @remarks The loss_func_name could be either system provided or a custom one.           
  */
  common::Status AddLossFuncion(const LossFunctionInfo& loss_func_info);

  common::Status BuildGradientGraph(const std::vector<std::string>& weights_to_train, const std::string& loss_function_output_name);

  common::Status Initialize();

  // Compute gradients.
  common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches);

  // Save a model, 3 options:
  // 1. save with updated weights
  // 2. save with updated weights and loss function
  // 3. save with updated weights, loss function and gradients
  enum class SaveOption {
    WITH_UPDATED_WEIGHTS,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS
  };

  common::Status Save(const std::string& model_uri, SaveOption opt);

  // TODO: remove or refine below temp interfaces.
  NameMLValMap GetWeights() const;
  common::Status UpdateWeights(const NameMLValMap& new_weights);
  std::unordered_set<std::string> GetModelInputNames() const;
  std::unordered_set<std::string> GetModelOutputNames() const;
  std::unordered_set<std::string> GetModelInitializers() const;

 private:
  std::unique_ptr<TrainingSessionImpl> impl_;
};
}  // namespace training
}  // namespace onnxruntime
