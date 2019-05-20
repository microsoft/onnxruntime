// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/session/inference_session.h"
#include "core/graph/training/loss_func/loss_func_common.h"
#include "core/graph/training/loss_function_registry.h"
#include "core/graph/training/in_graph_training_optimizer.h"

namespace onnxruntime {
namespace training {

class TrainingSession : public InferenceSession {
 public:
  explicit TrainingSession(const SessionOptions& session_options,
                           logging::LoggingManager* logging_manager = nullptr)
      : InferenceSession(session_options, logging_manager) {}

  /** Add a system provided or an op as loss function to the model.
  After the call, the model have one more input named as label_name and one more output named as loss_func_output_name.
  @param loss_func_info The loss function info.
  @returns Status indicating success or providing an error message.
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
  common::Status AddLossFuncion(const LossFunctionInfo& loss_func_info);

  /** Perform auto-diff to add backward graph into the model.
  @param weights_to_train a list of weights to be training.
  @param loss_function_output_name the name of the loss function's output.
  @param opt_info optional, specify the optimizers used by each weight in weights_to_train, 1-1 mapping to weights_to_train.
  @remarks if optimizer_and_params is not empty, in the gradient graph, every gradient will be fed into a new optimizer
           node:
           1. New inputs: the parameters of the optimizer are the new graph inputs
                          Optimizer with same names share the same parameters.
           2. New outputs: the output of optimizer will become the new graph outputs.
           3. Every weight in weights_to_train must have the optimizer info specified.
           4. Differnt weights can have different optimizers and parameters.
  */

  common::Status BuildGradientGraph(const std::vector<std::string>& weights_to_train,
                                    const std::string& loss_function_output_name,
                                    const std::vector<in_graph_optimizer::OptimizerInfo>& opt_info = {});

  /** Save a model, 3 options:
  1. save with updated weights
  2. save with updated weights and loss function
  3. save with updated weights, loss function and gradients
  */
  enum class SaveOption {
    NO_RELOAD,
    WITH_UPDATED_WEIGHTS,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS
  };

  /** Save the new model.
  @param model_uri the path for the new model.
  @param opt see SaveOption.
  */
  common::Status Save(const std::string& model_uri, SaveOption opt);

  // TODO: remove or refine below temp interfaces.
  NameMLValMap GetWeights() const;
  common::Status UpdateWeights(const NameMLValMap& new_weights);
  std::unordered_set<std::string> GetModelInputNames() const;
  std::unordered_set<std::string> GetModelOutputNames() const;
  std::unordered_set<std::string> GetModelInitializers() const;

 private:
  std::vector<std::string> weights_to_train_;
  LossFunctionInfo loss_func_info_;
  std::vector<in_graph_optimizer::OptimizerInfo> opt_info_;
};
}  // namespace training
}  // namespace onnxruntime
