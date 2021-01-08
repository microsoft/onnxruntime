// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/status.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace training {

/**
 * The training configuration options.
 */
struct ModuleGradientGraphBuilderConfiguration {
  // The names of the weights to train.
  std::vector<std::string> initializer_names_to_train{};
  // The names of inputs that require gradient.
  std::vector<std::string> input_names_require_grad{};

  // Gradient graph configuration.
  bool use_invertible_layernorm_grad = false;

  // TODO: add GraphTransformerConfiguration
};

/**
 * The information of split graphs for frontend.
 */
struct SplitGraphsInfo {
  // The user inputs.
  std::vector<std::string> user_input_names{};
  // Map from user input names to corresponding user input grad names for those user inputs that require grad.
  std::unordered_map<std::string, std::string> user_input_grad_names{};
  // Trainable initializers.
  std::vector<std::string> initializer_names_to_train{};
  // Trainable initializer grad names, ordered according to initializer_names_to_train.
  std::vector<std::string> initializer_grad_names_to_train{};
  // The user outputs.
  std::vector<std::string> user_output_names{};
  // The user inputs needed by backward graph.
  std::vector<std::string> backward_user_input_names{};
  // The trainable initializers needed by backward graph.
  std::vector<std::string> backward_intializer_names_as_input{};
  // The intermediate tensors from forward graph needed by backward graph.
  std::vector<std::string> intermediate_tensor_names{};
  // The user output grad names, ordered according to the user_output_names.
  std::vector<std::string> user_output_grad_names{};
  // The user output grad names that are actual required by the backward graph.
  std::vector<std::string> backward_output_grad_names{};
};

class ModuleGradientGraphBuilder {
 public:
  /**
   * Initialize the builder. It saves the initial model and the configuration.
   * It also removes the trainable initializers from initial model and move them to graph inputs.
   * @param model_istream The initial model as input stream.
   * @param config The configuration to control the builder.
   * @return The status of the initialization.
   */
  Status Initialize(std::istream& model_istream, const ModuleGradientGraphBuilderConfiguration& config);

  /**
   * Build the gradient graph and split it to forward and backward graphs.
   * @param input_shapes_ptr The pointer to vector of concrete shapes of the user inputs.
   * @return The status of the gradient graph building and forward/backward graphs splitting.
   */
  Status Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr = nullptr);

  /**
   * Get forward model.
   * @return The forward model serialized to string.
   */
  std::string GetForwardModel() const;

  /**
   * Get backward model.
   * @return The backward model serialized to string.
   */
  std::string GetBackwardModel() const;

  /**
   * Get the split graphs information.
   * @return The split graphs information.
   */
  SplitGraphsInfo GetSplitGraphsInfo() const { return split_graphs_info_; }

 private:
  // Set concrete shapes for graph inputs.
  void SetConcreteInputShapes(const std::vector<std::vector<int64_t>> input_shapes);

  // Build gradient graph as backward_model_.
  Status BuildGradientGraph();

  // Set the forward graph outputs and backward graph inputs.
  void SetForwardOutputsAndBackwardInputs();

  // Set the backward graph outputs.
  void SetBackwardOutputs();

  std::shared_ptr<Model> model_;
  std::shared_ptr<Model> forward_model_;
  std::shared_ptr<Model> backward_model_;
  SplitGraphsInfo split_graphs_info_;

  ModuleGradientGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger();  // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
