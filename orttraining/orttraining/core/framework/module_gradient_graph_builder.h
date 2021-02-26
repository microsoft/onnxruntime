// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

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
 * The information of training graphs for frontend.
 */
struct TrainingGraphInfo {
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
  // The user output grad names that are actually required by the backward graph 
  // mapped to the index of the correspoinding output of inference graph.
  std::unordered_map<std::string, size_t> backward_output_grad_names_map{};
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
   * Get gradient model.
   * @return The gradient model serialized to string.
   */
  std::string GetGradientModel() const;

  /**
   * Get the split graphs information.
   * @return The split graphs information.
   */
  TrainingGraphInfo GetTrainingGraphInfo() const { return training_graph_info_; }

 private:
  // Set concrete shapes for graph inputs.
  void SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes);

  // Build gradient graph.
  Status BuildGradientGraph();

  // Add Yield Op.
  void AddYieldOp();

  // Reorder gradient graph outputs.
  void ReorderOutputs();

  std::shared_ptr<onnxruntime::Model> model_;
  std::shared_ptr<onnxruntime::Model> gradient_model_;
  TrainingGraphInfo training_graph_info_;

  ModuleGradientGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger();  // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
