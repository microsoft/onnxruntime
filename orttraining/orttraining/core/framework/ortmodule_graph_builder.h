// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

#include "core/common/status.h"
#include "core/graph/model.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

/**
 * The training configuration options.
 */
struct OrtModuleGraphBuilderConfiguration {
  // The names of the weights.
  std::vector<std::string> initializer_names{};
  // The names of the weights to train.
  std::vector<std::string> initializer_names_to_train{};
  // The names of inputs that require gradient.
  std::vector<std::string> input_names_require_grad{};

  // Graph configuration.
  bool use_invertible_layernorm_grad = false;
  bool build_gradient_graph = true;

  // Graph transformer configuration
  TrainingSession::TrainingConfiguration::GraphTransformerConfiguration graph_transformer_config{};

  // Log severity
  logging::Severity loglevel{logging::Severity::kWARNING};
};

/**
 * The information of graphs for frontend.
 */
struct GraphInfo {
  // The user inputs.
  std::vector<std::string> user_input_names{};
  // Map from user input names to corresponding user input grad names for those user inputs that require grad.
  std::unordered_map<std::string, std::string> user_input_grad_names{};
  // All initializers (trainable as well as non trainable).
  std::vector<std::string> initializer_names{};
  // Trainable initializers.
  std::vector<std::string> initializer_names_to_train{};
  // Trainable initializer grad names, ordered according to initializer_names_to_train.
  std::vector<std::string> initializer_grad_names_to_train{};
  // The user outputs.
  std::vector<std::string> user_output_names{};
  // Indices of output grads that are non-differentiable.
  std::vector<size_t> output_grad_indices_non_differentiable{};
  // Indices of output grads that need to be materialized to full size all-0 tensor.
  // Otherwise, we can use scalar-0 tensor.
  std::vector<size_t> output_grad_indices_require_full_shape{};
};

class OrtModuleGraphBuilder {
 public:
  /**
   * Initialize the builder. It saves the initial model and the configuration.
   * It also removes the trainable initializers from initial model and move them to graph inputs.
   * @param model_istream The initial model as input stream.
   * @param config The configuration to control the builder.
   * @return The status of the initialization.
   */
  Status Initialize(std::istream& model_istream, const OrtModuleGraphBuilderConfiguration& config);

  /**
   * Optimize the inference graph and build the gradient graph.
   * @param input_shapes_ptr The pointer to vector of concrete shapes of the user inputs.
   * @return The status of the optimizing and building the gradient graph.
   */
  Status Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr = nullptr);

  /**
   * Get inference/gradient model.
   * @return The optimized inference/gradient model serialized to string.
   */
  std::string GetModel() const;

  /**
   * Get inference optimized model.
   * @return The gradient model serialized to string.
   */
  std::string GetInferenceOptimizedModel() const;

  /**
   * Get the graphs information.
   * @return The graphs information.
   */
  GraphInfo GetGraphInfo() const { return graph_info_; }

 private:
  // Set concrete shapes for graph inputs.
  void SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes);

  // Apply graph transformers
  Status OptimizeInferenceGraph(std::unordered_set<std::string>& x_node_arg_names);

  // Build gradient graph.
  Status BuildGradientGraph(const std::unordered_set<std::string>& x_node_arg_names);

  // Handle user outputs and output grads.
  void HandleOutputsAndGrads();

  // Reorder gradient graph outputs.
  void ReorderOutputs();

  std::shared_ptr<onnxruntime::Model> model_;
  std::shared_ptr<onnxruntime::Model> inference_optimized_model_;
  std::shared_ptr<onnxruntime::Model> gradient_model_;
  GraphInfo graph_info_;

  OrtModuleGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger();  // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
