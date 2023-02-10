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
  bool use_memory_efficient_gradient = false;
  bool build_gradient_graph = true;
  bool enable_caching = false;

  // Graph transformer configuration
  TrainingGraphTransformerConfiguration graph_transformer_config{};

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
  // Indices of module output that are needed for backward computation
  std::vector<size_t> module_output_indices_requires_save_for_backward{};
  // Names of module outputs' gradient
  std::vector<std::string> module_output_gradient_name{};
  // Names of the frontier tensor corresponding to param
  std::unordered_map<std::string, std::string> frontier_node_arg_map{};
  // Names of the frontier NodeArgs in the order in which they will
  // be retrieved in the forward pass
  std::vector<std::string> cached_node_arg_names{};
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
   * Get gradient model.
   * @return The optimized gradient model serialized to string.
   */
  std::string GetGradientModel() const;

  /**
   * Get inference/forward model. If it's training mode, the forward model is optimized.
   * @return The inference/gradient model serialized to string.
   */
  std::string GetForwardModel() const;

  /**
   * Get the graphs information.
   * @return The graphs information.
   */
  GraphInfo GetGraphInfo() const { return graph_info_; }

 private:
  // Set concrete shapes for graph inputs.
  Status SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes);

  // Apply graph transformers
  Status OptimizeForwardGraph(std::unordered_set<std::string>& x_node_arg_names);

  // Build gradient graph.
  Status BuildGradientGraph(const std::unordered_set<std::string>& x_node_arg_names);

  // Get the "frontier" tensors- the the output of series of operations
  // that only depend on the param values, eg Casting a param
  void GetFrontierTensors();

  // Handle user outputs and output grads.
  void HandleOutputsAndGrads();

  // Reorder gradient graph outputs.
  void ReorderOutputs();

  // Find the module output that are needed for backward computation
  void FindModuleOutputNeededForBackward();

  // Update require grad info for PythonOp.
  void UpdatePythonOpInputsRequireGradInfo(
      const std::unordered_map<std::string, std::vector<int64_t>>& python_op_input_require_grad_info);

  std::shared_ptr<onnxruntime::Model> original_model_;
  // For training case, the forward_model_ is the model after applying training graph transformers.
  // For inference case, the forward_model_ is same as original_model_ and have concrete shapes set if required.
  std::shared_ptr<onnxruntime::Model> forward_model_;
  std::shared_ptr<onnxruntime::Model> gradient_model_;
  GraphInfo graph_info_;

  OrtModuleGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger();  // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
