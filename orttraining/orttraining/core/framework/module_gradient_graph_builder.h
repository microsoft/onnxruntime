// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

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
  std::vector<std::string> user_input_names{};
  std::unordered_map<std::string, std::string> user_input_grad_names{};
  std::vector<std::string> initializer_names_to_train{};
  std::vector<std::string> initializer_grad_names_to_train{};
  std::vector<std::string> user_output_names{};
  std::vector<std::string> backward_user_input_names{};
  std::vector<std::string> backward_intializer_names_as_input{};
  std::vector<std::string> intermediate_tensor_names{};
  std::vector<std::string> user_output_grad_names{};
  std::vector<std::string> backward_output_grad_names{};
};

class ModuleGradientGraphBuilder {
 public:
  Status Initialize(std::istream& model_istream, const ModuleGradientGraphBuilderConfiguration& config);
  Status BuildAndSplit(const std::vector<std::vector<int64_t>>& input_shapes);
  Status Build();

  std::string GetForwardModel() const;
  std::string GetBackwardModel() const;
  SplitGraphsInfo GetSplitGraphsInfo() const { return split_graphs_info_; }

 private:
  Status Split();

  // Build gradient graph.
  Status BuildGradientGraph();

  // Add Yield Op.
  void AddYieldOp();

  // Reorder gradient graph inputs/outputs.
  void ReorderOutputs();

  std::shared_ptr<Model> model_;
  std::shared_ptr<Model> gradient_model_;
  std::shared_ptr<Model> forward_model_;
  std::shared_ptr<Model> backward_model_;
  SplitGraphsInfo split_graphs_info_;

  ModuleGradientGraphBuilderConfiguration config_;
  const logging::Logger* logger_ = &logging::LoggingManager::DefaultLogger(); // use default logger for now.
};

}  // namespace training
}  // namespace onnxruntime
