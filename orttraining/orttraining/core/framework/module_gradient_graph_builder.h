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
  bool set_gradients_as_graph_outputs = false;

  // TODO: add GraphTransformerConfiguration
  // TODO: add mixed precision config
  // TODO: do we need to support graph with loss?
};

/**
 * The information of split graphs for frontend.
 */
struct SplitGraphsInfo {
  std::vector<std::string> user_input_names{};
  std::vector<std::string> initializer_names_to_train{};
  std::vector<std::string> user_output_names{};
  std::vector<std::string> backward_user_input_names{};
  std::vector<std::string> backward_intializer_names_as_input{};
  std::vector<std::string> intermediate_tensor_names{};
  std::vector<std::string> user_output_grad_names{};
  std::vector<std::string> backward_output_grad_names{};
};

class ModuleGradientGraphBuilder {
 public:
  Status BuildAndSplit(std::istream& model_istream,
                       const ModuleGradientGraphBuilderConfiguration& config);

  std::string GetGradientModel() const;
  std::string GetForwardModel() const;
  std::string GetBackwardModel() const;
  SplitGraphsInfo GetSplitGraphsInfo() const {
    return split_graphs_info_;
  }

 private:
  Status Split();

  std::shared_ptr<onnxruntime::Model> model_;
  std::shared_ptr<onnxruntime::Model> forward_model_;
  std::shared_ptr<onnxruntime::Model> backward_model_;
  SplitGraphsInfo split_graphs_info_;

  const logging::Logger* logger_;
};

}  // namespace training
}  // namespace onnxruntime
