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
std::unordered_set<std::string> weight_names_to_train{};
// The names of inputs that require gradient.
std::unordered_set<std::string> input_names_require_grad{};
// The names of module outputs.
std::unordered_set<std::string> output_names{};

// Gradient graph configuration.
bool use_invertible_layernorm_grad = false;
bool set_gradients_as_graph_outputs = false;

// TODO: add GraphTransformerConfiguration
// TODO: add mixed precision config
// TODO: do we need to support graph with loss?
};

class ModuleGradientGraphBuilder {
 public:
  Status BuildAndSplit(std::istream& model_istream,
                       const ModuleGradientGraphBuilderConfiguration& config,
                       std::vector<std::string>& models_as_string);
 private:
  Status Split(const ModuleGradientGraphBuilderConfiguration& config,
               const std::vector<std::string>& graph_output_names);

  std::shared_ptr<onnxruntime::Model> model_;
  std::shared_ptr<onnxruntime::Model> forward_model_;
  std::shared_ptr<onnxruntime::Model> backward_model_;
  const logging::Logger* logger_;
};

}  // namespace training
}  // namespace onnxruntime
