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
  std::string Build(std::istream& model_istream, const ModuleGradientGraphBuilderConfiguration& config);
};

}  // namespace training
}  // namespace onnxruntime
