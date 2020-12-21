// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/optimizer_graph_builder.h"

namespace onnxruntime {
namespace training {

class AllreduceOptimizerGraphBuilder : public OptimizerGraphBuilder {
 public:
  AllreduceOptimizerGraphBuilder(
      const OptimizerBuilderRegistry& opt_builder_registry,
      const OptimizerGraphConfig& opt_graph_config,
      const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
      std::unordered_map<std::string, std::string>& updated_weight_names_map);

 protected:
  virtual Status BuildInternal(
      bool should_add_gradient_norm,
      bool should_add_gradient_finite_check,
      Graph& graph,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& weight_argdefs,
      std::vector<ArgDef>& gradient_argdefs,
      std::unordered_set<std::string>& optimizer_state_initializer_names,
      OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) override;
};

}  // namespace training
}  // namespace onnxruntime
