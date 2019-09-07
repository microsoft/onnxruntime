// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/training/training_optimizer.h"
#include "core/training/optimizer_config.h"

namespace onnxruntime {
namespace training {

/**
 * Builds the optimizer components on top of an existing training graph.
 * The optimizers used are determined by the weight_names_to_opt_configs parameter
 * of this class' constructor.
 * Note that each optimized weight tensor has its own optimizer.
 */
class OptimizerGraphBuilder {
 public:
  /**
   * Constructor.
   * @param opt_builder_registry The OptimizerBuilderRegistry instance.
   * @param opt_graph_config The overall optimizer configuration values.
   * @param weight_names_to_opt_configs Mapping from weight name to per optimizer
   *        configuration values.
   */
  OptimizerGraphBuilder(
      const OptimizerBuilderRegistry& opt_builder_registry,
      const OptimizerGraphConfig& opt_graph_config,
      const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs);

  /**
   * Builds the optimizer components on top of the graph.
   * @param graph The graph to build upon.
   * @return The status of the graph modification.
   */
  Status Build(Graph& graph);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerGraphBuilder);

  const OptimizerBuilderRegistry& opt_builder_registry_;
  const OptimizerGraphConfig opt_graph_config_;
  std::vector<std::string> weight_names_;
  std::vector<OptimizerNodeConfig> opt_configs_;
};

}  // namespace training
}  // namespace onnxruntime
