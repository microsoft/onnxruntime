// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/training/optimizer_builder.h"
#include "core/training/optimizer_config.h"

namespace onnxruntime {
namespace training {

constexpr const char* kGradientAccumulationOutputKey = "GRADIENT_ACCUMULATION_OUTPUT";
constexpr const char* kGradientAllIsFiniteOutputKey = "GRADIENT_ALL_IS_FINITE";
constexpr const char* kGlobalGradientNormOutputKey = "Global_GRADIENT_NORM";

// given a base name, return a name suitable for a graph NodeArg
using NodeArgNameGeneratorFn = std::function<std::string(const std::string&)>;

Status GetArgDefsFromGraph(
    const Graph& graph, const std::vector<std::string>& node_arg_names,
    std::vector<ArgDef>& argdefs);

ArgDef BuildGradientAccumulationNode(const NodeArgNameGeneratorFn& nodearg_name_generator,
                                     const ArgDef& gradient,
                                     ArgDef& gradient_accumulation_buffer,
                                     GraphAugmenter::GraphDefs& graph_defs,
                                     bool add_accumulate_buffer_as_initializers = true);

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
   * @param[out] optimizer_state_initializer_names The names of the
   *             initializers representing the optimizer state.
   * @param[out] optimizer_graph_outputs The outputs introduced in optimizer graph
   * @return The status of the graph modification.
   */
  Status Build(
      Graph& graph,
      std::unordered_set<std::string>& optimizer_state_initializer_names,
      std::unordered_map<std::string, std::string>& optimizer_graph_outputs);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerGraphBuilder);

  const OptimizerBuilderRegistry& opt_builder_registry_;
  const OptimizerGraphConfig opt_graph_config_;
  std::vector<std::string> weight_names_;
  std::vector<OptimizerNodeConfig> opt_configs_;
};

}  // namespace training
}  // namespace onnxruntime
