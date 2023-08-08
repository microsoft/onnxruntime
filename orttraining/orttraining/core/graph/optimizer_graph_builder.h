// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/graph/optimizer_graph_output_key.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace training {

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
   * @param updated_weight_names_map Mapping from original weight name to
   *        updated weight name.
   * @param weight_partition_info Mapping of (parititioned) weight name to partition info.
   */
  OptimizerGraphBuilder(
      const OptimizerBuilderRegistry& opt_builder_registry,
      const OptimizerGraphConfig& opt_graph_config,
      const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
      std::unordered_map<std::string, std::string>& updated_weight_names_map,
      std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info);

  virtual ~OptimizerGraphBuilder() {}

  /**
   * Builds the optimizer components on top of the graph.
   * @param graph The graph to build upon.
   * @param[out] weight_to_opt_mapping weight -> optimizer state mapping.
   * @param[out] optimizer_graph_outputs The outputs introduced in optimizer graph
   * @return The status of the graph modification.
   */
  Status Build(
      Graph& graph,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs);

 protected:
  virtual Status BuildInternal(
      bool should_add_gradient_norm,
      bool should_add_gradient_finite_check,
      Graph& graph,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& weight_argdefs,
      std::vector<ArgDef>& gradient_argdefs,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs);

  Status AddGradientPassThroughNode(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      std::vector<ArgDef>& gradient_argdefs,  // update argdefs in place
      GraphAugmenter::GraphDefs& graph_defs);

  Status AddGradientScalingNodes(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      const float scale,
      std::vector<ArgDef>& gradient_argdefs,
      ArgDef& fused_gradient_argdef,
      GraphAugmenter::GraphDefs& graph_defs,
      ONNX_NAMESPACE::TensorProto_DataType allreduce_element_type,
      const bool fuse_scaling_outputs);

  Status AddGradientScalingNodes(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      const float scale,
      std::vector<ArgDef>& gradient_argdefs,        // update argdefs in place
      std::vector<ArgDef>& output_gradient_argdef,  // update argdef in place
      GraphAugmenter::GraphDefs& graph_defs,
      ONNX_NAMESPACE::TensorProto_DataType target_type);

  Status AddGradientNorm(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      const std::vector<ArgDef>& grad_argdefs,
      GraphAugmenter::GraphDefs& graph_defs,
      ArgDef& grad_norm_argdef);

  Status AddFiniteGradientCheck(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      const std::vector<ArgDef>& grad_norm_argdefs,
      GraphAugmenter::GraphDefs& graph_defs,
      ArgDef& grad_norm_finite_argdef,
      const std::string& node_name = "all_gradients_finite");

  Status AddDirectWeightUpdate(
      const OptimizerBuilderRegistry& opt_builder_registry,
      std::vector<ArgDef>& weight_argdefs,
      std::vector<ArgDef>& gradient_argdefs,
      const ArgDef* global_gradient_norm_argdef,
      const ArgDef* global_gradient_norm_finite_argdef,
      const std::vector<OptimizerNodeConfig>& opt_configs,
      GraphAugmenter::GraphDefs& graph_defs,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping);

  // This function can be overriden by child classes to have different logic
  // for building optimizers.
  virtual Status BuildOptimizerNode(
      const std::unique_ptr<OptimizerBuilder>& opt_builder,
      const std::vector<ArgDef>& weight_argdefs,
      const std::vector<ArgDef>& gradient_argdefs,
      const ArgDef* global_gradient_norm_argdef,
      const ArgDef* global_gradient_norm_finite_argdef,
      const std::vector<OptimizerNodeConfig>& opt_configs,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<TensorProto>& new_initializers,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      std::vector<ArgDef>& output_weight_argdefs,
      std::vector<ArgDef>& output_gradient_argdefs);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerGraphBuilder);

  const OptimizerBuilderRegistry& opt_builder_registry_;
  const OptimizerGraphConfig opt_graph_config_;
  std::vector<std::string> weight_names_;
  std::vector<std::string> gradient_names_;
  std::vector<OptimizerNodeConfig> opt_configs_;
  std::unordered_map<std::string, std::string>& updated_weight_names_map_;
  std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info_;
};

}  // namespace training
}  // namespace onnxruntime
