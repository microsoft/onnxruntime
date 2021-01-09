// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"

namespace onnxruntime {
namespace training {

class AdasumOptimizerGraphBuilder : public AllreduceOptimizerGraphBuilder {
 public:
  AdasumOptimizerGraphBuilder(
      const OptimizerBuilderRegistry& opt_builder_registry,
      const OptimizerGraphConfig& opt_graph_config,
      const std::unordered_map<std::string, OptimizerNodeConfig>& weight_names_to_opt_configs,
      std::unordered_map<std::string, std::string>& updated_weight_names_map,
      std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info);

 protected:
  virtual Status BuildInternal(
      bool should_add_gradient_norm,
      bool should_add_gradient_finite_check,
      Graph& graph,
      GraphAugmenter::GraphDefs& graph_defs,
      std::vector<ArgDef>& weight_argdefs,
      std::vector<ArgDef>& gradient_argdefs,
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping,
      OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) override;

  ArgDef BuildWeightUpdateNode(
      const NodeArgNameGeneratorFn& nodearg_name_generator,
      const ArgDef& gradient,
      ArgDef& weight,
      const ArgDef& gradient_finite_argdef,
      GraphAugmenter::GraphDefs& graph_defs);

  Status AddWeightUpdateNodes(const NodeArgNameGeneratorFn& nodearg_name_generator,
                              std::vector<ArgDef>& gradient_argdefs,
                              std::vector<ArgDef>& weight_argdefs,
                              const ArgDef& adasum_gradient_finite_argdef,
                              GraphAugmenter::GraphDefs& graph_defs,
                              std::vector<ArgDef>& output_weight_argdefs);

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
      std::vector<ArgDef>& output_gradient_argdefs) override;
};

}  // namespace training
}  // namespace onnxruntime
