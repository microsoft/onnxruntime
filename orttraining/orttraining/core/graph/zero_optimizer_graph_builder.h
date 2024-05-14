// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/optimizer_graph_builder.h"

namespace onnxruntime {
namespace training {

class ZeROOptimizerGraphBuilder : public OptimizerGraphBuilder {
 public:
  ZeROOptimizerGraphBuilder(
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
      std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& optimizer_state_initializer_names,
      OptimizerOutputKeyMap<std::string>& optimizer_graph_outputs) override;
};

/**
 * Partitions the initial states according to the offset and
 * size provided when the optimizer state for a weight is to be
 * partitioned in Zero stage 1.
 *
 * @param partition_offset The offset for start of partition
 * @param partition_size The size(number of elements) of the partition
 * @param[out] initial_states The optimizer initial states modified in-place.
 */
void PartitionOptimizerState(
    const int64_t partition_offset,
    const int64_t partition_size,
    NameMLValMap& initial_states);

}  // namespace training
}  // namespace onnxruntime
