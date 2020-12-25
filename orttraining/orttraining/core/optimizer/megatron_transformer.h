// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {

class MegatronTransformer : public GraphTransformer {
 public:
  MegatronTransformer(int32_t horizontal_parallel_rank, int32_t horizontal_parallel_size,
                      std::unordered_map<std::string, std::string>& updated_weight_names,
                      std::unordered_set<std::string>& weights_to_train,
                      std::unordered_map<std::string, training::TrainingSession::PartitionInfo>& weight_partition_info,
                      const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MegatronTransformer", compatible_execution_providers),
        horizontal_parallel_rank_(horizontal_parallel_rank),
        horizontal_parallel_size_(horizontal_parallel_size),
        updated_weight_names_(updated_weight_names),
        weights_to_train_(weights_to_train),
        weight_partition_info_(weight_partition_info) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

 private:
  Status TransformMLP(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                      std::vector<Node*>& nodes_to_clear_shape,
                      int32_t& counter) const;

  Status TransformSelfAttention(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                std::vector<Node*>& nodes_to_clear_shape,
                                std::unordered_set<Node*>& dropout_nodes_to_transform,
                                int32_t& counter) const;

  Status TransformBARTSelfAttention(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                    std::vector<Node*>& nodes_to_clear_shape,
                                    std::unordered_set<Node*>& dropout_nodes_to_transform, int32_t& counter) const;

  Status TransformBARTMLP(Graph& graph, bool& modified, int graph_level,
                          const logging::Logger& logger,
                          std::vector<Node*>& nodes_to_clear_shape,
                          std::unordered_set<Node*>& dropout_nodes_to_transform, int32_t& counter) const;

  Status TransformDropout(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                          std::unordered_set<Node*>& dropout_nodes_to_transform, int32_t& counter) const;

  bool PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                               ONNX_NAMESPACE::TensorProto& initializer_partition,
                               int stride = 1) const;

  bool PartitionWeightByRow(const Graph& graph, const NodeArg& input_arg, ONNX_NAMESPACE::TensorProto& initializer_partition) const;

  const int32_t horizontal_parallel_rank_;
  const int32_t horizontal_parallel_size_;
  std::unordered_map<std::string, std::string>& updated_weight_names_;
  std::unordered_set<std::string>& weights_to_train_;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo>& weight_partition_info_;
};

}  // namespace onnxruntime