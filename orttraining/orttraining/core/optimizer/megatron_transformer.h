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
                      training::TrainingSession::OptimizerState& initial_optimizer_states,
                      IExecutionProvider& cpu_execution_provider,  // Required to get allocator for optimizer partitioning by Col
                      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MegatronTransformer", compatible_execution_providers),
        horizontal_parallel_rank_(horizontal_parallel_rank),
        horizontal_parallel_size_(horizontal_parallel_size),
        updated_weight_names_(updated_weight_names),
        weights_to_train_(weights_to_train),
        weight_partition_info_(weight_partition_info),
        initial_optimizer_states_(initial_optimizer_states),
        cpu_execution_provider_(cpu_execution_provider) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

 private:
  // GPT2 Pattern Match
  Status TransformGPT2MLP(Graph& graph, bool& modified,
                          InlinedVector<Node*>& nodes_to_clear_shape,
                          int32_t& counter,
                          NodeIndex node_index) const;

  Status TransformGPT2Attention(Graph& graph, bool& modified,
                                InlinedVector<Node*>& nodes_to_clear_shape,
                                InlinedHashSet<Node*>& dropout_nodes_to_transform,
                                int32_t& counter,
                                NodeIndex node_index) const;

  // BART Pattern Match
  Status TransformBARTMLP(Graph& graph, bool& modified,
                          InlinedVector<Node*>& nodes_to_clear_shape,
                          InlinedHashSet<Node*>& dropout_nodes_to_transform,
                          int32_t& counter,
                          NodeIndex node_index) const;

  Status TransformBARTAttention(Graph& graph, bool& modified,
                                InlinedVector<Node*>& nodes_to_clear_shape,
                                InlinedHashSet<Node*>& dropout_nodes_to_transform,
                                int32_t& counter,
                                NodeIndex node_index) const;

  // Shared Utilities
  Status DoTransform(Graph& graph, bool& modified, int graph_level,
                     const logging::Logger& logger,
                     InlinedVector<Node*>& nodes_to_clear_shape,
                     InlinedHashSet<Node*>& dropout_nodes_to_transform) const;

  Status TransformDropout(Graph& graph, bool& modified, int graph_level,
                          const logging::Logger& logger,
                          InlinedHashSet<Node*>& dropout_nodes_to_transform,
                          int32_t& counter) const;

  template <class T>
  void PartitionBufferByColumn(const T* input,
                               const int64_t row_count,
                               const int64_t column_count,
                               const int64_t column_stride,
                               const int stride,
                               InlinedVector<T>& result) const;

  bool PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                               ONNX_NAMESPACE::TensorProto& initializer_partition,
                               int stride = 1) const;

  bool PartitionWeightByRow(const Graph& graph, const NodeArg& input_arg,
                            ONNX_NAMESPACE::TensorProto& initializer_partition) const;

  const int32_t horizontal_parallel_rank_;
  const int32_t horizontal_parallel_size_;
  std::unordered_map<std::string, std::string>& updated_weight_names_;
  std::unordered_set<std::string>& weights_to_train_;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo>& weight_partition_info_;
  training::TrainingSession::OptimizerState& initial_optimizer_states_;
  IExecutionProvider& cpu_execution_provider_;
};

}  // namespace onnxruntime
