// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

class MegatronTransformer : public GraphTransformer {
 public:
  MegatronTransformer(int32_t horizontal_parallel_rank, int32_t horizontal_parallel_size,
                      const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MegatronTransformer", compatible_execution_providers),
        horizontal_parallel_rank_(horizontal_parallel_rank),
        horizontal_parallel_size_(horizontal_parallel_size) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

  Status TransformMLP(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                      std::vector<Node*>& nodes_to_clear_shape) const;

  bool PartitionWeightByColumn(const Graph& graph, const NodeArg& input_arg,
                               ONNX_NAMESPACE::TensorProto& initializer_partition) const;

  bool PartitionWeightByRow(const Graph& graph, const NodeArg& input_arg,
                            ONNX_NAMESPACE::TensorProto& initializer_partition) const;

 private:
  const int32_t horizontal_parallel_rank_;
  const int32_t horizontal_parallel_size_;
};

}  // namespace onnxruntime
