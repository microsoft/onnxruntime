// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class AttentionFusion
Rewrite graph fusing attention subgraph to a single Attention node.
*/
class AttentionFusion : public GraphTransformer {
 public:
  AttentionFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("AttentionFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

private:
  static bool FuseSubGraph(Node& layer_norm, const Node& add_after_layer_norm, Graph& graph, int64_t hidden_size, std::map<std::string, NodeArg*>& mask_index_map, const logging::Logger& logger);
};

}  // namespace onnxruntime
