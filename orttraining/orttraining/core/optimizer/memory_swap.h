// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MemorySwap

Graph transfomer for adding memory swap nodes.
*/
class MemorySwap : public GraphTransformer {
 public:
  MemorySwap(const std::string& stop_at_node_arg) noexcept
      : GraphTransformer("MemorySwap", {kCudaExecutionProvider}),
        stop_at_node_arg_(stop_at_node_arg) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  std::string stop_at_node_arg_;  // name of output[0] for the node to stop memory swap
};

}  // namespace onnxruntime
