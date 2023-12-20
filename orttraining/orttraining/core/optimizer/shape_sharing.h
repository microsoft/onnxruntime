// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class ShapeSharing

Transformer that traverses the graph top-down and performs Shape node sharing, i.e.,
Shape nodes having the same rank and dimensions will be replaced by one single (newly created) Shape node.
*/
class ShapeSharing : public GraphTransformer {
 public:
  /**
   * @param compatible_execution_providers compatible execution provider list for considered nodes.
   */
  ShapeSharing(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ShapeSharing", compatible_execution_providers) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
