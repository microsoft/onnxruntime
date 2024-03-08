// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class ShapeInputMerge
Merge all shape inputs having same shape value to a single shape input.
This change will not affect the performance, but it open chances for CSE fusion to merge nodes.
*/
class ShapeInputMerge : public GraphTransformer {
 public:
  ShapeInputMerge(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ShapeInputMerge", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
