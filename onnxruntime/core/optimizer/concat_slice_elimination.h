// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class ConcatSliceElimination
*/
class ConcatSliceElimination : public GraphTransformer {
 public:
  ConcatSliceElimination(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ConcatSliceElimination", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
private:
  static bool FuseConcatSliceSubgraph(Node& concat, Graph& graph, const logging::Logger& logger);
};

}  // namespace onnxruntime
