// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GeluFusion

Rewrite graph fusing Gelu activation subgraph to a single Gelu node.

The formula corresponding to Gelu activation subgraph:
x * 0.5 * (1.0 + erf(x / sqrt(2.0))), where x is the input.

*/
class GeluFusion : public GraphTransformer {
 private:
  TransformerLevel optimization_level_ = TransformerLevel::Level1;
  bool allow_contrib_op_in_level_1_ = false;
  std::string GetGeluFusionName(TransformerLevel level) {
    switch (level) {
      case TransformerLevel::Level1:
        return "GeluFusionL1";
      case TransformerLevel::Level2:
        return "GeluFusionL2";
      default:
        return "GeluFusion";
    }
  }

 public:
  GeluFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
             TransformerLevel level = TransformerLevel::Level1, bool allow_contrib_op_in_level_1 = false) noexcept
      : GraphTransformer(GetGeluFusionName(level), compatible_execution_providers),
        optimization_level_(level),
        allow_contrib_op_in_level_1_(allow_contrib_op_in_level_1) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
