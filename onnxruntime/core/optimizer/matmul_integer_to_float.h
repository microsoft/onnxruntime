// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MatMulIntegerToFloatFusion
Fuse MatMulInteger and corresponding cast and mul to MatMulIntegerToFloat
*/
class MatMulIntegerToFloatFusion : public GraphTransformer {
 public:
  MatMulIntegerToFloatFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MatMulIntegerToFloatFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
