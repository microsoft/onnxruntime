// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class DynamicQuantizeConvInteger
Fuse DynamicQuantizeLinear + ConvInteger and following cast and mul to Dequantze + Conv
*/
class DynamicQuantizeConvIntegerFusion : public GraphTransformer {
 public:
  DynamicQuantizeConvIntegerFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {kQnnExecutionProvider}) noexcept
      : GraphTransformer("DynamicQuantizeConvIntegerFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
