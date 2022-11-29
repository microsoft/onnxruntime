// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class QDQFusion
from:
         x (fp32)
            |
      QuantizeLinear
            |
    y (quantized type)
            |
     DeQuantizeLinear
            |
         z (fp32)

to:
         x (fp32)
            |
        FakeQuant
            |
         z (fp32)

*/

class QDQFusion : public GraphTransformer {
 public:
  explicit QDQFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
