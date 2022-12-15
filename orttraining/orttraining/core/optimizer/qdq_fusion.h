// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class QDQFusion

This transformer will be used during QAT (Quantization Aware Training). For QAT scenarios,
an onnx graph that has Q->DQ nodes needs to be made ready for training. The output of the
Q node is a quantized type. Backpropagation on quantized type is not supported in ort.
So, we replace the occurrences of Q->DQ with FakeQuant which internally will perform the
Q->DQ opeeration and at the same time can support backpropagation.

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
