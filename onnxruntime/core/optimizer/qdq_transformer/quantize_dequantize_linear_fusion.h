// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QuantizeDequantizeLinearFusion

    Replaces a 16-bit QuantizeLinear whose output reaches only DequantizeLinear nodes with the same constant scalar
    scale and zero point, directly or through data-movement ops such as Transpose or Reshape, by a single
    com.microsoft.QuantizeDequantizeLinear node. The data-movement ops then run on the float values. The result is
    bitwise identical, but the quantized tensors and the extra nodes and kernel launches are removed.

    It must run after the transformers that fuse Q/DQ node units, since it removes the DQ/Q nodes they match.
*/
class QuantizeDequantizeLinearFusion : public GraphTransformer {
 public:
  explicit QuantizeDequantizeLinearFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QuantizeDequantizeLinearFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
