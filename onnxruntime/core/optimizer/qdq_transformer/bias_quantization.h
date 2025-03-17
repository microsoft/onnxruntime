// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @class BiasQuantization
 *
 * Some quantized models do not have Gemm/Conv's bias quantized. This optimization adds a subgraph to quantize the bias
 * with scale = scale_input_0 * scale_input_1 and zero_point = 0.
 *
 * Normally the ConstantFolding optimizer would fold the bias initializer into an int32_t initializer, which is consumed
 * by a DequantizeLinear node.
 */
class BiasQuantization : public GraphTransformer {
 public:
  BiasQuantization() noexcept : GraphTransformer("BiasQuantization") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
