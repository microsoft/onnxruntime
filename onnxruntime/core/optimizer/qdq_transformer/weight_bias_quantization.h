// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @class WeightBiasQuantization
 *
 * Some quantized models do not have Gemm/Conv/ConvTranspose's weight and/or bias quantized. This optimization adds
 * subgraphs to quantize the weight and/or bias.
 *   For weight, it's quantized to symmetric per-tensor INT8 tensor.
 *   For bias, it's quantized to a INT32 tensor with scale = scale_input_0 * scale_input_1 and zero_point = 0.
 *
 * Normally the ConstantFolding optimizer would fold the weight and/or bias initializers into the target initializers,
 * which is consumed by a DequantizeLinear node respectively.
 */
class WeightBiasQuantization : public GraphTransformer {
 public:
  WeightBiasQuantization() noexcept : GraphTransformer("WeightBiasQuantization") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
