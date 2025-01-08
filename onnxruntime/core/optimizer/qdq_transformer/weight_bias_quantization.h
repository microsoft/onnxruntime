// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @class WeightBiasQuantization
 *
 * Some quantized models do not have Gemm/Conv/ConvTranspose's weight and/or bias quantized. This optimization adds
 * subgraphs with Q->DQ after weight and/or bias. It's possible that the ConstantFolding optimizer would fold the Q Op
 * so that weight and/or bias initializers are folded to initializers in target data types, followed by DQ Op.
 *   For weight, the Q output is a symmetric per-tensor INT8 tensor.
 *   For bias, the Q's scale = scale_input_0 * scale_input_1 and zero_point = (INT32)0.
 */
class WeightBiasQuantization : public GraphTransformer {
 public:
  WeightBiasQuantization() noexcept : GraphTransformer("WeightBiasQuantization") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
