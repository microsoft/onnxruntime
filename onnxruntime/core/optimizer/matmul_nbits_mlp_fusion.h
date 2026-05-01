// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Fuses the SwiGLU MLP block (gate / up / down MatMulNBits projections around a
// SimplifiedLayerNormalization anchor) into a single MatMulNBitsMlp contrib op:
//
//   ... -> [Skip]SimplifiedLayerNormalization -+-> MatMulNBits (gate) -+-> Sigmoid -+
//                              |               |                       |            v
//                              |               |                       +----------> Mul (silu) -+
//                              |               +-> MatMulNBits (up) ---------------------------+--> Mul -> MatMulNBits (down) -> out
//                              +--> (optional) skip residual passthrough --> downstream consumers
//
// becomes
//
//   ... -> [Skip]SimplifiedLayerNormalization --> MatMulNBitsMlp(activation="silu") -+-> out
//                                                                                    +-> (optional) residual passthrough
//
// Only activation="silu" (i.e. x * Sigmoid(x)) is matched / emitted, and the fusion is restricted
// to the WebGPU EP because MatMulNBitsMlp is a WebGPU-only contrib op.
class MatMulNBitsMlpFusion : public GraphTransformer {
 public:
  explicit MatMulNBitsMlpFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MatMulNBitsMlpFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
