// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Fuses three sibling MatMulNBits Q/K/V projections that share a SimplifiedLayerNormalization
// (or SkipSimplifiedLayerNormalization) anchor into a single MatMulNBitsQkv contrib op:
//
//   ... -> [Skip]SimplifiedLayerNormalization -+-> MatMulNBits (Q proj) -+
//                              |               +-> MatMulNBits (K proj) -+--> downstream consumers
//                              |               +-> MatMulNBits (V proj) -+
//                              +--> (optional) skip residual passthrough --> downstream consumers
//
// becomes
//
//   ... -> [Skip]SimplifiedLayerNormalization --> MatMulNBitsQkv -+-> Q out
//                                                                 +-> K out
//                                                                 +-> V out
//                                                                 +-> (optional) residual passthrough
//
// The fusion is restricted to the WebGPU EP because MatMulNBitsQkv is a WebGPU-only contrib op.
class MatMulNBitsQkvFusion : public GraphTransformer {
 public:
  explicit MatMulNBitsQkvFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MatMulNBitsQkvFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
