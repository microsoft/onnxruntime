// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MulReduceSumToMatMulFusion

Rewrites a vector-matrix product written as a broadcast multiply and a sum as a MatMul:

    ReduceSum(Mul(X, Unsqueeze(v, axes=[-1])), axes=[-2], keepdims=0)
    -> Squeeze(MatMul(Unsqueeze(v, axes=[-2]), X), axes=[-2])

with X of shape [..., K, N] and v of shape [..., K]. Both compute sum_k v[..., k] * X[..., k, n], but the MatMul does
not write the full-size product to memory and read it back. With keepdims=1 the Squeeze is left out.

The rewrite requires X's second-to-last dimension and v's last dimension to be the same, so that the Mul only
broadcasts v along X's last axis and the MatMul accepts its inputs.
*/
class MulReduceSumToMatMulFusion : public GraphTransformer {
 public:
  explicit MulReduceSumToMatMulFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MulReduceSumToMatMulFusion", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
