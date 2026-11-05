// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MultiHeadAttentionFusion

Fuses the unmasked scaled-dot-product attention emitted by ViT-style exporters (e.g. HuggingFace
ViT / DINOv2) with separate Q, K and V projections into a com.microsoft MultiHeadAttention node:

    Q -> Reshape[*,*,N,H] -> Transpose(0,2,1,3) --+
                                                  MatMul -> Div|Mul(scalar)? -> Softmax(-1)
    K -> Reshape[*,*,N,H] -> Transpose(0,2,3,1) --+                                 |
                                                                                    MatMul -> Transpose(0,2,1,3)
    V -> Reshape[*,*,N,H] -> Transpose(0,2,1,3) ------------------------------------+         -> Reshape(out)

is rewritten to

    Q -> Reshape[*,*,N,H] -> Reshape[0,0,-1] --+
    K -> Reshape[*,*,N,H] -> Reshape[0,0,-1] --+-> MultiHeadAttention(num_heads=N, scale) -> Reshape(out)
    V -> Reshape[*,*,N,H] -> Reshape[0,0,-1] --+

The added Reshapes only fold the trailing (N, H) dims back to the BSD layout MultiHeadAttention expects, so they are
free for EPs whose Reshape aliases its input. This keeps the [B, N, S, S] score tensor out of memory, which dominates
the attention cost for long vision sequences.

Besides the structural match, the fusion requires Q, K and V to share their batch dim, head count and head size, and
it only fires when the WebGPU MultiHeadAttention kernel would take its dense FlashAttention prefill path: a head size
that is a multiple of 8 and fits the guaranteed workgroup storage, and a query length that is not statically known to
be below 32. These cutoffs are specific to the WebGPU kernel, so the transformer is only meant to be registered for
the WebGPU EP.
*/
class MultiHeadAttentionFusion : public GraphTransformer {
 public:
  explicit MultiHeadAttentionFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers) noexcept
      : GraphTransformer("MultiHeadAttentionFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
