// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Rewrites the standard CPU-bound attention-mask reformatting subgraph that feeds
// GroupQueryAttention's seqlens_k (input 5) and total_sequence_length (input 6)
// inputs into an equivalent GPU-resident form, so that WebGPU graph capture can
// successfully replay the resulting graph.
//
// Standard (CPU-bound) form, as emitted by genai builders when the
// `enable_webgpu_graph` flag is OFF:
//
//   attention_mask --+-> ReduceSum(axes=[1], keepdims=0) -> Sub(_, 1) -> Cast(INT32) -> seqlens_k
//                    |
//                    +-> Shape -> Gather(_, 1, axis=0) -> Cast(INT32) -> total_seq_len
//
// The Shape op has CPU-only output, so under graph capture the captured GPU
// command list cannot be replayed without a CPU dependency, and seqlens_k /
// total_seq_len become stale, corrupting attention.
//
// Rewritten (GPU-friendly) form:
//
//   attention_mask -> Cast(INT32) -> ReduceSum(axes=[1], keepdims=0) -+-> Sub(_, 1) -> seqlens_k
//                                                                     |
//                                                                     +-> ReduceMax(axes=[]) -> total_seq_len
//
// The rewrite relies on the invariant that the trailing position of the
// attention_mask is always 1 (true for genai left-padded causal masks), so
// `ReduceMax(ReduceSum(mask, axis=1))` is equivalent to `Shape(mask)[1]`.
class GqaMaskReformattingToGraphCapture : public GraphTransformer {
 public:
  explicit GqaMaskReformattingToGraphCapture(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GqaMaskReformattingToGraphCapture", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
