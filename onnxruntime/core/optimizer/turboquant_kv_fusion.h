// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class TurboQuantKVFusion

Pattern-matches existing GroupQueryAttention nodes in the graph and rewrites
them to use TurboQuant KV cache compression.

What it does:
  1. For each GroupQueryAttention node, set kv_quant_method = "turboquant",
     key_quant_bits = 3 or 4, value_quant_bits = 4, norm_correction = true
     (configurable via session options).
  2. Inject the static Lloyd-Max codebook (computed by the Python calibration
     tool turboquant_kv_quantizer.py) as a constant graph initializer
     `<node_name>__k_centroids`.
  3. Inject the Walsh-Hadamard rotation matrix as a constant initializer
     `<node_name>__hadamard` (head_dim x head_dim, fp16).
  4. Rewrite past_key_values / present_key_values tensor types from fp16 to
     uint8 with the new packed shape.

Mirrors the structure of:
  - core/optimizer/group_query_attention_fusion.cc
  - core/optimizer/dq_matmulnbits_fusion.cc

Gated by session option kOrtSessionOptionsTurboQuantKV.
*/
class TurboQuantKVFusion : public GraphTransformer {
 public:
  TurboQuantKVFusion(
      const std::string& preset = "",
      int boundary_n = 2,
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TurboQuantKVFusion", compatible_execution_providers),
        preset_(preset),
        boundary_n_(boundary_n) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

  std::string preset_;
  int boundary_n_;
};

}  // namespace onnxruntime
