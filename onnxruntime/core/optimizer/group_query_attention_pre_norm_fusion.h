// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GroupQueryAttentionPreNormFusion

Folds the Qwen3-style per-head Q/K RMSNorm prologue into the GroupQueryAttention
node by adding optional q_norm_weight and k_norm_weight inputs (slots 14 and 15)
and a qk_norm_epsilon attribute. The transform looks for the following pattern
on inputs 0 (query) and 1 (key) of an unfused GroupQueryAttention node:

    Q_proj_out -> Reshape[*,*,head_size]
               -> SimplifiedLayerNormalization(weight = q_norm_weight)
               -> Reshape[*,*,num_heads * head_size]
               -> GQA[input 0]

    K_proj_out -> Reshape[*,*,head_size]
               -> SimplifiedLayerNormalization(weight = k_norm_weight)
               -> Reshape[*,*,kv_num_heads * head_size]
               -> GQA[input 1]

When matched, the six Reshape/SLN nodes are removed and the pre-norm Q and K
projections feed GQA directly. The kernel is responsible for applying the RMS
norm internally (currently the CUDA and WebGPU EPs).

Only fires for execution providers passed in `compatible_execution_providers`.
At present this fusion is registered for the CUDA and WebGPU EPs, because those
kernels implement the in-kernel norm path. CPU and JSEP GroupQueryAttention
kernels reject q_norm_weight / k_norm_weight inputs.
*/
class GroupQueryAttentionPreNormFusion : public GraphTransformer {
 public:
  explicit GroupQueryAttentionPreNormFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GroupQueryAttentionPreNormFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
