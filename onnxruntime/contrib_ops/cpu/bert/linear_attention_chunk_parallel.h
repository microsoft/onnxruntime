// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/linear_attention_recurrent.h"  // for LinearAttentionUpdateRule

namespace onnxruntime {
namespace contrib {

template <typename T>
class LinearAttentionChunkParallel final : public OpKernel {
 public:
  explicit LinearAttentionChunkParallel(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  LinearAttentionUpdateRule update_rule_;
  int chunk_size_;
  float scale_;

  // Apply one recurrent step (token t) to the float state for head (b,h).
  // Identical math to LinearAttentionRecurrent::ComputeSingleHead.
  void StepSingleHead(
      const float* q,      // [d_k]
      const float* k,      // [d_k]
      const float* v,      // [d_v]
      float* state,        // [d_k * d_v], updated in-place
      const float* decay,  // [d_k] — already exp(·), or nullptr
      float beta_val,
      float* output,       // [d_v]
      int d_k, int d_v, float scale) const;
};

}  // namespace contrib
}  // namespace onnxruntime
