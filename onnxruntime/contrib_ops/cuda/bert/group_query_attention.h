// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class GroupQueryAttention final : public CudaKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;    // number of attention heads
  int num_heads_k_;  // different for k and v for group query attention
  bool is_unidirectional_; // causal
  float scale_;
  bool disable_flash_attention_;
  int min_seq_len_for_flash_attention_packed_qkv_;
  mutable CumulatedSequenceLengthCache cumulated_sequence_length_q_cache_;
  mutable CumulatedSequenceLengthCache cumulated_sequence_length_kv_cache_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
