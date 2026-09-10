// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/dynamic_sparse_attention_helper.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class DynamicSparseAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit DynamicSparseAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  int rotary_offset_;
  float scale_;
  float qk_norm_epsilon_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool use_smooth_softmax_;
  bool auxiliary_kv_shared_;
  DynamicSparseAttentionMode attention_mode_;
  DynamicSparseAttentionKvSource selected_kv_source_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
