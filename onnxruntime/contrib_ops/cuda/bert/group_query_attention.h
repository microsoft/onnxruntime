// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename U>
class GroupQueryAttention final : public CudaKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool is_unidirectional_;
  bool is_past_bsnh_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool use_smooth_softmax_;
  float scale_;
  float softcap_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  bool disable_flash_decode_;
  bool enable_xqa_;

  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  int kv_cache_bit_width_;

  static constexpr int kZerosCount = 256;  // In prompt case we create a zero buffer of size 256 for seqlen (assume batch_size <= 256)
  IAllocatorUniquePtr<int> zeros_;
  const AttentionKernelOptions* kernel_options_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
