// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// T is the activation type (float16 / bfloat16). TCACHE is the paged KV cache element type:
// the same as T for an unquantized cache, or int8_t / Float8E4M3FN for a quantized one.
template <typename T, typename TCACHE>
class PagedAttention final : public CudaKernel {
 public:
  PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool smooth_softmax_;
  float scale_;
  float softcap_;
  float qk_norm_epsilon_;
  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  int kv_cache_bit_width_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  const AttentionKernelOptions* kernel_options_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
