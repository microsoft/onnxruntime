// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
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

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 protected:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool is_unidirectional_;
  bool is_past_bsnh_;
  bool do_rotary_;
  bool rotary_interleaved_;
  bool use_smooth_softmax_;
  float qk_norm_epsilon_;  // epsilon for the per-head Q/K RMSNorm (QK-Norm) prologue
  float scale_;
  float softcap_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  bool disable_flash_decode_;
  bool enable_xqa_;                         // True when ORT_ENABLE_XQA != 0 (default: on) and T is fp16/bf16.
  bool enable_cudnn_flash_attention_;       // cuDNN SDPA explicitly enabled (env / sdpa_kernel)
  bool auto_enable_cudnn_flash_attention_;  // auto-prefer cuDNN SDPA on SM>=90 when no explicit kernel pinned

  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  int kv_cache_bit_width_;

  static constexpr int kZerosCount = 256;  // In prompt case we create a zero buffer of size 256 for seqlen (assume batch_size <= 256)
  IAllocatorUniquePtr<int> zeros_;
  // FP32 head_sink cached in PrePack for the XQA path (empty when head_sink is not a constant initializer).
  IAllocatorUniquePtr<float> xqa_head_sink_;
  int xqa_head_sink_count_ = 0;  // Number of elements in xqa_head_sink_ (0 when not prepacked).
  // Cached result of the XQA shared-memory fit check for this node (-1 unknown, 0 does not fit,
  // 1 fits). head_size and group size are constant for a given GQA node, so the (device-symbol)
  // query is done once and reused to avoid a per-decode-step device->host copy.
  mutable std::atomic<int> xqa_shared_memory_ok_{-1};
  const AttentionKernelOptions* kernel_options_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
