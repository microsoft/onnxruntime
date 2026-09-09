// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
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
  bool is_causal_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  float softcap_;
  float qk_norm_epsilon_;
  KVQuantizationType k_quant_type_;
  KVQuantizationType v_quant_type_;
  // Logical element type stored in the cache when it cannot be expressed by the cache tensor's own
  // element type (sub-byte formats packed into uint8). DEFAULT uses the tensor's element type. The
  // attribute string is parsed once here so the hot path only compares enums.
  KVCacheDataType k_cache_dtype_;
  KVCacheDataType v_cache_dtype_;
  // Multi-head Latent Attention (docs/contrib_ops/cuda/paged_attention.md §12). is_latent_kv_ comes
  // from kv_cache_layout == "LATENT"; v_head_size_ == 0 means "same as head_size".
  bool is_latent_kv_;
  int v_head_size_;
  int rotary_offset_;
  bool has_explicit_scale_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  bool disable_paged_decode_;
  // Tensor-core XQA decode kernel for a quantized paged cache. Defaults on; ORT_ENABLE_XQA=0
  // disables it and falls back to the portable PagedDecodeSplitKV kernel.
  bool enable_xqa_;
  // Native FP16/BF16 cache specializations are opt-in because FlashAttention is competitive.
  bool enable_native_xqa_;
  // cuDNN paged SDPA (decode-only tier). Mirrors GroupQueryAttention: the standard sdpa_kernel bit
  // and ORT_ENABLE_CUDNN_FLASH_ATTENTION opt users in explicitly, and sm>=90 gets it automatically
  // via AllowCudnnFlashAttentionAuto(). ORT_ENABLE_CUDNN_FLASH_ATTENTION=0 is the shared kill switch
  // for every cuDNN attention path in the CUDA EP.
  bool enable_cudnn_paged_;
  bool auto_enable_cudnn_paged_;
  // -1 = not yet resolved, 0 = the kernel needs more shared memory than this device allows,
  // 1 = it fits. Resolved once per node because it only depends on head_size / group size.
  mutable std::atomic<int> xqa_shared_memory_ok_{-1};
  mutable std::atomic<int> xqa_spec_dec_shared_memory_ok_{-1};
  // -1 = not yet probed, 0 = try_build_paged_graph returned false (planner rejected the shape
  // signature for this node), 1 = the cuDNN paged graph built successfully. Probed on the first
  // non-capturing Compute call when cuDNN paged is otherwise eligible; a failed probe makes the
  // cascade fall back to FlashAttention / MemoryEfficientAttention for the remainder of the
  // node's lifetime rather than throwing at build time and killing user inference.
  mutable std::atomic<int> cudnn_paged_build_ok_{-1};
  const AttentionKernelOptions* kernel_options_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
