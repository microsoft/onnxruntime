// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "contrib_ops/cuda/bert/packed_attention.h"

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "driver_types.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

struct SelectResult {
  size_t workSpaceSize = 0;
  void* fused_runner = nullptr;
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
  bool no_qkv_workspace = false;
};

template <typename T>
class PagedAttention;

template <typename T>
struct AttentionSelector {
 public:
  AttentionSelector(PagedAttention<T>* op);

  SelectResult Select(PackedAttentionParameters parameters, const cudaDeviceProp& device_prop) const;

 public:
  bool disable_flash_attention_;
  bool disable_TRT_flash_attention_;
  bool disable_memory_efficient_attention_;
  bool enable_fused_causal_attention_;
  PagedAttention<T>* op_;
};

template <typename T>
class PagedAttention final : public TrtFusedAttention<T>, public CudaKernel {
 public:
  PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status RotaryEmbeddings(const GroupQueryAttentionParameters& parameters,
                          const Tensor* query, const Tensor* key,
                          const Tensor* cos_cache, const Tensor* sin_cache) const;

  Status GroupQueryAttention(const GroupQueryAttentionParameters& parameters,
                             const Tensor* query, const Tensor* key, const Tensor* value,
                             const Tensor* seqlens_k, const Tensor* cos_cache,
                             const Tensor* sin_cache, Tensor* output,
                             OpKernelContext* context) const;

  Status WriteToPagedCache(const GroupQueryAttentionParameters& parameters,
                           OpKernelContext* context,
                           const Tensor* key, const Tensor* value,
                           const Tensor* slot_mappings,
                           Tensor* key_cache, Tensor* value_cache) const;

  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool is_unidirectional_;
  bool is_past_bsnh_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  static constexpr int kZerosCount = 256;  // In prompt case we create a zero buffer of size 256 for seqlen (assume batch_size <= 256)
  IAllocatorUniquePtr<int> zeros_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
