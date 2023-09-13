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

struct InputMetadata;

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
  Status CheckInputs(
      const Tensor* query,
      const Tensor* key,
      const Tensor* value,
      const InputMetadata* input_metadata,
      PackedAttentionParameters& parameters) const;
  Status RunMultiHeadAttention(Tensor* output, OpKernelContext* context, PackedAttentionParameters parameters, IAllocatorUniquePtr<T>& gemm_buffer) const;
  Status DoQKVProjectionIfNeed(OpKernelContext* context, PackedAttentionParameters parameters,
                               IAllocatorUniquePtr<T>& gemm_buffer) const;

  int32_t num_heads_;                  // number of attention heads
  int32_t num_kv_heads_;                  // number of attention kv_heads
  int32_t head_size_;                      // number of attention heads
  float scale_;                            // sqrt(head_size_)
  std::string mask_type_;                  // position embedding type
  void* flash_attention_v2_kernel_ = nullptr;  // cuda kernel
  IAllocatorUniquePtr<int32_t> head_mapping_;
  int32_t num_queries_per_kv_;

  AttentionSelector<T> selector_;
  friend struct AttentionSelector<T>;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
