// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/rocm/bert/attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

struct InputMetadata;

template <typename T>
class PagedAttention final : public RocmKernel {
 public:
  PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(
      OpKernelContext* context,
      const InputMetadata* input_metadata,
      PackedAttentionParameters& parameters) const;
  Status RunMultiHeadAttention(Tensor* output, OpKernelContext* context, InputMetadata* input_metadata,
                               PackedAttentionParameters parameters, IAllocatorUniquePtr<T>& gemm_buffer) const;
  Status DoQKVProjectionIfNeed(OpKernelContext* context, InputMetadata* input_metadata, PackedAttentionParameters parameters,
                               IAllocatorUniquePtr<T>& gemm_buffer) const;

  int32_t num_heads_;                  // number of attention heads
  int32_t num_kv_heads_;                  // number of attention kv_heads
  int32_t head_size_;                      // number of attention heads
  float scale_;                            // sqrt(head_size_)
  std::string mask_type_;                  // position embedding type
  IAllocatorUniquePtr<int32_t> head_mapping_;
  int32_t num_queries_per_kv_;
  // std::shared_ptr<void> tunable_op_;
  void* flash_attention_v2_kernel_ = nullptr; // cuda kernel
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
