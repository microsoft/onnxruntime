// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cpu/bert/attention_base.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;

class QOrderedAttention final : public CudaKernel, public AttentionBase {
 public:
  explicit QOrderedAttention(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11040

 public:
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  int64_t input_hidden_size_;
  int64_t qkv_total_hidden_size_;
  BufferUniquePtr merged_qkv_weight_;
  BufferUniquePtr merged_qkv_alpha_;
  BufferUniquePtr merged_qkv_bias_;
  BufferUniquePtr softmax_lookup_;
  int order_input_;
  int order_weight_;
  int order_output_;
  float const_scale_input_;
  float const_scale_qkv_layer_[3];
  int qkv_weight_const_count_, scale_qkv_weight_const_count_, qkv_bias_const_cout_;

 private:
  Status PutIntoMergedWeight(const Tensor& tensor, AllocatorPtr alloc, int qkv_index, cudaStream_t cuda_stream);
  Status PutIntoMergedWeightScale(const Tensor& tensor, AllocatorPtr alloc, int qkv_index);
  Status PutIntoMergedBias(const Tensor& tensor, AllocatorPtr alloc, int qkv_index);

#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
