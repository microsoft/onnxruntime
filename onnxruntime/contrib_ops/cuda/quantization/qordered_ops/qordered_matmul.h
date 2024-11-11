// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using onnxruntime::cuda::CudaKernel;

class QOrderedMatMul final : public CudaKernel {
 public:
  QOrderedMatMul(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  int order_A_;
  int order_B_;
  int order_Y_;
  float const_scale_A_;
  int scale_b_size_;
  const float* origin_scale_B_vector_;
  BufferUniquePtr calculated_alpha_;
  BufferUniquePtr const_bias_scaled_;
  int const_bias_size_;
  float const_scale_B_;
  float const_scale_C_;
  float const_scale_Y_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
