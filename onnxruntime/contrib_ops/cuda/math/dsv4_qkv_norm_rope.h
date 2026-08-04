// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// Fuses the per-token normalisation, partial rotary embedding and simulated activation
// quantisation that sit between the QKV projections and the attention kernel.
template <typename T>
class DSV4QKVNormRope final : public CudaKernel {
 public:
  DSV4QKVNormRope(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  float epsilon_;
  bool act_quant_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
