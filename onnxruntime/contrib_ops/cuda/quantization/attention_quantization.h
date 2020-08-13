// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T, typename TQuant>
class QAttention;

template <typename T>
class QAttention<T, int8_t> final : public CudaKernel, public AttentionBase {
  using Base = CudaKernel;

 public:
  QAttention(const OpKernelInfo& info) : CudaKernel(info),
                                         AttentionBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(const Tensor* input,
                     const Tensor* weights,
                     const Tensor* bias,
                     const Tensor* input_scale_tensor,
                     const Tensor* weight_scale_tensor,
                     const Tensor* mask_index,
                     const Tensor* i_zp_tensor,
                     const Tensor* w_zp_tensor,
                     const Tensor* past_tensor) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
