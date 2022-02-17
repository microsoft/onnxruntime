// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "orttraining/training_ops/cuda/nn/bitmask_dropout_grad_impl.h"

namespace onnxruntime {
namespace cuda {

class BitmaskDropoutGrad final : public CudaKernel {
 public:
  BitmaskDropoutGrad(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  static constexpr float default_ratio_ = 0.5f;
};

}  // namespace cuda
}  // namespace onnxruntime
