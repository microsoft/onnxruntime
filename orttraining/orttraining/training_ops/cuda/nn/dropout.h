// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/nn/dropout_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2>
class DropoutGrad final : public CudaKernel {
 public:
  DropoutGrad(const OpKernelInfo& info) : CudaKernel(info), default_ratio_(0.5) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float default_ratio_;
};

}  // namespace cuda
}  // namespace onnxruntime
