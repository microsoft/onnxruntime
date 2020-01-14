// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status SoftMaxComputeHelper(
    const T* input,
    const TensorShape& shape,
    T* Y,
    cudnnHandle_t handle,
    int64_t axis);

template <typename T>
class Softmax final : public CudaKernel {
 public:
  Softmax(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

template <typename T>
class SoftmaxGrad final : public CudaKernel {
 public:
  SoftmaxGrad(const OpKernelInfo& info) : CudaKernel{info} {
    info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(1));
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
