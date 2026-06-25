// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class LpNorm final : public CudaKernel {
 public:
  LpNorm(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("p", &p_).IsOK());
    ORT_ENFORCE(p_ == 1 || p_ == 2);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t p_;
};

}  // namespace cuda
}  // namespace onnxruntime
