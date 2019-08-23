// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Clip final : public CudaKernel {
 public:
  Clip(const OpKernelInfo& info) : CudaKernel{info} {
    ORT_ENFORCE(info.GetAttr<T>("max", &max_).IsOK());
    ORT_ENFORCE(info.GetAttr<T>("min", &min_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  T min_, max_;
};

}  // namespace cuda
}  // namespace onnxruntime
