// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Scale final : public CudaKernel {
 public:
  Scale(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool scale_down_;
};

}  // namespace cuda
}  // namespace onnxruntime
