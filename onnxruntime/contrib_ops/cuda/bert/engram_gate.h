// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class EngramGate final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit EngramGate(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float epsilon_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
