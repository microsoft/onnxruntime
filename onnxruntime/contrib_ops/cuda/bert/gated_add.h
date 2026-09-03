// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// output = x + round_to_T(y * gate), with gate broadcast across the last dimension.
template <typename T>
class GatedAdd final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit GatedAdd(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime