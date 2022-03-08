// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename TIn, typename TOut>
class ReduceAllL2 final : public CudaKernel {
 public:
  ReduceAllL2(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
