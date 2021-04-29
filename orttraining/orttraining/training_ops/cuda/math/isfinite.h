// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
class IsFiniteOp final : public CudaKernel {
 public:
  IsFiniteOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename TSrc>
void IsFinite(cudaStream_t stream, const TSrc* input, bool* output, size_t N);

}  // namespace cuda
}  // namespace onnxruntime