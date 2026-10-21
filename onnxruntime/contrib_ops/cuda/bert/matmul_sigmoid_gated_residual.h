// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulSigmoidGatedResidual final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulSigmoidGatedResidual(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

Status LaunchMatMulSigmoidGatedResidual(cudaStream_t stream, const void* x, const void* weight,
                                       const void* shared, const void* routed, const void* residual,
                                       void* output, int rows, int k, int n);

}  // namespace onnxruntime::contrib::cuda
