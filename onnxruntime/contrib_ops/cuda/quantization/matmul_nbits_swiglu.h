// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulNBitsSwiGLU final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulNBitsSwiGLU(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int k_;
  int n_;
};

Status LaunchMatMulNBitsSwiGLU(cudaStream_t stream, const void* x, const uint8_t* weight,
                               const void* scales, void* output, int m, int n, int k);
Status LaunchMatMulNBitsSwiGLUEpilogue(cudaStream_t stream, const void* projected,
                                       void* output, int m, int n);

}  // namespace onnxruntime::contrib::cuda
