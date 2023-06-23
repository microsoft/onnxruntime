// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "gemm_float8_impl.cuh"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace contrib {
namespace cuda {

class GemmFloat8 final : public CudaKernel {
  using Base = CudaKernel;

 public:
  GemmFloat8(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  GemmFloat8_Impl params_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
