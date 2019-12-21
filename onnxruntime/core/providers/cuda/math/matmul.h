// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/cublas_gemm_algo_selector.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  CublasGemmAlgoSelector gemm_algo_selector;
};
}  // namespace cuda
}  // namespace onnxruntime
