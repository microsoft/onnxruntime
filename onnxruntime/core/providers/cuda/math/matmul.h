// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cublasLt.h>

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info),
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0} {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;

  bool valid_algo_{false};
  size_t workspace_size_{0};
  cublasLtMatmulAlgo_t algo_;

  cublasLtMatmulDesc_t operationDesc{nullptr};
  cublasLtMatrixLayout_t Adesc{nullptr}, Bdesc{nullptr}, Cdesc{nullptr}, Ddesc{nullptr};
};
}  // namespace cuda
}  // namespace onnxruntime
