// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {

class MatMulComputeHelper;

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

  struct SparseInfo;

 private:
#ifdef USE_CUSPARSELT
  Status PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) override;
  Status ComputeSparse(const MatMulComputeHelper& helper, bool transa, bool transb,
                       const Tensor* left, const Tensor* right, Tensor* C) const;
#endif

  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  // Argument 1 is a sparse weight coming from constant initializer
  // if set
  std::unique_ptr<SparseInfo> sparse_info_;
};
}  // namespace cuda
}  // namespace onnxruntime
