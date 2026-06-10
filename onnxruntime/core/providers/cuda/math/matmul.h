// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"

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
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0},
        trans_batch_a_{info.GetAttrOrDefault<int64_t>("transBatchA", 0) != 0},
        trans_batch_b_{info.GetAttrOrDefault<int64_t>("transBatchB", 0) != 0} {}

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeDefault(OpKernelContext* context, MatMulComputeHelper& helper) const;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

 private:
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;

  // Decode (M==1) GEMV fast path: a transposed [N, K] copy of a small constant
  // fp16/bf16 weight is built once in PrePack so ComputeInternal can dispatch a
  // single-kernel GEMV instead of cuBLAS's split-K + reduce. The original B
  // input is left intact (is_packed stays false) so the M>1 cuBLAS path still
  // works.
  IAllocatorUniquePtr<void> gemv_b_transposed_;
  bool gemv_enabled_ = false;
  int64_t gemv_n_ = 0;
  int64_t gemv_k_ = 0;
};

template <typename T>
Status FuncMatMul(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const CudaKernel* cuda_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const Tensor* A,
    const Tensor* B,
    float alpha,
    bool trans_A,
    bool trans_B,
    bool trans_batch_A,
    bool trans_batch_B,
    Tensor* Y);

}  // namespace cuda
}  // namespace onnxruntime
