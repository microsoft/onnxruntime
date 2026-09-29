// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/gemm_auto_tuner.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace cuda {

// Resolves the fp16/bf16 small-M kernel policy from ORT_ENABLE_SMALL_N_GEMV, the session config
// ep.cuda.enable_gemm_auto_tune and ORT_CUDA_GEMM_AUTO_TUNE. The default is cuBLAS.
GemmDispatchPolicy GetGemmDispatchPolicy(const OpKernelInfo& info);

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
        trans_batch_b_{info.GetAttrOrDefault<int64_t>("transBatchB", 0) != 0},
        gemm_policy_{GetGemmDispatchPolicy(info)} {}

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeDefault(OpKernelContext* context, MatMulComputeHelper& helper) const;

 private:
  // Picks cuBLAS (`run_cublas`) or one of the eligible `candidates` (GemmKernelBit mask) for a single GEMM.
  template <typename RunCublas>
  Status SelectGemmKernel(OpKernelContext* ctx, const void* a, const void* b, void* c, int m, int n, int k,
                          uint8_t candidates, const RunCublas& run_cublas, GemmKernel& selected) const;
  Status RunGemmKernel(OpKernelContext* ctx, GemmKernel kernel, const void* a, const void* b, void* c, int m, int n,
                       int k) const;

  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;
  const GemmDispatchPolicy gemm_policy_;
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
