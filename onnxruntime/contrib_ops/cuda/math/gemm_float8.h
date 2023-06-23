// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "cublas_v2.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Calls https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul.
// D = alpha*(A*B)
class GemmFloat8 final : public onnxruntime::cuda::CudaKernel {
 public:
  GemmFloat8(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void set(const TensorShape& shape_a,
           const TensorShape& shape_b,
           int& M, int& N, int& K,
           int& lda, int& ldb, int& ldd) const;
  Status set_check(const TensorShape& shape_a,
                   const TensorShape& shape_b,
                   int& M, int& N, int& K) const;

  float alpha_;
  bool transA_;
  bool transB_;
  bool fast_accumulation_mode_;
  int64_t sm_count_;
  int64_t dtype_;
  // TODO: update the design when a decision is made about storage_order_.
  // The current implementation assumes the input tensor are stored based
  // on that order but that's not the case in onnxruntime. Tensor are always row major.
  cublasLtOrder_t storage_order_;
  cublasComputeType_t compute_type_;

  // TODO: add epilogue (= activation function, Relu or Gelu are available).
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
