// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cublas_v2.h"
#include "core/providers/cuda/cuda_kernel.h"

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
  void SetParams(const TensorShape& shape_a,
                 const TensorShape& shape_b,
                 int& M, int& N, int& K,
                 int& lda, int& ldb, int& ldd) const;
  Status SetCheck(const TensorShape& shape_a,
                  const TensorShape& shape_b,
                  int& M, int& N, int& K) const;

  Status ComputeRowMajor(OpKernelContext* ctx, int n_inputs, bool has_bias,
                         bool has_scales, const Tensor* input_A,
                         const Tensor* input_B, const Tensor* input_C,
                         const Tensor* scale_A, const Tensor* scale_B,
                         const Tensor* scale_Y) const;
  Status ComputeColMajor(OpKernelContext* ctx, int n_inputs, bool has_bias,
                         bool has_scales, const Tensor* input_A,
                         const Tensor* input_B, const Tensor* input_C,
                         const Tensor* scale_A, const Tensor* scale_B,
                         const Tensor* scale_Y) const;

  Status ComputeGemm(
      OpKernelContext* ctx, int n_inputs, bool has_bias, bool has_scales,
      int32_t dtype_A, int32_t dtype_b,
      int32_t dtype_c, int32_t dtype_Y,
      const TensorShape& shape_A, const TensorShape& shape_B,
      const TensorShape& shape_C, const TensorShape& shape_Y,
      bool transa, bool transb, const void* p_input_a, const void* p_input_b,
      const void* p_input_c, const void* p_scale_a, const void* p_scale_b,
      const void* p_scale_y, void* p_output_y, int M, int N, int K, int lda,
      int ldb, int ldd, bool row_major_compute) const;

  float alpha_;
  float beta_;
  bool transA_;
  bool transB_;
  int64_t sm_count_;
  int64_t dtype_;
  cublasLtEpilogue_t epilogue_;

  // TODO(xadupre): add epilogue (= activation function, Relu or Gelu are available).
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
