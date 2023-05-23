// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace contrib {
namespace cuda {

// It probably exists some where already.
template <typename T>
cudaDataType ToCudaDataType();

struct GemmFloat8_Impl;

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
struct GemmFloat8_Impl_Compute {
  onnxruntime::Status Compute(const GemmFloat8_Impl& params, cudaStream_t stream, cublasLtHandle_t handle,
               const Tensor* A, const Tensor* B, const Tensor* C, Tensor* D, BiasType* relu_bias,
               int M, int N, int K, int lda, int ldb, int ldd) const;
};

struct GemmFloat8_Impl {
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescAttributes_t#cublasltmatmuldescattributes-t
  bool fast_accumulation_mode_;
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasComputeType_t#cublascomputetype-t
  cublasComputeType_t compute_type_;
  int64_t sm_count_;
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;

  void set(int M, int N, int K, int& lda, int& ldb, int& ldd) const;

  template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
  onnxruntime::Status CudaCompute(cudaStream_t stream, cublasLtHandle_t handle,
                   const Tensor* A, const Tensor* B, const Tensor* C, Tensor* D, BiasType* relu_bias,
                   int M, int N, int K) const {
    int lda, ldb, ldd;
    set(M, N, K, lda, ldb, ldd);
    return GemmFloat8_Impl_Compute<AType, BType, CType, DType, BiasType>().Compute(*this, stream, handle, A, B, C, D, relu_bias, M, N, K, lda, ldb, ldd);
  }
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
