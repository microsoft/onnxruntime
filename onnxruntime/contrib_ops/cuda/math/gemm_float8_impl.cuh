// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct GemmFloat8_Impl {
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescAttributes_t#cublasltmatmuldescattributes-t
  bool fastAccumulationMode_;
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasComputeType_t#cublascomputetype-t
  cublasComputeType_t computeType_;
  cudaDataType_t scaleType_;
  cudaDataType_t outputType_;
  int64_t smCount_;
  bool transA_;
  bool transB_;
  float alpha_;
  bool rowMajor_;

  void set(int M, int N, int K, int& lda, int& ldb, int& ldd) const;

  onnxruntime::Status CudaCompute(const int32_t* dtypes, cudaStream_t stream, cublasLtHandle_t handle,
                                  const Tensor* A, const Tensor* B, const Tensor* scaleA, Tensor* scaleB,
                                  int M, int N, int K) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
