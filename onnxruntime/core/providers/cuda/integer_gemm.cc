// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/integer_gemm.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {

inline int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

Status GemmInt8(int m, int n, int k,
                int32_t alpha, int32_t beta,
                const int8_t* a, int lda, const int8_t* b, int ldb, int32_t* c, int ldc,
                const CudaKernel* cuda_kernel) {
  ORT_ENFORCE(a != nullptr && b != nullptr && c != nullptr, "input matrix should not be null");
  ORT_ENFORCE(cuda_kernel != nullptr, "kernel is null");

  cudaStream_t stream = cuda_kernel->Stream();

  // pad A and B to make their leading dimension be multiples of 32
  // because cublasGemmEx requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned

  const int mask = 0x1F;
  int lda_aligned = lda;
  IAllocatorUniquePtr<int8_t> a_padded;
  if ((mask & lda_aligned) != 0) {
    lda_aligned = roundoff(lda, 32);
    a_padded = cuda_kernel->GetScratchBuffer<int8_t>(m * lda_aligned);
    cudaMemcpy2DAsync(a_padded.get(), lda_aligned, a, lda, k, m, cudaMemcpyDeviceToDevice, stream);
  }

  int ldb_aligned = ldb;
  IAllocatorUniquePtr<int8_t> b_padded;
  if ((mask & ldb_aligned) != 0) {
    ldb_aligned = roundoff(ldb, 32);
    b_padded = cuda_kernel->GetScratchBuffer<int8_t>(k * ldb_aligned);
    cudaMemcpy2DAsync(b_padded.get(), ldb_aligned, b, ldb, n, k, cudaMemcpyDeviceToDevice, stream);
  }

  cublasHandle_t cublas = cuda_kernel->CublasHandle();
  cublasSetStream(cublas, stream);
  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
      cublas,
      CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      ldb_aligned == ldb ? b : b_padded.get(), CUDA_R_8I, ldb_aligned,
      lda_aligned == lda ? a : a_padded.get(), CUDA_R_8I, lda_aligned,
      &beta,
      c, CUDA_R_32I, ldc, CUDA_R_32I,
      CUBLAS_GEMM_DFALT));
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
