// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/integer_gemm.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace cuda {

inline int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

__global__ void PadMatrixKernel(int8_t* dst, int col_dst, const int8_t* src, int col_src) {
  for (int32_t i = threadIdx.x; i < col_src; i += blockDim.x) {
    *(dst + blockIdx.x * col_dst + i) = *(src + blockIdx.x * col_src + i);
  }
}

void PadMatrix(int8_t* dst, int pitch, const int8_t* src, int row, int col) {
  PadMatrixKernel<<<row, GridDim::maxThreadsPerBlock, 0>>>(
      dst,
      pitch,
      src,
      col);
}

Status GemmInt8(int m, int n, int k,
              int32_t alpha, int32_t beta,
              const int8_t* a, int lda, const int8_t* b, int ldb, int32_t* c, int ldc,
              const CudaKernel* cuda_kernel) {
  ORT_ENFORCE(a != nullptr && b != nullptr && c != nullptr, "input matrix should not be null");
  ORT_ENFORCE(cuda_kernel != nullptr, "kernel is null");

  // pad A and B to make their leading dimension be multiples of 32
  // because cublasGemmEx requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned

  const int64_t mask = 0x1F;
  int64_t lda_aligned = lda;
  IAllocatorUniquePtr<int8_t> a_padded;
  if (mask & lda_aligned != 0) {
    lda_aligned = roundoff(lda, 32);
    a_padded = cuda_kernel->GetScratchBuffer<int8_t>(m * lda_aligned);
    PadMatrix(a_padded.get(), lda_aligned, a, m, lda);
  }

  int64_t ldb_aligned = ldb;
  IAllocatorUniquePtr<int8_t> b_padded;
  if (mask & ldb_aligned) {
    ldb_aligned = roundoff(ldb, 32);
    b_padded = cuda_kernel->GetScratchBuffer<int8_t>(k * ldb_aligned);
    PadMatrix(b_padded.get(), ldb_aligned, b, k, ldb);
  }

  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
      cuda_kernel->CublasHandle(),
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
