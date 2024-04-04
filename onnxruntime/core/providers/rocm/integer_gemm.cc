// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "core/providers/rocm/shared_inc/integer_gemm.h"

#include "core/common/safeint.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {

constexpr int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

Status GemmInt8(int m, int n, int k,
                int32_t alpha, int32_t beta,
                const int8_t* a, int lda, const int8_t* b, int ldb, int32_t* c, int ldc,
                const RocmKernel* rocm_kernel, onnxruntime::Stream* ort_stream) {
  ORT_ENFORCE(a != nullptr && b != nullptr && c != nullptr, "input matrix should not be null");
  ORT_ENFORCE(rocm_kernel != nullptr, "kernel is null");
  ORT_ENFORCE(ort_stream != nullptr, "Rocm kernel must have the stream instance");

  hipStream_t stream = static_cast<hipStream_t>(ort_stream->GetHandle());

  // pad A and B to make their leading dimension be multiples of 32
  // because rocblas_gemm_ex requires:
  // 1. leading dimension is multiples of 4
  // 2. A, B is 32-bit aligned

  constexpr int mask = 0x1F;
  int lda_aligned = lda;
  IAllocatorUniquePtr<int8_t> a_padded;
  if ((mask & lda_aligned) != 0) {
    lda_aligned = roundoff(lda, 32);
    a_padded = rocm_kernel->GetScratchBuffer<int8_t>(SafeInt<size_t>(m) * lda_aligned, ort_stream);
    HIP_RETURN_IF_ERROR(hipMemcpy2DAsync(a_padded.get(), lda_aligned, a, lda, k, m, hipMemcpyDeviceToDevice, stream));
  }

  int ldb_aligned = ldb;
  IAllocatorUniquePtr<int8_t> b_padded;
  if ((mask & ldb_aligned) != 0) {
    ldb_aligned = roundoff(ldb, 32);
    b_padded = rocm_kernel->GetScratchBuffer<int8_t>(SafeInt<size_t>(k) * ldb_aligned, ort_stream);
    HIP_RETURN_IF_ERROR(hipMemcpy2DAsync(b_padded.get(), ldb_aligned, b, ldb, n, k, hipMemcpyDeviceToDevice, stream));
  }

  auto* ort_rocm_stream = dynamic_cast<RocmStream*>(ort_stream);
  auto rocblas = ort_rocm_stream->rocblas_handle_;

  ROCBLAS_RETURN_IF_ERROR(rocblas_gemm_ex(
      rocblas,
      rocblas_operation_none, rocblas_operation_none,
      n, m, k,
      &alpha,
      ldb_aligned == ldb ? b : b_padded.get(), rocblas_datatype_i8_r, ldb_aligned,
      lda_aligned == lda ? a : a_padded.get(), rocblas_datatype_i8_r, lda_aligned,
      &beta,
      c, rocblas_datatype_i32_r, ldc,
      c, rocblas_datatype_i32_r, ldc,  // C == D
      rocblas_datatype_i32_r,
      rocblas_gemm_algo_standard,
      0, 0));
  return Status::OK();
}
}  // namespace rocm
}  // namespace onnxruntime
