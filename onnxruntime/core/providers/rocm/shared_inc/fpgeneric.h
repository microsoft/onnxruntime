// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"

// Generalize library calls to be use in template functions

// gemm
inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                        rocblas_operation transa,
                                        rocblas_operation transb,
                                        int m, int n, int k,
                                        const float* alpha,
                                        const float* A, int lda,
                                        const float* B, int ldb,
                                        const float* beta,
                                        float* C, int ldc) {
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         alpha,
                         A, rocblas_datatype_f32_r, lda,
                         B, rocblas_datatype_f32_r, ldb,
                         beta,
                         C, rocblas_datatype_f32_r, ldc,
                         C, rocblas_datatype_f32_r, ldc,
                         rocblas_datatype_f32_r,
                         rocblas_gemm_algo_standard, 0, 0);
}
inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                         rocblas_operation transa,
                                         rocblas_operation transb,
                                         int m, int n, int k,
                                         const double* alpha,
                                         const double* A, int lda,
                                         const double* B, int ldb,
                                         const double* beta,
                                         double* C, int ldc) {
  return rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                         rocblas_operation transa,
                                         rocblas_operation transb,
                                         int m, int n, int k,
                                         const half* alpha,
                                         const half* A, int lda,
                                         const half* B, int ldb,
                                         const half* beta,
                                         half* C, int ldc) {
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         &h_a,
                         A, rocblas_datatype_f16_r, lda,
                         B, rocblas_datatype_f16_r, ldb,
                         &h_b,
                         C, rocblas_datatype_f16_r, ldc,
                         C, rocblas_datatype_f16_r, ldc,
                         rocblas_datatype_f32_r,
                         rocblas_gemm_algo_standard, 0, 0);
}

// batched gemm
inline rocblas_status rocblasGemmBatchedHelper(rocblas_handle handle,
                                                rocblas_operation transa,
                                                rocblas_operation transb,
                                                int m, int n, int k,
                                                const float* alpha,
                                                const float* Aarray[], int lda,
                                                const float* Barray[], int ldb,
                                                const float* beta,
                                                float* Carray[], int ldc,
                                                int batchCount) {
  return rocblas_gemm_batched_ex(handle,
                                 transa,
                                 transb,
                                 m, n, k,
                                 alpha,
                                 (const void**)Aarray, rocblas_datatype_f32_r, lda,
                                 (const void**)Barray, rocblas_datatype_f32_r, ldb,
                                 beta,
                                 (void**)Carray, rocblas_datatype_f32_r, ldc,
                                 (void**)Carray, rocblas_datatype_f32_r, ldc,
                                 batchCount,
                                 rocblas_datatype_f32_r,
                                 rocblas_gemm_algo_standard, 0, 0);
}
inline rocblas_status rocblasGemmBatchedHelper(rocblas_handle handle,
                                                rocblas_operation transa,
                                                rocblas_operation transb,
                                                int m, int n, int k,
                                                const double* alpha,
                                                const double* Aarray[], int lda,
                                                const double* Barray[], int ldb,
                                                const double* beta,
                                                double* Carray[], int ldc,
                                                int batchCount) {
  return rocblas_dgemm_batched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline rocblas_status rocblasGemmBatchedHelper(rocblas_handle handle,
                                                rocblas_operation transa,
                                                rocblas_operation transb,
                                                int m, int n, int k,
                                                const half* alpha,
                                                const half* Aarray[], int lda,
                                                const half* Barray[], int ldb,
                                                const half* beta,
                                                half* Carray[], int ldc,
                                                int batchCount) {
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return rocblas_gemm_batched_ex(handle,
                                 transa,
                                 transb,
                                 m, n, k,
                                 &h_a,
                                 (const void**)Aarray, rocblas_datatype_f16_r, lda,
                                 (const void**)Barray, rocblas_datatype_f16_r, ldb,
                                 &h_b,
                                 (void**)Carray, rocblas_datatype_f16_r, ldc,
                                 (void**)Carray, rocblas_datatype_f16_r, ldc,
                                 batchCount,
                                 rocblas_datatype_f32_r,
                                 rocblas_gemm_algo_standard, 0, 0);
}

// strided batched gemm
inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                       rocblas_operation transa,
                                                       rocblas_operation transb,
                                                       int m, int n, int k,
                                                       const float* alpha,
                                                       const float* A, int lda,
                                                       long long int strideA,
                                                       const float* B, int ldb,
                                                       long long int strideB,
                                                       const float* beta,
                                                       float* C, int ldc,
                                                       long long int strideC,
                                                       int batchCount) {
  return rocblas_gemm_strided_batched_ex(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, rocblas_datatype_f32_r, lda, strideA,
                                         B, rocblas_datatype_f32_r, ldb, strideB,
                                         beta,
                                         C, rocblas_datatype_f32_r, ldc, strideC,
                                         C, rocblas_datatype_f32_r, ldc, strideC,
                                         batchCount,
                                         rocblas_datatype_f32_r,
                                         rocblas_gemm_algo_standard, 0, 0);
}

inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                       rocblas_operation transa,
                                                       rocblas_operation transb,
                                                       int m, int n, int k,
                                                       const double* alpha,
                                                       const double* A, int lda,
                                                       long long int strideA,
                                                       const double* B, int ldb,
                                                       long long int strideB,
                                                       const double* beta,
                                                       double* C, int ldc,
                                                       long long int strideC,
                                                       int batchCount){
  return rocblas_dgemm_strided_batched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                       rocblas_operation transa,
                                                       rocblas_operation transb,
                                                       int m, int n, int k,
                                                       const __half* alpha,
                                                       const __half* A, int lda,
                                                       long long int strideA,
                                                       const __half* B, int ldb,
                                                       long long int strideB,
                                                       const __half* beta,
                                                       __half* C, int ldc,
                                                       long long int strideC,
                                                       int batchCount) {
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return rocblas_gemm_strided_batched_ex(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         &h_a,
                                         A, rocblas_datatype_f16_r, lda, strideA,
                                         B, rocblas_datatype_f16_r, ldb, strideB,
                                         &h_b,
                                         C, rocblas_datatype_f16_r, ldc, strideC,
                                         C, rocblas_datatype_f16_r, ldc, strideC,
                                         batchCount,
                                         rocblas_datatype_f32_r,
                                         rocblas_gemm_algo_standard, 0, 0);
}

// transpose using geam
inline rocblas_status rocblasTransposeHelper(hipStream_t /*stream*/, rocblas_handle handle, rocblas_operation  transa, rocblas_operation  transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return rocblas_sgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline rocblas_status rocblasTransposeHelper(hipStream_t /*stream*/, rocblas_handle handle, rocblas_operation  transa, rocblas_operation  transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return rocblas_dgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
rocblas_status rocblasTransposeHelper(hipStream_t stream, rocblas_handle, rocblas_operation , rocblas_operation , int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// copy
inline rocblas_status rocblasCopyHelper(hipStream_t /*stream*/, rocblas_handle handle, int n, const float* x, int incx, float* y, int incy) {
  return rocblas_scopy(handle, n, x, incx, y, incy);
}
inline rocblas_status rocblasCopyHelper(hipStream_t /*stream*/, rocblas_handle handle, int n, const double* x, int incx, double* y, int incy) {
  return rocblas_dcopy(handle, n, x, incx, y, incy);
}
rocblas_status rocblasCopyHelper(hipStream_t stream, rocblas_handle handle, int n, const half* x, int incx, half* y, int incy);
