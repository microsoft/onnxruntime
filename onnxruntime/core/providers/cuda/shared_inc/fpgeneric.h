//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half

#pragma once

#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;

// Generalize library calls to be use in template functions
inline cublasStatus_t
cublasGemmHelper(cublasHandle_t handle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m, int n, int k,
                 const float* alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 const float* beta,
                 float* C, int ldc,
                 const cudaDeviceProp& prop,
                 bool use_tf32) {
#if defined(USE_CUDA)
  // To disable TF32, set environment variable NVIDIA_TF32_OVERRIDE = 0 or set provider option use_tf32 = 0
  cublasMath_t mode = use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, mode);
#else
  ORT_UNUSED_PARAMETER(prop);
  ORT_UNUSED_PARAMETER(use_tf32);
#endif

  return cublasSgemm(handle,
                     transa,
                     transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const double* alpha,
                                       const double* A, int lda,
                                       const double* B, int ldb,
                                       const double* beta,
                                       double* C, int ldc,
                                       const cudaDeviceProp& /*prop*/,
                                       bool /*use_tf32*/) {
  return cublasDgemm(handle,
                     transa,
                     transb,
                     m, n, k,
                     alpha,
                     A, lda,
                     B, ldb,
                     beta,
                     C, ldc);
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const half* alpha,
                                       const half* A, int lda,
                                       const half* B, int ldb,
                                       const half* beta,
                                       half* C, int ldc,
                                       const cudaDeviceProp& prop,
                                       bool /*use_tf32*/) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        alpha,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        beta,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        &h_a,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        &h_b,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m, int n, int k,
                                       const float* alpha,
                                       const half* A, int lda,
                                       const half* B, int ldb,
                                       const float* beta,
                                       half* C, int ldc,
                                       const cudaDeviceProp& prop,
                                       bool /*use_tf32*/) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = onnxruntime::math::floatToHalf(*alpha);
    uint16_t h_b = onnxruntime::math::floatToHalf(*beta);
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        &h_a,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        &h_b,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m, n, k,
                        alpha,
                        A, CUDA_R_16F, lda,
                        B, CUDA_R_16F, ldb,
                        beta,
                        C, CUDA_R_16F, ldc,
                        half_options->GetComputeType(),
                        CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(USE_CUDA)
inline cublasStatus_t cublasGemmHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
    int n, int k, const BFloat16* alpha, const BFloat16* A, int lda,
    const BFloat16* B, int ldb, const BFloat16* beta, BFloat16* C, int ldc,
    const cudaDeviceProp& /*prop*/, bool /*use_tf32*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16BF, lda, B, CUDA_R_16BF, ldb, &h_b, C,
                      CUDA_R_16BF, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                       const BFloat16*, const BFloat16*, int, const BFloat16*, int, const BFloat16*,
                                       BFloat16*, int, const cudaDeviceProp&, bool /*use_tf32*/) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const float* alpha,
                                              const float* Aarray[], int lda,
                                              const float* Barray[], int ldb,
                                              const float* beta,
                                              float* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop,
                                              bool use_tf32) {
// The caller shall check memory alignments of the matrices when use_tf32 is true.
#if defined(USE_CUDA)
  cublasMath_t mode = use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, mode);
#else
  ORT_UNUSED_PARAMETER(prop);
  ORT_UNUSED_PARAMETER(use_tf32);
#endif

  return cublasSgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const double* alpha,
                                              const double* Aarray[], int lda,
                                              const double* Barray[], int ldb,
                                              const double* beta,
                                              double* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& /*prop*/,
                                              bool /*use_tf32*/) {
  return cublasDgemmBatched(handle,
                            transa,
                            transb,
                            m, n, k,
                            alpha,
                            Aarray, lda,
                            Barray, ldb,
                            beta,
                            Carray, ldc,
                            batch_count);
}

inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m, int n, int k,
                                              const half* alpha,
                                              const half* Aarray[], int lda,
                                              const half* Barray[], int ldb,
                                              const half* beta,
                                              half* Carray[], int ldc,
                                              int batch_count,
                                              const cudaDeviceProp& prop,
                                              bool /*use_tf32*/) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m, n, k,
                               alpha,
                               (const void**)Aarray, CUDA_R_16F, lda,
                               (const void**)Barray, CUDA_R_16F, ldb,
                               beta,
                               (void**)Carray, CUDA_R_16F, ldc,
                               batch_count,
                               half_options->GetComputeType(),
                               CUBLAS_GEMM_DEFAULT);
  } else {
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m, n, k,
                               &h_a,
                               (const void**)Aarray, CUDA_R_16F, lda,
                               (const void**)Barray, CUDA_R_16F, ldb,
                               &h_b,
                               (void**)Carray, CUDA_R_16F, ldc,
                               batch_count,
                               half_options->GetComputeType(),
                               CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(USE_CUDA)
inline cublasStatus_t cublasGemmBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const BFloat16* alpha, const BFloat16* Aarray[],
    int lda, const BFloat16* Barray[], int ldb, const BFloat16* beta,
    BFloat16* Carray[], int ldc, int batch_count,
    const cudaDeviceProp& /*prop*/, bool /*use_tf32*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return cublasGemmBatchedEx(handle, transa, transb, m, n, k, &h_a, (const void**)Aarray, CUDA_R_16BF, lda,
                             (const void**)Barray, CUDA_R_16BF, ldb, &h_b, (void**)Carray, CUDA_R_16BF, ldc,
                             batch_count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                              const BFloat16*, const BFloat16*[], int, const BFloat16*[], int,
                                              const BFloat16*, BFloat16*[], int, int, const cudaDeviceProp&,
                                              bool /*use_tf32*/) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// strided batched gemm
inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const float* A, int lda,
                                                     long long int strideA,
                                                     const float* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     float* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop,
                                                     bool use_tf32) {
#if defined(USE_CUDA)
  cublasMath_t mode = use_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, mode);
#else
  ORT_UNUSED_PARAMETER(prop);
  ORT_UNUSED_PARAMETER(use_tf32);
#endif

  return cublasSgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, lda, strideA,
                                   B, ldb, strideB,
                                   beta,
                                   C, ldc, strideC,
                                   batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const double* alpha,
                                                     const double* A, int lda,
                                                     long long int strideA,
                                                     const double* B, int ldb,
                                                     long long int strideB,
                                                     const double* beta,
                                                     double* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& /*prop*/,
                                                     bool /*use_tf32*/) {
  return cublasDgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m, n, k,
                                   alpha,
                                   A, lda, strideA,
                                   B, ldb, strideB,
                                   beta,
                                   C, ldc, strideC,
                                   batch_count);
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const __half* alpha,
                                                     const __half* A, int lda,
                                                     long long int strideA,
                                                     const __half* B, int ldb,
                                                     long long int strideB,
                                                     const __half* beta,
                                                     __half* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop,
                                                     bool /*use_tf32*/) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      alpha,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      beta,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  } else {
    // The alpha and beta shall have same precision as compute type.
    float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
    float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      &h_a,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      &h_b,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  }
}

inline cublasStatus_t cublasGemmStridedBatchedHelper(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const __half* A, int lda,
                                                     long long int strideA,
                                                     const __half* B, int ldb,
                                                     long long int strideB,
                                                     const float* beta,
                                                     __half* C, int ldc,
                                                     long long int strideC,
                                                     int batch_count,
                                                     const cudaDeviceProp& prop,
                                                     bool /*use_tf32*/) {
  const HalfGemmOptions* half_options = HalfGemmOptions::GetInstance();
  onnxruntime::cuda::CublasMathModeSetter math_mode_setter(prop, handle, half_options->GetMathMode());
  if (half_options->IsCompute16F()) {
    // The alpha and beta shall have same precision as compute type.
    uint16_t h_a = onnxruntime::math::floatToHalf(*alpha);
    uint16_t h_b = onnxruntime::math::floatToHalf(*beta);
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      &h_a,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      &h_b,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  } else {
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m, n, k,
                                      alpha,
                                      A, CUDA_R_16F, lda, strideA,
                                      B, CUDA_R_16F, ldb, strideB,
                                      beta,
                                      C, CUDA_R_16F, ldc, strideC,
                                      batch_count,
                                      half_options->GetComputeType(),
                                      CUBLAS_GEMM_DEFAULT);
  }
}

#if defined(USE_CUDA)
inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n, int k,
    const BFloat16* alpha, const BFloat16* A, int lda,
    long long int strideA, const BFloat16* B, int ldb,
    long long int strideB, const BFloat16* beta, BFloat16* C, int ldc,
    long long int strideC, int batch_count,
    const cudaDeviceProp& /*prop*/, bool /*use_tf32*/) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();
  // accumulating in FP32
  return cublasGemmStridedBatchedEx(
      handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16BF, lda, strideA, B, CUDA_R_16BF,
      ldb, strideB, &h_b, C, CUDA_R_16BF, ldc, strideC, batch_count, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}
#else
inline cublasStatus_t cublasGemmStridedBatchedHelper(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
    int, const BFloat16*, const BFloat16*, int, long long int,
    const BFloat16*, int, long long int, const BFloat16*, BFloat16*,
    int, long long int, int, const cudaDeviceProp&, bool /*use_tf32*/) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
#endif

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(
    cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline cublasStatus_t cublasTransposeHelper(
    cudaStream_t, cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

bool CanUse_cublasTransposeHelper_MLFloat16(int m, int n);

cublasStatus_t cublasTransposeHelper(
    cudaStream_t, cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// copy
inline cublasStatus_t cublasCopyHelper(
    cudaStream_t, cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}

inline cublasStatus_t cublasCopyHelper(
    cudaStream_t, cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCopyHelper(
    cudaStream_t stream, cublasHandle_t handle, int n, const half* x, int incx, half* y, int incy);

cublasStatus_t cublasCopyHelper(
    cudaStream_t stream, cublasHandle_t handle, int n, const BFloat16* x, int incx, BFloat16* y, int incy);
