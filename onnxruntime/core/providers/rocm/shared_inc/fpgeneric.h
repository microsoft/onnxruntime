// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/backward_guard.h"
#include "core/providers/rocm/rocm_common.h"

#define ORT_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
#if ORT_ROCBLAS_VERSION_DECIMAL >= 242
#define FLAG rocblas_gemm_flags_fp16_alt_impl
#else
#define FLAG 0
#endif
// needed to work around calling rocblas API instead of hipblas API
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op) {
  switch (op) {
    case HIPBLAS_OP_N:
      return rocblas_operation_none;
    case HIPBLAS_OP_T:
      return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
      return rocblas_operation_conjugate_transpose;
  }
  assert(0 && "HIPBLAS_STATUS_INVALID_ENUM");
}
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error) {
  switch (error) {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
      return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
      return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
      return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
      return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
      return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
      return HIPBLAS_STATUS_INTERNAL_ERROR;
    default:
      assert(0 && "ROCBLAS_STATUS_INVALID_ENUM");
      return HIPBLAS_STATUS_INTERNAL_ERROR;
  }
}

using namespace onnxruntime;

inline int get_flag() {
  int result = BackwardPassGuard::is_backward_pass() ? FLAG : 0;
  return result;
}

// Generalize library calls to be use in template functions

// hipblas

// gemm
inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const float* A, int lda,
                                         const float* B, int ldb,
                                         const float* beta,
                                         float* C, int ldc) {
  return hipblasGemmEx(handle,
                       transa,
                       transb,
                       m, n, k,
                       alpha,
                       A, HIP_R_32F, lda,
                       B, HIP_R_32F, ldb,
                       beta,
                       C, HIP_R_32F, ldc,
                       HIPBLAS_COMPUTE_32F,
                       HIPBLAS_GEMM_DEFAULT);
}

inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const double* alpha,
                                         const double* A, int lda,
                                         const double* B, int ldb,
                                         const double* beta,
                                         double* C, int ldc) {
  return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const half* alpha,
                                         const half* A, int lda,
                                         const half* B, int ldb,
                                         const half* beta,
                                         half* C, int ldc) {
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return rocBLASStatusToHIPStatus(rocblas_gemm_ex((rocblas_handle)handle,
                                                  hipOperationToRocOperation(transa),
                                                  hipOperationToRocOperation(transb),
                                                  m, n, k,
                                                  &h_a,
                                                  A, rocblas_datatype_f16_r, lda,
                                                  B, rocblas_datatype_f16_r, ldb,
                                                  &h_b,
                                                  C, rocblas_datatype_f16_r, ldc,
                                                  C, rocblas_datatype_f16_r, ldc,
                                                  rocblas_datatype_f32_r,
                                                  rocblas_gemm_algo_standard, 0, get_flag()));
}

inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const half* A, int lda,
                                         const half* B, int ldb,
                                         const float* beta,
                                         half* C, int ldc) {
  return rocBLASStatusToHIPStatus(rocblas_gemm_ex((rocblas_handle)handle,
                                                  hipOperationToRocOperation(transa),
                                                  hipOperationToRocOperation(transb),
                                                  m, n, k,
                                                  alpha,
                                                  A, rocblas_datatype_f16_r, lda,
                                                  B, rocblas_datatype_f16_r, ldb,
                                                  beta,
                                                  C, rocblas_datatype_f16_r, ldc,
                                                  C, rocblas_datatype_f16_r, ldc,
                                                  rocblas_datatype_f32_r,
                                                  rocblas_gemm_algo_standard, 0, get_flag()));
}

inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const float* alpha,
                                         const half* A, int lda,
                                         const half* B, int ldb,
                                         const float* beta,
                                         half* C, int ldc,
                                         const hipDeviceProp_t&,
                                         bool /*use_tf32*/) {
  return hipblasGemmHelper(handle,
                           transa,
                           transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
}

inline hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                         hipblasOperation_t transa,
                                         hipblasOperation_t transb,
                                         int m, int n, int k,
                                         const BFloat16* alpha,
                                         const BFloat16* A, int lda,
                                         const BFloat16* B, int ldb,
                                         const BFloat16* beta,
                                         BFloat16* C, int ldc) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return hipblasGemmEx(handle,
                       transa,
                       transb,
                       m, n, k,
                       &h_a,
                       A, HIP_R_16BF, lda,
                       B, HIP_R_16BF, ldb,
                       &h_b,
                       C, HIP_R_16BF, ldc,
                       HIPBLAS_COMPUTE_32F,
                       HIPBLAS_GEMM_DEFAULT);
}

// Compatible for function call with extra arguments (see cublasGemmHelper)
template <typename Scalar>
hipblasStatus_t hipblasGemmHelper(hipblasHandle_t handle,
                                  hipblasOperation_t transa,
                                  hipblasOperation_t transb,
                                  int m, int n, int k,
                                  const Scalar* alpha,
                                  const Scalar* A, int lda,
                                  const Scalar* B, int ldb,
                                  const Scalar* beta,
                                  Scalar* C, int ldc,
                                  const hipDeviceProp_t&,
                                  bool /*use_tf32*/) {
  return hipblasGemmHelper(handle,
                           transa,
                           transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
}

// batched gemm
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle,
                                                hipblasOperation_t transa,
                                                hipblasOperation_t transb,
                                                int m, int n, int k,
                                                const float* alpha,
                                                const float* Aarray[], int lda,
                                                const float* Barray[], int ldb,
                                                const float* beta,
                                                float* Carray[], int ldc,
                                                int batchCount) {
  return hipblasGemmBatchedEx(handle,
                              transa,
                              transb,
                              m, n, k,
                              alpha,
                              (const void**)Aarray, HIP_R_32F, lda,
                              (const void**)Barray, HIP_R_32F, ldb,
                              beta,
                              (void**)Carray, HIP_R_32F, ldc,
                              batchCount,
                              HIPBLAS_COMPUTE_32F,
                              HIPBLAS_GEMM_DEFAULT);
}
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle,
                                                hipblasOperation_t transa,
                                                hipblasOperation_t transb,
                                                int m, int n, int k,
                                                const double* alpha,
                                                const double* Aarray[], int lda,
                                                const double* Barray[], int ldb,
                                                const double* beta,
                                                double* Carray[], int ldc,
                                                int batchCount) {
  return hipblasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle,
                                                hipblasOperation_t transa,
                                                hipblasOperation_t transb,
                                                int m, int n, int k,
                                                const half* alpha,
                                                const half* Aarray[], int lda,
                                                const half* Barray[], int ldb,
                                                const half* beta,
                                                half* Carray[], int ldc,
                                                int batchCount) {
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return rocBLASStatusToHIPStatus(rocblas_gemm_batched_ex((rocblas_handle)handle,
                                                          hipOperationToRocOperation(transa),
                                                          hipOperationToRocOperation(transb),
                                                          m, n, k,
                                                          &h_a,
                                                          (const void**)Aarray, rocblas_datatype_f16_r, lda,
                                                          (const void**)Barray, rocblas_datatype_f16_r, ldb,
                                                          &h_b,
                                                          (void**)Carray, rocblas_datatype_f16_r, ldc,
                                                          (void**)Carray, rocblas_datatype_f16_r, ldc,
                                                          batchCount,
                                                          rocblas_datatype_f32_r,
                                                          rocblas_gemm_algo_standard, 0, get_flag()));
}

inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle,
                                                hipblasOperation_t transa,
                                                hipblasOperation_t transb,
                                                int m, int n, int k,
                                                const BFloat16* alpha,
                                                const BFloat16* Aarray[], int lda,
                                                const BFloat16* Barray[], int ldb,
                                                const BFloat16* beta,
                                                BFloat16* Carray[], int ldc,
                                                int batch_count) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return hipblasGemmBatchedEx(handle,
                              transa,
                              transb,
                              m, n, k,
                              &h_a,
                              (const void**)Aarray, HIP_R_16BF, lda,
                              (const void**)Barray, HIP_R_16BF, ldb,
                              &h_b,
                              (void**)Carray, HIP_R_16BF, ldc,
                              batch_count,
                              HIPBLAS_COMPUTE_32F,
                              HIPBLAS_GEMM_DEFAULT);
}

// strided batched gemm
inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
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
  return hipblasGemmStridedBatchedEx(handle,
                                     transa,
                                     transb,
                                     m, n, k,
                                     alpha,
                                     A, HIP_R_32F, lda, strideA,
                                     B, HIP_R_32F, ldb, strideB,
                                     beta,
                                     C, HIP_R_32F, ldc, strideC,
                                     batchCount,
                                     HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT);
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
                                                       int m, int n, int k,
                                                       const double* alpha,
                                                       const double* A, int lda,
                                                       long long int strideA,
                                                       const double* B, int ldb,
                                                       long long int strideB,
                                                       const double* beta,
                                                       double* C, int ldc,
                                                       long long int strideC,
                                                       int batchCount) {
  return hipblasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
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
  return rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                                                  hipOperationToRocOperation(transa),
                                                                  hipOperationToRocOperation(transb),
                                                                  m, n, k,
                                                                  &h_a,
                                                                  A, rocblas_datatype_f16_r, lda, strideA,
                                                                  B, rocblas_datatype_f16_r, ldb, strideB,
                                                                  &h_b,
                                                                  C, rocblas_datatype_f16_r, ldc, strideC,
                                                                  C, rocblas_datatype_f16_r, ldc, strideC,
                                                                  batchCount,
                                                                  rocblas_datatype_f32_r,
                                                                  rocblas_gemm_algo_standard, 0, get_flag()));
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
                                                       int m, int n, int k,
                                                       const float* alpha,
                                                       const __half* A, int lda,
                                                       intmax_t strideA,
                                                       const __half* B, int ldb,
                                                       intmax_t strideB,
                                                       const float* beta,
                                                       __half* C, int ldc,
                                                       intmax_t strideC,
                                                       int batchCount) {
  return rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                                                  hipOperationToRocOperation(transa),
                                                                  hipOperationToRocOperation(transb),
                                                                  m, n, k,
                                                                  alpha,
                                                                  A, rocblas_datatype_f16_r, lda, strideA,
                                                                  B, rocblas_datatype_f16_r, ldb, strideB,
                                                                  beta,
                                                                  C, rocblas_datatype_f16_r, ldc, strideC,
                                                                  C, rocblas_datatype_f16_r, ldc, strideC,
                                                                  batchCount,
                                                                  rocblas_datatype_f32_r,
                                                                  rocblas_gemm_algo_standard, 0, get_flag()));
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
                                                       int m, int n, int k,
                                                       const BFloat16* alpha,
                                                       const BFloat16* A, int lda,
                                                       intmax_t strideA,
                                                       const BFloat16* B, int ldb,
                                                       intmax_t strideB,
                                                       const BFloat16* beta,
                                                       BFloat16* C, int ldc,
                                                       intmax_t strideC,
                                                       int batch_count) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();
  // accumulating in FP32
  return hipblasGemmStridedBatchedEx(handle,
                                     transa,
                                     transb,
                                     m, n, k,
                                     &h_a,
                                     A, HIP_R_16BF, lda, strideA,
                                     B, HIP_R_16BF, ldb, strideB,
                                     &h_b,
                                     C, HIP_R_16BF, ldc, strideC,
                                     batch_count,
                                     HIPBLAS_COMPUTE_32F,
                                     HIPBLAS_GEMM_DEFAULT);
}

// Compatible for function call with with extra arguments (see cublasGemmStridedBatchedHelper)
template <typename Scalar>
hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                hipblasOperation_t transa,
                                                hipblasOperation_t transb,
                                                int m, int n, int k,
                                                const Scalar* alpha,
                                                const Scalar* A, int lda,
                                                intmax_t strideA,
                                                const Scalar* B, int ldb,
                                                intmax_t strideB,
                                                const Scalar* beta,
                                                Scalar* C, int ldc,
                                                intmax_t strideC,
                                                int batchCount,
                                                const hipDeviceProp_t&,
                                                bool /*use_tf32*/) {
  return hipblasGemmStridedBatchedHelper(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
}

inline hipblasStatus_t hipblasGemmStridedBatchedHelper(hipblasHandle_t handle,
                                                       hipblasOperation_t transa,
                                                       hipblasOperation_t transb,
                                                       int m, int n, int k,
                                                       const float* alpha,
                                                       const __half* A, int lda,
                                                       intmax_t strideA,
                                                       const __half* B, int ldb,
                                                       intmax_t strideB,
                                                       const float* beta,
                                                       __half* C, int ldc,
                                                       intmax_t strideC,
                                                       int batchCount,
                                                       const hipDeviceProp_t&,
                                                       bool /*use_tf32*/) {
  return hipblasGemmStridedBatchedHelper(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
}

// transpose using geam
inline hipblasStatus_t hipblasTransposeHelper(hipStream_t /*stream*/, hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return hipblasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline hipblasStatus_t hipblasTransposeHelper(hipStream_t /*stream*/, hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return hipblasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline bool CanUse_hipblasTransposeHelper_MLFloat16(int /*m*/, int /*n*/) { return true; }  // CUDA has a limited grid size of 65536, ROCm has higher limits.
hipblasStatus_t hipblasTransposeHelper(hipStream_t stream, hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// copy
inline hipblasStatus_t hipblasCopyHelper(hipStream_t /*stream*/, hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return hipblasScopy(handle, n, x, incx, y, incy);
}
inline hipblasStatus_t hipblasCopyHelper(hipStream_t /*stream*/, hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return hipblasDcopy(handle, n, x, incx, y, incy);
}
hipblasStatus_t hipblasCopyHelper(hipStream_t stream, hipblasHandle_t handle, int n, const half* x, int incx, half* y, int incy);
hipblasStatus_t hipblasCopyHelper(hipStream_t stream, hipblasHandle_t handle, int n, const BFloat16* x, int incx, BFloat16* y, int incy);

// rocblas

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
                         rocblas_gemm_algo_standard, 0, get_flag());
}

inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                        rocblas_operation transa,
                                        rocblas_operation transb,
                                        int m, int n, int k,
                                        const float* alpha,
                                        const half* A, int lda,
                                        const half* B, int ldb,
                                        const float* beta,
                                        half* C, int ldc) {
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         alpha,
                         A, rocblas_datatype_f16_r, lda,
                         B, rocblas_datatype_f16_r, ldb,
                         beta,
                         C, rocblas_datatype_f16_r, ldc,
                         C, rocblas_datatype_f16_r, ldc,
                         rocblas_datatype_f32_r,
                         rocblas_gemm_algo_standard, 0, get_flag());
}

inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                        rocblas_operation transa,
                                        rocblas_operation transb,
                                        int m, int n, int k,
                                        const float* alpha,
                                        const half* A, int lda,
                                        const half* B, int ldb,
                                        const float* beta,
                                        half* C, int ldc,
                                        const hipDeviceProp_t&,
                                        bool /*use_tf32*/) {
  return rocblasGemmHelper(handle,
                           transa,
                           transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
}

inline rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                        rocblas_operation transa,
                                        rocblas_operation transb,
                                        int m, int n, int k,
                                        const BFloat16* alpha,
                                        const BFloat16* A, int lda,
                                        const BFloat16* B, int ldb,
                                        const BFloat16* beta,
                                        BFloat16* C, int ldc) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return rocblas_gemm_ex(handle,
                         transa,
                         transb,
                         m, n, k,
                         &h_a,
                         A, rocblas_datatype_bf16_r, lda,
                         B, rocblas_datatype_bf16_r, ldb,
                         &h_b,
                         C, rocblas_datatype_bf16_r, ldc,
                         C, rocblas_datatype_bf16_r, ldc,
                         rocblas_datatype_f32_r,
                         rocblas_gemm_algo_standard, 0, 0);
}

// Compatible for function call with extra arguments (see cublasGemmHelper)
template <typename Scalar>
rocblas_status rocblasGemmHelper(rocblas_handle handle,
                                 rocblas_operation transa,
                                 rocblas_operation transb,
                                 int m, int n, int k,
                                 const Scalar* alpha,
                                 const Scalar* A, int lda,
                                 const Scalar* B, int ldb,
                                 const Scalar* beta,
                                 Scalar* C, int ldc,
                                 const hipDeviceProp_t&,
                                 bool /*use_tf32*/) {
  return rocblasGemmHelper(handle,
                           transa,
                           transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
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
                                 rocblas_gemm_algo_standard, 0, get_flag());
}

inline rocblas_status rocblasGemmBatchedHelper(rocblas_handle handle,
                                               rocblas_operation transa,
                                               rocblas_operation transb,
                                               int m, int n, int k,
                                               const BFloat16* alpha,
                                               const BFloat16* Aarray[], int lda,
                                               const BFloat16* Barray[], int ldb,
                                               const BFloat16* beta,
                                               BFloat16* Carray[], int ldc,
                                               int batch_count) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();

  // accumulating in FP32
  return rocblas_gemm_batched_ex(handle,
                                 transa,
                                 transb,
                                 m, n, k,
                                 &h_a,
                                 (const void**)Aarray, rocblas_datatype_bf16_r, lda,
                                 (const void**)Barray, rocblas_datatype_bf16_r, ldb,
                                 &h_b,
                                 (void**)Carray, rocblas_datatype_bf16_r, ldc,
                                 (void**)Carray, rocblas_datatype_bf16_r, ldc,
                                 batch_count,
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
                                                      int batchCount) {
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
                                         rocblas_gemm_algo_standard, 0, get_flag());
}

inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                      rocblas_operation transa,
                                                      rocblas_operation transb,
                                                      int m, int n, int k,
                                                      const float* alpha,
                                                      const __half* A, int lda,
                                                      intmax_t strideA,
                                                      const __half* B, int ldb,
                                                      intmax_t strideB,
                                                      const float* beta,
                                                      __half* C, int ldc,
                                                      intmax_t strideC,
                                                      int batchCount) {
  return rocblas_gemm_strided_batched_ex(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, rocblas_datatype_f16_r, lda, strideA,
                                         B, rocblas_datatype_f16_r, ldb, strideB,
                                         beta,
                                         C, rocblas_datatype_f16_r, ldc, strideC,
                                         C, rocblas_datatype_f16_r, ldc, strideC,
                                         batchCount,
                                         rocblas_datatype_f32_r,
                                         rocblas_gemm_algo_standard, 0, get_flag());
}

inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                      rocblas_operation transa,
                                                      rocblas_operation transb,
                                                      int m, int n, int k,
                                                      const BFloat16* alpha,
                                                      const BFloat16* A, int lda,
                                                      intmax_t strideA,
                                                      const BFloat16* B, int ldb,
                                                      intmax_t strideB,
                                                      const BFloat16* beta,
                                                      BFloat16* C, int ldc,
                                                      intmax_t strideC,
                                                      int batch_count) {
  float h_a = alpha->ToFloat();
  float h_b = beta->ToFloat();
  // accumulating in FP32
  return rocblas_gemm_strided_batched_ex(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         &h_a,
                                         A, rocblas_datatype_bf16_r, lda, strideA,
                                         B, rocblas_datatype_bf16_r, ldb, strideB,
                                         &h_b,
                                         C, rocblas_datatype_bf16_r, ldc, strideC,
                                         C, rocblas_datatype_bf16_r, ldc, strideC,
                                         batch_count,
                                         rocblas_datatype_f32_r,
                                         rocblas_gemm_algo_standard, 0, 0);
}

// Compatible for function call with with extra arguments (see cublasGemmStridedBatchedHelper)
template <typename Scalar>
rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                               rocblas_operation transa,
                                               rocblas_operation transb,
                                               int m, int n, int k,
                                               const Scalar* alpha,
                                               const Scalar* A, int lda,
                                               intmax_t strideA,
                                               const Scalar* B, int ldb,
                                               intmax_t strideB,
                                               const Scalar* beta,
                                               Scalar* C, int ldc,
                                               intmax_t strideC,
                                               int batchCount,
                                               const hipDeviceProp_t&,
                                               bool /*use_tf32*/) {
  return rocblasGemmStridedBatchedHelper(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
}

inline rocblas_status rocblasGemmStridedBatchedHelper(rocblas_handle handle,
                                                      rocblas_operation transa,
                                                      rocblas_operation transb,
                                                      int m, int n, int k,
                                                      const float* alpha,
                                                      const __half* A, int lda,
                                                      intmax_t strideA,
                                                      const __half* B, int ldb,
                                                      intmax_t strideB,
                                                      const float* beta,
                                                      __half* C, int ldc,
                                                      intmax_t strideC,
                                                      int batchCount,
                                                      const hipDeviceProp_t&,
                                                      bool /*use_tf32*/) {
  return rocblasGemmStridedBatchedHelper(handle,
                                         transa,
                                         transb,
                                         m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
}
bool CanUse_hipblasTransposeHelper_MLFloat16(int m, int n);
hipblasStatus_t hipblasTransposeHelper(hipStream_t stream, rocblas_handle, rocblas_operation, rocblas_operation, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);
