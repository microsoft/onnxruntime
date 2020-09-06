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

#include "core/providers/hip/hip_common.h"

// Generalize library calls to be use in template functions

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
  //return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return hipblasGemmEx(handle,
                       transa,
                       transb,
                       m, n, k,
                       alpha,
                       A, HIPBLAS_R_32F, lda,
                       B, HIPBLAS_R_32F, ldb,
                       beta,
                       C, HIPBLAS_R_32F, ldc,
                       HIPBLAS_R_32F,
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
  //return hipblasHgemm(handle, transa, transb, m, n, k, (const hipblasHalf*)alpha, (const hipblasHalf*)A, lda, (const hipblasHalf*)B, ldb, (const hipblasHalf*)beta, (hipblasHalf*)C, ldc);
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return hipblasGemmEx(handle,
                       transa,
                       transb,
                       m, n, k,
                       &h_a,
                       A, HIPBLAS_R_16F, lda,
                       B, HIPBLAS_R_16F, ldb,
                       &h_b,
                       C, HIPBLAS_R_16F, ldc,
                       HIPBLAS_R_32F,
                       HIPBLAS_GEMM_DEFAULT);
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
  //return hipblasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
  return hipblasGemmBatchedEx(handle,
                              transa,
                              transb,
                              m, n, k,
                              alpha,
                              (const void**)Aarray, HIPBLAS_R_32F, lda,
                              (const void**)Barray, HIPBLAS_R_32F, ldb,
                              beta,
                              (void**)Carray, HIPBLAS_R_32F, ldc,
                              batchCount,
                              HIPBLAS_R_32F,
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
  //return hipblasHgemmBatched(handle, transa, transb, m, n, k, (const hipblasHalf*)alpha, (const hipblasHalf**)Aarray, lda, (const hipblasHalf**)Barray, ldb, (const hipblasHalf*)beta, (hipblasHalf**)Carray, ldc, batchCount);
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return hipblasGemmBatchedEx(handle,
                              transa,
                              transb,
                              m, n, k,
                              &h_a,
                              (const void**)Aarray, HIPBLAS_R_16F, lda,
                              (const void**)Barray, HIPBLAS_R_16F, ldb,
                              &h_b,
                              (void**)Carray, HIPBLAS_R_16F, ldc,
                              batchCount,
                              HIPBLAS_R_32F,
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
  //return hipblasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  return hipblasGemmStridedBatchedEx(handle,
                                     transa,
                                     transb,
                                     m, n, k,
                                     alpha,
                                     A, HIPBLAS_R_32F, lda, strideA,
                                     B, HIPBLAS_R_32F, ldb, strideB,
                                     beta,
                                     C, HIPBLAS_R_32F, ldc, strideC,
                                     batchCount,
                                     HIPBLAS_R_32F,
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
                                                       int batchCount){
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
  //return hipblasHgemmStridedBatched(handle, transa, transb, m, n, k, (const hipblasHalf*)alpha, (const hipblasHalf*)A, lda, strideA, (const hipblasHalf*)B, ldb, strideB, (const hipblasHalf*)beta, (hipblasHalf*)C, ldc, strideC, batchCount);
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  return hipblasGemmStridedBatchedEx(handle,
                                     transa,
                                     transb,
                                     m, n, k,
                                     &h_a,
                                     A, HIPBLAS_R_16F, lda, strideA,
                                     B, HIPBLAS_R_16F, ldb, strideB,
                                     &h_b,
                                     C, HIPBLAS_R_16F, ldc, strideC,
                                     batchCount,
                                     HIPBLAS_R_32F,
                                     HIPBLAS_GEMM_DEFAULT);
}



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

// // axpy
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
//   return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
// }
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
//   return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
// }
// inline hipblasStatus_t hipblasAxpyHelper(hipblasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy) {
//   float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
//   return hipblasAxpyEx(handle, n, (void*)&tmp_alpha, HIPBLAS_R_32F, (void*)x, HIPBLAS_R_16F, incx, (void*)y, HIPBLAS_R_16F, incy, HIPBLAS_R_32F);
// }

// transpose using geam
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return hipblasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return hipblasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// // asum
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t handle, int n, const float* x, int incx, float* result) {
//   return hipblasSasum(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t handle, int n, const double* x, int incx, double* result) {
//   return hipblasDasum(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAsumHelper(hipblasHandle_t, int n, const half* x, int incx, half* result) {
//   // pass in miopen handle/descriptor to remove overhead?
//   miopenHandle_t miopenHandle;
//   miopenTensorDescriptor_t srcTensorDesc, dstTensorDesc;
//   miopenReduceTensorDescriptor_t reduceTensorDesc;

//   miopenCreate(&miopenHandle);
//   miopenCreateTensorDescriptor(&srcTensorDesc);
//   miopenCreateTensorDescriptor(&dstTensorDesc);
//   miopenCreateReduceTensorDescriptor(&reduceTensorDesc);

//   miopenSetTensor4dDescriptorEx(srcTensorDesc, MIOPEN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
//   miopenSetTensor4dDescriptorEx(dstTensorDesc, MIOPEN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
//   miopenSetReduceTensorDescriptor(reduceTensorDesc,
//                                  MIOPEN_REDUCE_TENSOR_NORM1,
//                                  MIOPEN_DATA_FLOAT,
//                                  MIOPEN_NOT_PROPAGATE_NAN,
//                                  MIOPEN_REDUCE_TENSOR_NO_INDICES,
//                                  MIOPEN_32BIT_INDICES);

//   void* workspace = NULL;
//   size_t workspaceSizeInBytes = 0;
//   miopenGetReductionWorkspaceSize(miopenHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
//   if (workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

//   float alpha = 1.0f;
//   float beta = 0.0f;

//   void* d_res;
//   hipMalloc(&d_res, sizeof(half));

//   miopenReduceTensor(miopenHandle,
//                     reduceTensorDesc,
//                     NULL,
//                     0,
//                     workspace,
//                     workspaceSizeInBytes,
//                     &alpha,
//                     srcTensorDesc,
//                     (void*)x,
//                     &beta,
//                     dstTensorDesc,
//                     d_res);

//   hipMemcpy((void*)result, d_res, sizeof(half), hipMemcpyDeviceToHost);

//   miopenDestroyReduceTensorDescriptor(reduceTensorDesc);
//   miopenDestroyTensorDescriptor(srcTensorDesc);
//   miopenDestroyTensorDescriptor(dstTensorDesc);
//   miopenDestroy(miopenHandle);
//   hipFree(d_res);
//   hipFree(workspace);

//   return (hipblasStatus_t)0;
// }

// // amax
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t handle, int n, const float* x, int incx, int* result) {
//   return hipblasIsamax(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t handle, int n, const double* x, int incx, int* result) {
//   return hipblasIdamax(handle, n, x, incx, result);
// }
// inline hipblasStatus_t hipblasAmaxHelper(hipblasHandle_t, int n, const half* x, int incx, int* result) {
//   unsigned int h_result_uint = 0;
//   // pass in miopen handle/descriptor to remove overhead?
//   miopenHandle_t miopenHandle;
//   miopenTensorDescriptor_t srcTensorDesc, dstTensorDesc;
//   miopenReduceTensorDescriptor_t reduceTensorDesc;

//   miopenCreate(&miopenHandle);
//   miopenCreateTensorDescriptor(&srcTensorDesc);
//   miopenCreateTensorDescriptor(&dstTensorDesc);
//   miopenCreateReduceTensorDescriptor(&reduceTensorDesc);

//   miopenSetTensor4dDescriptorEx(srcTensorDesc, MIOPEN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
//   miopenSetTensor4dDescriptorEx(dstTensorDesc, MIOPEN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
//   miopenSetReduceTensorDescriptor(reduceTensorDesc,
//                                  MIOPEN_REDUCE_TENSOR_AMAX,
//                                  MIOPEN_DATA_FLOAT,
//                                  MIOPEN_NOT_PROPAGATE_NAN,
//                                  MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES,
//                                  MIOPEN_32BIT_INDICES);

//   void* workspace = NULL;
//   size_t workspaceSizeInBytes = 0;
//   miopenGetReductionWorkspaceSize(miopenHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
//   if (workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

//   float alpha = 1.0f;
//   float beta = 0.0f;
//   void* d_max;
//   hipMalloc(&d_max, sizeof(half));
//   void* d_result_uint;
//   hipMalloc(&d_result_uint, sizeof(unsigned int));

//   miopenReduceTensor(miopenHandle,
//                     reduceTensorDesc,
//                     d_result_uint,
//                     sizeof(unsigned int),
//                     workspace,
//                     workspaceSizeInBytes,
//                     &alpha,
//                     srcTensorDesc,
//                     (void*)x,
//                     &beta,
//                     dstTensorDesc,
//                     d_max);

//   hipMemcpy(&h_result_uint, d_result_uint, sizeof(unsigned int), hipMemcpyDeviceToHost);

//   miopenDestroyReduceTensorDescriptor(reduceTensorDesc);
//   miopenDestroyTensorDescriptor(srcTensorDesc);
//   miopenDestroyTensorDescriptor(dstTensorDesc);
//   miopenDestroy(miopenHandle);
//   hipFree(workspace);
//   hipFree(d_max);
//   hipFree(d_result_uint);

//   *result = (int)h_result_uint;
//   return (hipblasStatus_t)0;
// }

// scal
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx) {
  return hipblasSscal(handle, n, alpha, x, incx);
}
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx) {
  return hipblasDscal(handle, n, alpha, x, incx);
}
// inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t handle, int n, const half* alpha, half* x, int incx) {
//   float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
//   return hipblasScalEx(handle, n, (void*)&tmp_alpha, HIPBLAS_R_32F, (void*)x, HIPBLAS_R_16F, incx, HIPBLAS_R_32F);
// }
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t, int, const char*, char*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(char) in hipblas_scal");
}
inline hipblasStatus_t hipblasScalHelper(hipblasHandle_t, int, const short*, short*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(short) in hipblas_scal");
}

// dot
inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
  return hipblasSdot(handle, n, x, incx, y, incy, result);
}
inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
  return hipblasDdot(handle, n, x, incx, y, incy, result);
}
// inline hipblasStatus_t hipblasDotHelper(hipblasHandle_t handle, int n, const half* x, int incx, const half* y, int incy, half* result) {
//   return hipblasDotEx(handle, n, (void*)x, HIPBLAS_R_16F, incx, (void*)y, HIPBLAS_R_16F, incy, (void*)result, HIPBLAS_R_16F, HIPBLAS_R_32F);
// }

// copy
inline hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return hipblasScopy(handle, n, x, incx, y, incy);
}
inline hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return hipblasDcopy(handle, n, x, incx, y, incy);
}
hipblasStatus_t hipblasCopyHelper(hipblasHandle_t handle, int n, const half* x, int incx, half* y, int incy);
