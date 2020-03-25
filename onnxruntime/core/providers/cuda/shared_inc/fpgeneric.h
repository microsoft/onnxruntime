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
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime;

// Generalize library calls to be use in template functions

// gemm
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half* A, int lda, const half* B, int ldb, const half* beta, half* C, int ldc) {
  // Disable below to make sure merged result is on par with before-merge.
  // This does true FP16 computation which is slow for non-Volta GPUs
  //if (cuda::DeviceProp().GetDeviceProps().major >= 7) {
  //   cuda::CublasMathModeSetter math_mode_setter( handle, CUBLAS_TENSOR_OP_MATH );
  //  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  //}
  // This does pseudo FP16 computation (input/output in fp16, computation in fp32)
  float h_a = math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT);
}

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* Aarray[], int lda, const float* Barray[], int ldb, const float* beta, float* Carray[], int ldc, int batch_count) {
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batch_count);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* Aarray[], int lda, const double* Barray[], int ldb, const double* beta, double* Carray[], int ldc, int batch_count) {
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batch_count);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half* Aarray[], int lda, const half* Barray[], int ldb, const half* beta, half* Carray[], int ldc, int batch_count) {
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half**)Aarray, lda, (const __half**)Barray, ldb, beta, (__half**)Carray, ldc, batch_count);
}

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
                                                     int batch_count) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
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
                                                     int batch_count) {
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
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
                                                     int batch_count) {
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  return cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batch_count);
}

inline int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

// Use cublasLtMatmul to perform the tensor op Igemm with the memory
// order transforms on all buffers.
//
// For better performance the data order transforms should be offline
// as much as possible.
//
// Transa, transb assumed N; alpha, beta are host pointers; Tensor ops
// allowed. Alpha assumed 1, beta assumed 0, and stream assumed 0.

inline void LtIgemmTensor(cublasLtHandle_t ltHandle,
                          int m,
                          int n,
                          int k,
                          int32_t alpha,
                          int32_t beta,
                          const int8_t* A,
                          int lda,
                          const int8_t* B,
                          int ldb,
                          int32_t* C,
                          int ldc) {
  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL;
  cublasOperation_t op_transpose = CUBLAS_OP_T;

  // The tensor op igemm kernels require specialized memory order of
  // data.
  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  int8_t *a_transform = NULL, *b_transform = NULL;
  int32_t* c_transform = NULL;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

  int ldatransform = 32 * m;
  int ldbtransform = 32 * roundoff(n, 8);
  int ldctransform = 32 * m;

  CUDA_CALL_THROW(cudaMalloc(&a_transform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
  CUDA_CALL_THROW(cudaMalloc(&b_transform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
  CUDA_CALL_THROW(cudaMalloc(&c_transform, sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));

  CUBLAS_CALL_THROW(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));

  //// B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
  //CUBLAS_CALL_THROW(cublasLtMatrixTransformDescSetAttribute(transform_desc,
  //                                                          CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
  //                                                          &op_transpose,
  //                                                          sizeof(op_transpose)));

  // Tensor op igemm kernels only support NT gemm
  CUBLAS_CALL_THROW(cublasLtMatmulDescCreate(&matmulDesc, CUDA_R_32I));
  CUBLAS_CALL_THROW(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_transpose, sizeof(op_transpose)));

  // Create descriptors for the original matrices
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8I, m, k, lda));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, n, k, ldb));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32I, m, n, ldc));

  // Create descriptors for the transformed matrices
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

  CUBLAS_CALL_THROW(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
  CUBLAS_CALL_THROW(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

  // Transforms and computation
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle,
                                            transform_desc,
                                            &transformAlpha,
                                            A,
                                            a_desc,
                                            &transformBeta,
                                            NULL,
                                            NULL,
                                            a_transform,
                                            AtransformDesc,
                                            0));
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle,
                                            transform_desc,
                                            &transformAlpha,
                                            B,
                                            b_desc,
                                            &transformBeta,
                                            NULL,
                                            NULL,
                                            b_transform,
                                            BtransformDesc,
                                            0));

  // No need to transform C matrix as beta is assumed to be 0
  CUBLAS_CALL_THROW(cublasLtMatmul(ltHandle,
                                   matmulDesc,
                                   &alpha,
                                   a_transform,
                                   AtransformDesc,
                                   b_transform,
                                   BtransformDesc,
                                   &beta,
                                   c_transform,
                                   CtransformDesc,
                                   c_transform,
                                   CtransformDesc,
                                   NULL,
                                   NULL,
                                   0,
                                   0));

  // Transform the outputs to COL order
  CUBLAS_CALL_THROW(cublasLtMatrixTransform(ltHandle,
                                            transform_desc,
                                            &transformAlpha,
                                            c_transform,
                                            CtransformDesc,
                                            &transformBeta,
                                            NULL,
                                            NULL,
                                            C,
                                            c_desc,
                                            0));

  // Descriptors are no longer needed as all GPU work was already
  // enqueued.
  if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
  if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
  if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
  if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
  if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
  if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
  if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
  if (transform_desc) cublasLtMatrixTransformDescDestroy(transform_desc);

  // Wait until device is done before freeing transformed buffers
  if (c_transform) cudaFree(c_transform);
  if (b_transform) cudaFree(b_transform);
  if (a_transform) cudaFree(a_transform);
}

// axpy
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy) {
  float tmp_alpha = math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  return cublasAxpyEx(handle, n, (void*)&tmp_alpha, CUDA_R_32F, (void*)x, CUDA_R_16F, incx, (void*)y, CUDA_R_16F, incy, CUDA_R_32F);
}

// transpose using geam
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasTransposeHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
cublasStatus_t cublasTransposeHelper(cublasHandle_t, cublasOperation_t, cublasOperation_t, int m, int n, const half*, const half* A, int, const half*, const half*, int, half* C, int);

// asum
inline cublasStatus_t cublasAsumHelper(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
  return cublasSasum(handle, n, x, incx, result);
}
inline cublasStatus_t cublasAsumHelper(cublasHandle_t handle, int n, const double* x, int incx, double* result) {
  return cublasDasum(handle, n, x, incx, result);
}
inline cublasStatus_t cublasAsumHelper(cublasHandle_t, int n, const half* x, int incx, half* result) {
  // pass in cudnn handle/descriptor to remove overhead?
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
  cudnnReduceTensorDescriptor_t reduceTensorDesc;

  cudnnCreate(&cudnnHandle);
  cudnnCreateTensorDescriptor(&srcTensorDesc);
  cudnnCreateTensorDescriptor(&dstTensorDesc);
  cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);

  cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
  cudnnSetTensor4dDescriptorEx(dstTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
  cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                 CUDNN_REDUCE_TENSOR_NORM1,
                                 CUDNN_DATA_FLOAT,
                                 CUDNN_NOT_PROPAGATE_NAN,
                                 CUDNN_REDUCE_TENSOR_NO_INDICES,
                                 CUDNN_32BIT_INDICES);

  void* workspace = NULL;
  size_t workspaceSizeInBytes = 0;
  cudnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
  if (workspaceSizeInBytes > 0) cudaMalloc(&workspace, workspaceSizeInBytes);

  float alpha = 1.0f;
  float beta = 0.0f;

  void* d_res;
  cudaMalloc(&d_res, sizeof(half));

  cudnnReduceTensor(cudnnHandle,
                    reduceTensorDesc,
                    NULL,
                    0,
                    workspace,
                    workspaceSizeInBytes,
                    &alpha,
                    srcTensorDesc,
                    (void*)x,
                    &beta,
                    dstTensorDesc,
                    d_res);

  cudaMemcpy((void*)result, d_res, sizeof(half), cudaMemcpyDeviceToHost);

  cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  cudnnDestroyTensorDescriptor(srcTensorDesc);
  cudnnDestroyTensorDescriptor(dstTensorDesc);
  cudnnDestroy(cudnnHandle);
  cudaFree(d_res);
  cudaFree(workspace);

  return (cublasStatus_t)0;
}

// amax
inline cublasStatus_t cublasAmaxHelper(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
  return cublasIsamax(handle, n, x, incx, result);
}
inline cublasStatus_t cublasAmaxHelper(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
  return cublasIdamax(handle, n, x, incx, result);
}
inline cublasStatus_t cublasAmaxHelper(cublasHandle_t, int n, const half* x, int incx, int* result) {
  unsigned int h_result_uint = 0;
  // pass in cudnn handle/descriptor to remove overhead?
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
  cudnnReduceTensorDescriptor_t reduceTensorDesc;

  cudnnCreate(&cudnnHandle);
  cudnnCreateTensorDescriptor(&srcTensorDesc);
  cudnnCreateTensorDescriptor(&dstTensorDesc);
  cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);

  cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
  cudnnSetTensor4dDescriptorEx(dstTensorDesc, CUDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
  cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                 CUDNN_REDUCE_TENSOR_AMAX,
                                 CUDNN_DATA_FLOAT,
                                 CUDNN_NOT_PROPAGATE_NAN,
                                 CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                                 CUDNN_32BIT_INDICES);

  void* workspace = NULL;
  size_t workspaceSizeInBytes = 0;
  cudnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
  if (workspaceSizeInBytes > 0) cudaMalloc(&workspace, workspaceSizeInBytes);

  float alpha = 1.0f;
  float beta = 0.0f;
  void* d_max;
  cudaMalloc(&d_max, sizeof(half));
  void* d_result_uint;
  cudaMalloc(&d_result_uint, sizeof(unsigned int));

  cudnnReduceTensor(cudnnHandle,
                    reduceTensorDesc,
                    d_result_uint,
                    sizeof(unsigned int),
                    workspace,
                    workspaceSizeInBytes,
                    &alpha,
                    srcTensorDesc,
                    (void*)x,
                    &beta,
                    dstTensorDesc,
                    d_max);

  cudaMemcpy(&h_result_uint, d_result_uint, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  cudnnDestroyTensorDescriptor(srcTensorDesc);
  cudnnDestroyTensorDescriptor(dstTensorDesc);
  cudnnDestroy(cudnnHandle);
  cudaFree(workspace);
  cudaFree(d_max);
  cudaFree(d_result_uint);

  *result = (int)h_result_uint;
  return (cublasStatus_t)0;
}

// scal
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
  return cublasSscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
  return cublasDscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const half* alpha, half* x, int incx) {
  float tmp_alpha = math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  return cublasScalEx(handle, n, (void*)&tmp_alpha, CUDA_R_32F, (void*)x, CUDA_R_16F, incx, CUDA_R_32F);
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t, int, const char*, char*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(char) in cublas_scal");
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t, int, const short*, short*, int) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(short) in cublas_scal");
}

// dot
inline cublasStatus_t cublasDotHelper(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasDotHelper(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}
inline cublasStatus_t cublasDotHelper(cublasHandle_t handle, int n, const half* x, int incx, const half* y, int incy, half* result) {
  return cublasDotEx(handle, n, (void*)x, CUDA_R_16F, incx, (void*)y, CUDA_R_16F, incy, (void*)result, CUDA_R_16F, CUDA_R_32F);
}

// copy
inline cublasStatus_t cublasCopyHelper(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}
inline cublasStatus_t cublasCopyHelper(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}
cublasStatus_t cublasCopyHelper(cublasHandle_t handle, int n, const half* x, int incx, half* y, int incy);

// curand
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, float* outputPtr, size_t num) {
  return curandGenerateUniform(generator, outputPtr, num);
}
inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t generator, double* outputPtr, size_t num) {
  return curandGenerateUniformDouble(generator, outputPtr, num);
}
curandStatus_t curandGenerateUniformHelper(curandGenerator_t, half* outputPtr, size_t num);

inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, char*, size_t) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(char) in GPUSparseMatrix");
}

inline curandStatus_t curandGenerateUniformHelper(curandGenerator_t, short*, size_t) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(short) in GPUSparseMatrix");
}

inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, float* outputPtr, size_t n, float mean, float stddev) {
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}
inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t generator, double* outputPtr, size_t n, double mean, double stddev) {
  return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
curandStatus_t curandGenerateNormalHelper(curandGenerator_t, half* outputPtr, size_t n, half mean, half stddev);

inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, char*, size_t, char, char) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(char) in GPUSparseMatrix");
}

inline curandStatus_t curandGenerateNormalHelper(curandGenerator_t, short*, size_t, short, short) {
  ORT_NOT_IMPLEMENTED("Unsupported template argument(short) in GPUSparseMatrix");
}
