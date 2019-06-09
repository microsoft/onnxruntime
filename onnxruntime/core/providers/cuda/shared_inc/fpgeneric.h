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

// Generalize library calls to be use in template functions

// gemm
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline cublasStatus_t cublasGemmHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half* A, int lda, const half* B, int ldb, const half* beta, half* C, int ldc) {
  // This does true FP16 computation which is slow for non-Volta GPUs
  //return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  // This does pseudo FP16 computation (input/output in fp16, computation in fp32)
  float h_a = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
  float h_b = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(beta));
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  return cublasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT);
}

// batched gemm
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* Aarray[], int lda, const float* Barray[], int ldb, const float* beta, float* Carray[], int ldc, int batchCount) {
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* Aarray[], int lda, const double* Barray[], int ldb, const double* beta, double* Carray[], int ldc, int batchCount) {
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline cublasStatus_t cublasGemmBatchedHelper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const half* alpha, const half* Aarray[], int lda, const half* Barray[], int ldb, const half* beta, half* Carray[], int ldc, int batchCount) {
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, static_cast<const __half**>(Aarray), lda,
                            static_cast<const __half**>(Barray), ldb, beta, static_cast<__half**>(Carray), ldc,
                            batchCount);
}

// axpy
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
inline cublasStatus_t cublasAxpyHelper(cublasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy) {
  float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
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
  cudnnTensorDescriptor_t srcTensorDesc;
  cudnnTensorDescriptor_t dstTensorDesc;
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

  void* workspace = nullptr;
  size_t workspaceSizeInBytes = 0;
  cudnnGetReductionWorkspaceSize(cudnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
  if (workspaceSizeInBytes > 0) cudaMalloc(&workspace, workspaceSizeInBytes);

  float alpha = 1.0f;
  float beta = 0.0f;

  void* d_res;
  cudaMalloc(&d_res, sizeof(half));

  cudnnReduceTensor(cudnnHandle, reduceTensorDesc, nullptr, 0, workspace, workspaceSizeInBytes, &alpha, srcTensorDesc,
                    (void*)x, &beta, dstTensorDesc, d_res);

  cudaMemcpy((void*)result, d_res, sizeof(half), cudaMemcpyDeviceToHost);

  cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
  cudnnDestroyTensorDescriptor(srcTensorDesc);
  cudnnDestroyTensorDescriptor(dstTensorDesc);
  cudnnDestroy(cudnnHandle);
  cudaFree(d_res);
  cudaFree(workspace);

  return static_cast<cublasStatus_t>(0);
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
  cudnnTensorDescriptor_t srcTensorDesc;
  cudnnTensorDescriptor_t dstTensorDesc;
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

  void* workspace = nullptr;
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

  *result = static_cast<int>(h_result_uint);
  return static_cast<cublasStatus_t>(0);
}

// scal
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
  return cublasSscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
  return cublasDscal(handle, n, alpha, x, incx);
}
inline cublasStatus_t cublasScalHelper(cublasHandle_t handle, int n, const half* alpha, half* x, int incx) {
  float tmp_alpha = onnxruntime::math::halfToFloat(*reinterpret_cast<const uint16_t*>(alpha));
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
