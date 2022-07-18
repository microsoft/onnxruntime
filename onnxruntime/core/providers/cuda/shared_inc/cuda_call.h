// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_pch.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
bool CudaCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

void set_cudnn_stream(cudnnHandle_t handle, cudaStream_t stream, cudnnStatus_t& output);

void set_cublas_stream(cublasHandle_t handle, cudaStream_t stream, cublasStatus_t& output);

template <typename ERRTYPE, typename HANDLETYPE>
ERRTYPE cuda_lib_call_with_stream(cudaStream_t stream, 
    HANDLETYPE handle, 
    std::function<void(HANDLETYPE, cudaStream_t, ERRTYPE&)> set_stream_f,
    std::function<void(ERRTYPE&)> f,
    ERRTYPE successCode) {
  OrtMutex stream_mutex;
  std::lock_guard<OrtMutex> lock(stream_mutex);
  ERRTYPE status;
  set_stream_f(handle, stream, status);
  if (status != successCode)
    return status;
  f(status);
  return status;
}

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CONFIG_CALL(expr) (CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))

#define CUBLAS_CALL(expr, handle, stream) (CudaCall<cublasStatus_t, false>(                \
    cuda_lib_call_with_stream<cublasStatus_t, cublasHandle_t>(                             \
        stream, handle,                                                                    \
        set_cublas_stream, \
        [&](cublasStatus_t& output) { output = (expr); }, CUBLAS_STATUS_SUCCESS), \
    #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))

#define CUSPARSE_CALL(expr) (CudaCall<cusparseStatus_t, false>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL(expr) (CudaCall<curandStatus_t, false>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))
// the cudnn configuration call that doesn't need set stream
#define CUDNN_CONFIG_CALL(expr) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))

#define CUDNN_CALL(expr, handle, stream) (CudaCall<cudnnStatus_t, false>(                \
    cuda_lib_call_with_stream<cudnnStatus_t, cudnnHandle_t>(                                \
        stream, handle, set_cudnn_stream, \
        [&](cudnnStatus_t& output) { output = (expr); }, CUDNN_STATUS_SUCCESS), \
    #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL_2(expr, handle, stream, m) (CudaCall<cudnnStatus_t, false>(                                                                  \
    cuda_lib_call_with_stream<cudnnStatus_t, cudnnHandle_t>(                                                                               \
        stream, handle, set_cudnn_stream, \
        [&](cudnnStatus_t& output) { output = (expr); }, CUDNN_STATUS_SUCCESS),                                                            \
    #expr, "CUDNN", CUDNN_STATUS_SUCCESS), m)

#define CUFFT_CALL(expr) (CudaCall<cufftResult, false>((expr), #expr, "CUFFT", CUFFT_SUCCESS))


#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CONFIG_CALL_TRHOW(expr) (CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))
#define CUBLAS_CALL_THROW(expr, handle, stream) (CudaCall<cublasStatus_t, true>(                                                   \
    cuda_lib_call_with_stream<cublasStatus_t, cublasHandle_t>(                                                                \
        stream, handle,                                                                                                       \
        [](cublasHandle_t handle, cudaStream_t stream, cublasStatus_t& output) { output = cublasSetStream(handle, stream); }, \
        [&](cublasStatus_t& output) { output = (expr); }, CUBLAS_STATUS_SUCCESS),                                                                     \
    #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))

#define CUSPARSE_CALL_THROW(expr) (CudaCall<cusparseStatus_t, true>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL_THROW(expr) (CudaCall<curandStatus_t, true>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))

// the cudnn configuration call that doesn't need set stream
#define CUDNN_CONFIG_CALL_THROW(expr) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))

#define CUDNN_CALL_THROW(expr, handle, stream) (CudaCall<cudnnStatus_t, true>(                                                                  \
    cuda_lib_call_with_stream<cudnnStatus_t, cudnnHandle_t>(                                                                               \
        stream, handle, set_cudnn_stream, \
        [&](cudnnStatus_t& output) { output = (expr); }, CUDNN_STATUS_SUCCESS),                                                            \
    #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL_THROW2(expr, handle, stream, m) (CudaCall<cudnnStatus_t, true>(                                                                  \
    cuda_lib_call_with_stream<cudnnStatus_t, cudnnHandle_t>(                                                                               \
        stream, handle, set_cudnn_stream, \
        [&](cudnnStatus_t& output) { output = (expr); }, CUDNN_STATUS_SUCCESS),                                                            \
    #expr, "CUDNN", CUDNN_STATUS_SUCCESS), m)

#define CUFFT_CALL_THROW(expr) (CudaCall<cufftResult, true>((expr), #expr, "CUFFT", CUFFT_SUCCESS))

#ifdef ORT_USE_NCCL
#define NCCL_CALL(expr) (CudaCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess))
#define NCCL_CALL_THROW(expr) (CudaCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess))
#endif

}  // namespace onnxruntime
