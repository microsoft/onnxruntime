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

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CALL(expr) (CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))
#define CUSPARSE_CALL(expr) (CudaCall<cusparseStatus_t, false>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL(expr) (CudaCall<curandStatus_t, false>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))
#define CUDNN_CALL(expr) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL2(expr, m) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m))
#define CUFFT_CALL(expr) (CudaCall<cufftResult, false>((expr), #expr, "CUFFT", CUFFT_SUCCESS))

#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess))
#define CUBLAS_CALL_THROW(expr) (CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS))
#define CUSPARSE_CALL_THROW(expr) (CudaCall<cusparseStatus_t, true>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS))
#define CURAND_CALL_THROW(expr) (CudaCall<curandStatus_t, true>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS))
#define CUDNN_CALL_THROW(expr) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS))
#define CUDNN_CALL_THROW2(expr, m) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m))
#define CUFFT_CALL_THROW(expr) (CudaCall<cufftResult, true>((expr), #expr, "CUFFT", CUFFT_SUCCESS))

#ifdef USE_NCCL
#define NCCL_CALL(expr) (CudaCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess))
#define NCCL_CALL_THROW(expr) (CudaCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess))
#endif

}  // namespace onnxruntime
