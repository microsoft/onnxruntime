// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_pch.h"
#if defined(ENABLE_CUDA_NHWC_OPS) && !defined(USE_CUDA_MINIMAL) && !defined(__CUDACC__)
#include <cudnn_frontend.h>
#endif
namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW, typename SUCCTYPE = ERRTYPE>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg, const char* file, const int line);

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUBLAS_CALL(expr) (CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CUSPARSE_CALL(expr) (CudaCall<cusparseStatus_t, false>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CURAND_CALL(expr) (CudaCall<curandStatus_t, false>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CUDNN_CALL(expr) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CUDNN_CALL2(expr, m) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, m, __FILE__, __LINE__))

#define CUDNN_FE_CALL(expr) (CudaCall<cudnn_frontend::error_t, false, cudnn_frontend::error_code_t>((cudnn_frontend::error_t)(expr), #expr, "CUDNN_FE", cudnn_frontend::error_code_t::OK, "", __FILE__, __LINE__))

#define CUFFT_CALL(expr) (CudaCall<cufftResult, false>((expr), #expr, "CUFFT", CUFFT_SUCCESS, "", __FILE__, __LINE__))

#define CUDA_CALL_THROW(expr) (CudaCall<cudaError, true>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUBLAS_CALL_THROW(expr) (CudaCall<cublasStatus_t, true>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CUSPARSE_CALL_THROW(expr) (CudaCall<cusparseStatus_t, true>((expr), #expr, "CUSPARSE", CUSPARSE_STATUS_SUCCESS, "", __FILE__, __LINE__))
#define CURAND_CALL_THROW(expr) (CudaCall<curandStatus_t, true>((expr), #expr, "CURAND", CURAND_STATUS_SUCCESS, "", __FILE__, __LINE__))

// the cudnn configuration call that doesn't need set stream
#define CUDNN_CALL_THROW(expr) (CudaCall<cudnnStatus_t, true>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, "", __FILE__, __LINE__))

#define CUFFT_CALL_THROW(expr) (CudaCall<cufftResult, true>((expr), #expr, "CUFFT", CUFFT_SUCCESS, "", __FILE__, __LINE__))

#ifdef ORT_USE_NCCL
#define NCCL_CALL(expr) (CudaCall<ncclResult_t, false>((expr), #expr, "NCCL", ncclSuccess, "", __FILE__, __LINE__))
#define NCCL_CALL_THROW(expr) (CudaCall<ncclResult_t, true>((expr), #expr, "NCCL", ncclSuccess, "", __FILE__, __LINE__))
#endif

}  // namespace onnxruntime
