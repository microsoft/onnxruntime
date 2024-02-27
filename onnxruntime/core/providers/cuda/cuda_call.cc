// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "shared_inc/cuda_call.h"
#include <core/platform/env.h>

#ifdef _WIN32
#else  // POSIX
#include <unistd.h>
#include <string.h>
#endif

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* CudaErrString(ERRTYPE) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* CudaErrString<cudaError_t>(cudaError_t x) {
  cudaDeviceSynchronize();
  return cudaGetErrorString(x);
}

#ifndef USE_CUDA_MINIMAL
template <>
const char* CudaErrString<cublasStatus_t>(cublasStatus_t e) {
  cudaDeviceSynchronize();
  switch (e) {
    CASE_ENUM_TO_STR(CUBLAS_STATUS_SUCCESS);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_INITIALIZED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_ARCH_MISMATCH);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_MAPPING_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_EXECUTION_FAILED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_NOT_SUPPORTED);
    CASE_ENUM_TO_STR(CUBLAS_STATUS_LICENSE_ERROR);
    default:
      return "(look for CUBLAS_STATUS_xxx in cublas_api.h)";
  }
}

template <>
const char* CudaErrString<curandStatus>(curandStatus) {
  cudaDeviceSynchronize();
  return "(see curand.h & look for curandStatus or CURAND_STATUS_xxx)";
}

template <>
const char* CudaErrString<cudnnStatus_t>(cudnnStatus_t e) {
  cudaDeviceSynchronize();
  return cudnnGetErrorString(e);
}

template <>
const char* CudaErrString<cufftResult>(cufftResult e) {
  cudaDeviceSynchronize();
  switch (e) {
    CASE_ENUM_TO_STR(CUFFT_SUCCESS);
    CASE_ENUM_TO_STR(CUFFT_ALLOC_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_VALUE);
    CASE_ENUM_TO_STR(CUFFT_INTERNAL_ERROR);
    CASE_ENUM_TO_STR(CUFFT_SETUP_FAILED);
    CASE_ENUM_TO_STR(CUFFT_INVALID_SIZE);
    default:
      return "Unknown cufft error status";
  }
}
#endif

#if defined(ENABLE_CUDA_NHWC_OPS) && !defined(__CUDACC__)
#define CASE_ENUM_TO_STR_CUDNN_FE(x)    \
  case cudnn_frontend::error_code_t::x: \
    return #x
template <>
const char* CudaErrString<cudnn_frontend::error_t>(cudnn_frontend::error_t x) {
  cudaDeviceSynchronize();
  LOGS_DEFAULT(ERROR) << x.get_message();
  switch (x.get_code()) {
    CASE_ENUM_TO_STR_CUDNN_FE(OK);
    CASE_ENUM_TO_STR_CUDNN_FE(ATTRIBUTE_NOT_SET);
    CASE_ENUM_TO_STR_CUDNN_FE(SHAPE_DEDUCTION_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(INVALID_TENSOR_NAME);
    CASE_ENUM_TO_STR_CUDNN_FE(INVALID_VARIANT_PACK);
    CASE_ENUM_TO_STR_CUDNN_FE(GRAPH_NOT_SUPPORTED);
    CASE_ENUM_TO_STR_CUDNN_FE(GRAPH_EXECUTION_PLAN_CREATION_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(GRAPH_EXECUTION_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(HEURISTIC_QUERY_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(UNSUPPORTED_GRAPH_FORMAT);
    CASE_ENUM_TO_STR_CUDNN_FE(CUDA_API_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(CUDNN_BACKEND_API_FAILED);
    CASE_ENUM_TO_STR_CUDNN_FE(INVALID_CUDA_DEVICE);
    CASE_ENUM_TO_STR_CUDNN_FE(HANDLE_ERROR);
    default:
      return "Unknown CUDNN_FRONTEND error status";
  }
}
#endif

#ifdef ORT_USE_NCCL
template <>
const char* CudaErrString<ncclResult_t>(ncclResult_t e) {
  cudaDeviceSynchronize();
  return ncclGetErrorString(e);
}
#endif

template <typename ERRTYPE>
int GetErrorCode(ERRTYPE err) {
  return (int)err;
}

#if defined(ENABLE_CUDA_NHWC_OPS) && !defined(__CUDACC__)
template <>
int GetErrorCode(cudnn_frontend::error_t err) {
  return (int)err.get_code();
}
#endif

template <typename ERRTYPE, bool THRW, typename SUCCTYPE>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg, const char* file, const int line) {
  if (retCode != successCode) {
    try {
#ifdef _WIN32
      std::string hostname_str = GetEnvironmentVar("COMPUTERNAME");
      if (hostname_str.empty()) {
        hostname_str = "?";
      }
      const char* hostname = hostname_str.c_str();
#else
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
#endif
      int currentCudaDevice;
      cudaGetDevice(&currentCudaDevice);
      cudaGetLastError();  // clear last CUDA error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; file=%s ; line=%d ; expr=%s; %s",
               libName, GetErrorCode(retCode), CudaErrString(retCode), currentCudaDevice,
               hostname,
               file, line, exprString, msg);
      if constexpr (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, str);
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction, so we'd never get to see the error
      if constexpr (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
      }
    }
  }
  if constexpr (!THRW) {
    return Status::OK();
  }
}

template Status CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg, const char* file, const int line);
#ifndef USE_CUDA_MINIMAL
#if defined(ENABLE_CUDA_NHWC_OPS) && !defined(__CUDACC__)
template Status CudaCall<cudnn_frontend::error_t, false, cudnn_frontend::error_code_t>(cudnn_frontend::error_t retCode, const char* exprString, const char* libName, cudnn_frontend::error_code_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudnn_frontend::error_t, true, cudnn_frontend::error_code_t>(cudnn_frontend::error_t retCode, const char* exprString, const char* libName, cudnn_frontend::error_code_t successCode, const char* msg, const char* file, const int line);
#endif
template Status CudaCall<cublasStatus_t, false>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cublasStatus_t, true>(cublasStatus_t retCode, const char* exprString, const char* libName, cublasStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cudnnStatus_t, false>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudnnStatus_t, true>(cudnnStatus_t retCode, const char* exprString, const char* libName, cudnnStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<curandStatus_t, false>(curandStatus_t retCode, const char* exprString, const char* libName, curandStatus_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<curandStatus_t, true>(curandStatus_t retCode, const char* exprString, const char* libName, curandStatus_t successCode, const char* msg, const char* file, const int line);
template Status CudaCall<cufftResult, false>(cufftResult retCode, const char* exprString, const char* libName, cufftResult successCode, const char* msg, const char* file, const int line);
template void CudaCall<cufftResult, true>(cufftResult retCode, const char* exprString, const char* libName, cufftResult successCode, const char* msg, const char* file, const int line);
#endif

#ifdef ORT_USE_NCCL
template Status CudaCall<ncclResult_t, false>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<ncclResult_t, true>(ncclResult_t retCode, const char* exprString, const char* libName, ncclResult_t successCode, const char* msg, const char* file, const int line);
#endif
}  // namespace onnxruntime
