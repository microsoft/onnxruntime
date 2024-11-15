// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/cudnn_fe_call.h"
#include "core/providers/shared_library/provider_api.h"
#include <core/platform/env.h>
#if !defined(__CUDACC__) && !defined(USE_CUDA_MINIMAL)
#include <cudnn_frontend.h>
#endif
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

#if !defined(__CUDACC__) && !defined(USE_CUDA_MINIMAL)
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

template <typename ERRTYPE>
int GetErrorCode(ERRTYPE err) {
  return static_cast<int>(err);
}

template <>
int GetErrorCode(cudnn_frontend::error_t err) {
  return static_cast<int>(err.get_code());
}

template <typename ERRTYPE, bool THRW, typename SUCCTYPE>
std::conditional_t<THRW, void, Status> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg,
    const char* file, const int line) {
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
    } catch (const std::exception& e) {  // catch, log, and rethrow since CUDA code sometimes hangs in destruction,
      // so we'd never get to see the error
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

template Status CudaCall<cudnn_frontend::error_t, false, cudnn_frontend::error_code_t>(
    cudnn_frontend::error_t retCode, const char* exprString, const char* libName,
    cudnn_frontend::error_code_t successCode, const char* msg, const char* file, const int line);
template void CudaCall<cudnn_frontend::error_t, true, cudnn_frontend::error_code_t>(
    cudnn_frontend::error_t retCode, const char* exprString, const char* libName,
    cudnn_frontend::error_code_t successCode, const char* msg, const char* file, const int line);

#endif
}  // namespace onnxruntime
