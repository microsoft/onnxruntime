// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Common utilities, error-handling macros, and type definitions shared by
// all source files in the CUDA plugin EP implementation.

#pragma once

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include "core/common/common.h"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cublasLt.h>

// Error handling macros

#ifndef PL_CUDA_RETURN_IF_ERROR
#define PL_CUDA_RETURN_IF_ERROR(cuda_call_expr)                               \
  do {                                                                        \
    cudaError_t _cuda_err = (cuda_call_expr);                                 \
    if (_cuda_err != cudaSuccess) {                                           \
      return Ort::GetApi().CreateStatus(                                      \
          ORT_EP_FAIL,                                                        \
          (std::string("CUDA error: ") + cudaGetErrorName(_cuda_err) + ": " + \
           cudaGetErrorString(_cuda_err))                                     \
              .c_str());                                                      \
    }                                                                         \
  } while (0)
#endif

inline Ort::Status StatusFromCudaError(cudaError_t cuda_err) {
  if (cuda_err == cudaSuccess) {
    return Ort::Status{};
  }

  return Ort::Status{
      (std::string("CUDA error: ") + cudaGetErrorName(cuda_err) + ": " +
       cudaGetErrorString(cuda_err))
          .c_str(),
      ORT_EP_FAIL};
}

inline Ort::Status StatusFromCublasError(cublasStatus_t cublas_err) {
  if (cublas_err == CUBLAS_STATUS_SUCCESS) {
    return Ort::Status{};
  }

  return Ort::Status{
      (std::string("cuBLAS error: ") + cublasGetStatusString(cublas_err)).c_str(),
      ORT_EP_FAIL};
}

inline Ort::Status StatusFromCudnnError(cudnnStatus_t cudnn_err) {
  if (cudnn_err == CUDNN_STATUS_SUCCESS) {
    return Ort::Status{};
  }

  return Ort::Status{
      (std::string("cuDNN error: ") + cudnnGetErrorString(cudnn_err)).c_str(),
      ORT_EP_FAIL};
}

inline bool TryGetCurrentCudaDevice(int& device_id) noexcept {
  return cudaGetDevice(&device_id) == cudaSuccess;
}

// Throwing variant for use in constructors and non-OrtStatus contexts.
// Analogous to CUDA_CALL_THROW in the non-plugin build.
#ifndef PL_CUDA_CALL_THROW
#define PL_CUDA_CALL_THROW(cuda_call_expr)                         \
  do {                                                             \
    cudaError_t _cuda_err = (cuda_call_expr);                      \
    if (_cuda_err != cudaSuccess) {                                \
      ORT_THROW("CUDA error: ", cudaGetErrorName(_cuda_err), ": ", \
                cudaGetErrorString(_cuda_err));                    \
    }                                                              \
  } while (0)
#endif

#ifndef PL_CUBLAS_RETURN_IF_ERROR
#define PL_CUBLAS_RETURN_IF_ERROR(cublas_call_expr)  \
  do {                                               \
    cublasStatus_t _cublas_err = (cublas_call_expr); \
    if (_cublas_err != CUBLAS_STATUS_SUCCESS) {      \
      return Ort::GetApi().CreateStatus(             \
          ORT_EP_FAIL,                               \
          (std::string("cuBLAS error: ") +           \
           cublasGetStatusString(_cublas_err))       \
              .c_str());                             \
    }                                                \
  } while (0)
#endif

#ifndef PL_CUDNN_RETURN_IF_ERROR
#define PL_CUDNN_RETURN_IF_ERROR(cudnn_call_expr) \
  do {                                            \
    cudnnStatus_t _cudnn_err = (cudnn_call_expr); \
    if (_cudnn_err != CUDNN_STATUS_SUCCESS) {     \
      return Ort::GetApi().CreateStatus(          \
          ORT_EP_FAIL,                            \
          (std::string("cuDNN error: ") +         \
           cudnnGetErrorString(_cudnn_err))       \
              .c_str());                          \
    }                                             \
  } while (0)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
// C4702: unreachable code - the trailing return is required for ORT_NO_EXCEPTIONS builds
#define EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_BEGIN __pragma(warning(push)) __pragma(warning(disable : 4702))
#define EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_END __pragma(warning(pop))
#else
#define EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_BEGIN
#define EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_END
#endif

#define EXCEPTION_TO_STATUS_BEGIN EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_BEGIN ORT_TRY {
#define EXCEPTION_TO_STATUS_END                   \
  }                                               \
  ORT_CATCH(const Ort::Exception& ex) {           \
    OrtStatus* _ort_ex_st = nullptr;              \
    ORT_HANDLE_EXCEPTION([&]() {                  \
      Ort::Status status(ex);                     \
      _ort_ex_st = status.release();              \
    });                                           \
    return _ort_ex_st;                            \
  }                                               \
  ORT_CATCH(const std::exception& ex) {           \
    OrtStatus* _std_ex_st = nullptr;              \
    ORT_HANDLE_EXCEPTION([&]() {                  \
      Ort::Status status(ex.what(), ORT_EP_FAIL); \
      _std_ex_st = status.release();              \
    });                                           \
    return _std_ex_st;                            \
  }                                               \
  EXCEPTION_TO_STATUS_UNREACHABLE_GUARD_END

/// Stored API pointers accessible to all plugin components.
struct CudaPluginApis {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};
