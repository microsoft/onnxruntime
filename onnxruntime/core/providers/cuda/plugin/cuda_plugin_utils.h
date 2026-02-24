// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cublasLt.h>

// Error handling macros

#define RETURN_IF_ERROR(expr)               \
  do {                                      \
    OrtStatus* _status = (expr);            \
    if (_status != nullptr) return _status; \
  } while (0)

#define RETURN_IF(condition, ort_api, message)             \
  do {                                                     \
    if (condition) {                                       \
      return (ort_api).CreateStatus(ORT_EP_FAIL, message); \
    }                                                      \
  } while (0)

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

#ifndef PL_CUBLAS_RETURN_IF_ERROR
#define PL_CUBLAS_RETURN_IF_ERROR(cublas_call_expr)       \
  do {                                                    \
    cublasStatus_t _cublas_err = (cublas_call_expr);      \
    if (_cublas_err != CUBLAS_STATUS_SUCCESS) {           \
      return Ort::GetApi().CreateStatus(                  \
          ORT_EP_FAIL,                                    \
          (std::string("cuBLAS error: ") +                \
           std::to_string(static_cast<int>(_cublas_err))) \
              .c_str());                                  \
    }                                                     \
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

#define EXCEPTION_TO_STATUS_BEGIN try {
#define EXCEPTION_TO_STATUS_END                 \
  }                                             \
  catch (const Ort::Exception& ex) {            \
    Ort::Status status(ex);                     \
    return status.release();                    \
  }                                             \
  catch (const std::exception& ex) {            \
    Ort::Status status(ex.what(), ORT_EP_FAIL); \
    return status.release();                    \
  }

/// Stored API pointers accessible to all plugin components.
struct CudaPluginApis {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};
