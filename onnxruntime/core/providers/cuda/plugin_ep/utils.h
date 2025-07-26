// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once

#include <sstream>
#include <stdexcept>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace cuda_plugin_ep {

constexpr uint32_t NvidiaVendorId = 0x10DE;

// these values are all valid once the ORT calls CreateEpFactories.
// do not use in something statically initialized that may be created prior to this.
struct Shared {
  static const OrtLogger* default_logger;
  static const OrtApi* ort_api;
  static const OrtEpApi* ep_api;
};

template <typename ERRTYPE, bool THROW_ON_ERROR, typename SUCCTYPE = ERRTYPE>
std::conditional_t<THROW_ON_ERROR, void, OrtStatus*> CudaCall(
    ERRTYPE retCode, const char* exprString, const char* libName, SUCCTYPE successCode, const char* msg,
    const char* file, const int line);

#define CUDA_CALL(expr) (CudaCall<cudaError, false>((expr), #expr, "CUDA", cudaSuccess, "", __FILE__, __LINE__))
#define CUBLAS_CALL(expr) (CudaCall<cublasStatus_t, false>((expr), #expr, "CUBLAS", CUBLAS_STATUS_SUCCESS, "", \
                                                           __FILE__, __LINE__))
#define CUDNN_CALL(expr) (CudaCall<cudnnStatus_t, false>((expr), #expr, "CUDNN", CUDNN_STATUS_SUCCESS, "", \
                                                         __FILE__, __LINE__))

// see ORT_ENFORCE for implementations that also capture a stack trace and work in builds with exceptions disabled
// NOTE: In this simplistic implementation you must provide an argument, even it if's an empty string
#define EP_ENFORCE(condition, ...)                       \
  do {                                                   \
    if (!(condition)) {                                  \
      std::ostringstream oss;                            \
      oss << "EP_ENFORCE failed: " << #condition << " "; \
      oss << __VA_ARGS__;                                \
      throw std::runtime_error(oss.str());               \
    }                                                    \
  } while (false)

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF_STATUS_NOTOK(fn)                                                  \
  do {                                                                              \
    onnxruntime::Status _status = (fn);                                             \
    if (!_status.IsOK()) {                                                          \
      return Shared::ort_api.CreateStatus(static_cast<OrtErrorCode>(status.Code()), \
                                          status.ErrorMessage().c_str());           \
    }                                                                               \
  } while (0)

#define THROW_IF_ERROR(fn)                                                 \
  do {                                                                     \
    OrtStatus* _status = (fn);                                             \
    if (_status != nullptr) {                                              \
      throw std::runtime_error(Shared::ort_api->GetErrorMessage(_status)); \
    }                                                                      \
  } while (0)

#define CUDA_RETURN_IF_ERROR(expr) RETURN_IF_ERROR(CUDA_CALL(expr))
#define CUBLAS_RETURN_IF_ERROR(expr) RETURN_IF_ERROR(CUBLAS_CALL(expr))
#define CUDNN_RETURN_IF_ERROR(expr) RETURN_IF_ERROR(CUDNN_CALL(expr))

#define CUDA_THROW_IF_ERROR(expr) THROW_IF_ERROR(CUDA_CALL(expr))
#define CUBLAS_THROW_IF_ERROR(expr) THROW_IF_ERROR(CUBLAS_CALL(expr))
#define CUDNN_THROW_IF_ERROR(expr) THROW_IF_ERROR(CUDNN_CALL(expr))

#define LOG_DEFAULT(level, msg) \
  Shared::ort_api->Logger_LogMessage(Shared::default_logger, level, msg, __FILE__, __LINE__)

}  // namespace cuda_plugin_ep
