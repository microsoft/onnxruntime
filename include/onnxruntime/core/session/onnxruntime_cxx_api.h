// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
#include "core/common/exceptions.h"

//TODO: encode error code in the message?
#define ORT_THROW_ON_ERROR(expr)                                                 \
  do {                                                                           \
    OrtStatus* onnx_status = (expr);                                             \
    if (onnx_status != nullptr) {                                                \
      std::string ort_error_message = OrtGetErrorMessage(onnx_status);           \
      OrtErrorCode error_code = OrtGetErrorCode(onnx_status);                    \
      OrtReleaseStatus(onnx_status);                                             \
      switch (error_code) {                                                      \
        case ORT_NOT_IMPLEMENTED:                                                \
          throw onnxruntime::NotImplementedException(ort_error_message);         \
        default:                                                                 \
          throw onnxruntime::OnnxRuntimeException(ORT_WHERE, ort_error_message); \
      }                                                                          \
    }                                                                            \
  } while (0);

#define ORT_REDIRECT_SIMPLE_FUNCTION_CALL(NAME) \
  decltype(Ort##NAME(value.get())) NAME() {     \
    return Ort##NAME(value.get());              \
  }

namespace std {
template <>
struct default_delete<OrtAllocator> {
  void operator()(OrtAllocator* ptr) {
    OrtReleaseAllocator(ptr);
  }
};

template <>
struct default_delete<OrtEnv> {
  void operator()(OrtEnv* ptr) {
    OrtReleaseEnv(ptr);
  }
};

template <>
struct default_delete<OrtRunOptions> {
  void operator()(OrtRunOptions* ptr) {
    OrtReleaseRunOptions(ptr);
  }
};

template <>
struct default_delete<OrtTypeInfo> {
  void operator()(OrtTypeInfo* ptr) {
    OrtReleaseTypeInfo(ptr);
  }
};

template <>
struct default_delete<OrtTensorTypeAndShapeInfo> {
  void operator()(OrtTensorTypeAndShapeInfo* ptr) {
    OrtReleaseTensorTypeAndShapeInfo(ptr);
  }
};

template <>
struct default_delete<OrtSessionOptions> {
  void operator()(OrtSessionOptions* ptr) {
    OrtReleaseSessionOptions(ptr);
  }
};
}  // namespace std

namespace onnxruntime {
class SessionOptionsWrapper {
 private:
  std::unique_ptr<OrtSessionOptions> value;
  OrtEnv* env_;
  SessionOptionsWrapper(_In_ OrtEnv* env, OrtSessionOptions* p) : value(p), env_(env){};

 public:
  operator OrtSessionOptions*() { return value.get(); }

  //TODO: for the input arg, should we call addref here?
  SessionOptionsWrapper(_In_ OrtEnv* env) : value(OrtCreateSessionOptions()), env_(env){};
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableSequentialExecution)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableSequentialExecution)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableProfiling)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableMemPattern)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableMemPattern)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(EnableCpuMemArena)
  ORT_REDIRECT_SIMPLE_FUNCTION_CALL(DisableCpuMemArena)
  void EnableProfiling(_In_ const char* profile_file_prefix) {
    OrtEnableProfiling(value.get(), profile_file_prefix);
  }

  void SetSessionLogId(const char* logid) {
    OrtSetSessionLogId(value.get(), logid);
  }
  void SetSessionLogVerbosityLevel(uint32_t session_log_verbosity_level) {
    OrtSetSessionLogVerbosityLevel(value.get(), session_log_verbosity_level);
  }
  void SetSessionThreadPoolSize(int session_thread_pool_size) {
    OrtSetSessionThreadPoolSize(value.get(), session_thread_pool_size);
  }

  SessionOptionsWrapper clone() const {
    OrtSessionOptions* p = OrtCloneSessionOptions(value.get());
    return SessionOptionsWrapper(env_, p);
  }
#ifdef _WIN32
  OrtSession* OrtCreateSession(_In_ const wchar_t* model_path) {
    OrtSession* ret;
    ORT_THROW_ON_ERROR(::OrtCreateSession(env_, model_path, value.get(), &ret));
    return ret;
  }
#else
  OrtSession* OrtCreateSession(_In_ const char* model_path) {
    OrtSession* ret;
    ORT_THROW_ON_ERROR(::OrtCreateSession(env_, model_path, value.get(), &ret));
    return ret;
  }
#endif
  void AppendCustomOpLibPath(_In_ const char* lib_path) {
    OrtAppendCustomOpLibPath(value.get(), lib_path);
  }
};
inline OrtValue* OrtCreateTensorAsOrtValue(_Inout_ OrtAllocator* env, const std::vector<size_t>& shape, ONNXTensorElementDataType type) {
  OrtValue* ret;
  ORT_THROW_ON_ERROR(::OrtCreateTensorAsOrtValue(env, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline OrtValue* OrtCreateTensorWithDataAsOrtValue(_In_ const OrtAllocatorInfo* info, _In_ void* p_data, size_t p_data_len, const std::vector<size_t>& shape, ONNXTensorElementDataType type) {
  OrtValue* ret;
  ORT_THROW_ON_ERROR(::OrtCreateTensorWithDataAsOrtValue(info, p_data, p_data_len, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
  size_t dims = OrtGetNumOfDimensions(info);
  std::vector<int64_t> ret(dims);
  OrtGetDimensions(info, ret.data(), ret.size());
  return ret;
}
}  // namespace onnxruntime

#undef ORT_REDIRECT_SIMPLE_FUNCTION_CALL
