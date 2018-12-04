// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

//TODO: encode error code in the message?
#define ONNXRUNTIME_THROW_ON_ERROR(expr)                                                \
  do {                                                                                  \
    ONNXStatus* onnx_status = (expr);                                                   \
    if (onnx_status != nullptr) {                                                       \
      std::string onnx_runtime_error_message = ONNXRuntimeGetErrorMessage(onnx_status); \
      ReleaseONNXStatus(onnx_status);                                                   \
      throw std::runtime_error(onnx_runtime_error_message);                             \
    }                                                                                   \
  } while (0);

#define ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(NAME) \
  decltype(ONNXRuntime##NAME(value.get())) NAME() {     \
    return ONNXRuntime##NAME(value.get());              \
  }

#define DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(TYPE_NAME)  \
  namespace std {                                           \
  template <>                                               \
  struct default_delete<ONNXRuntime##TYPE_NAME> {           \
    void operator()(ONNXRuntime##TYPE_NAME* ptr) {          \
      (*reinterpret_cast<ONNXObject**>(ptr))->Release(ptr); \
    }                                                       \
  };                                                        \
  }

DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(Env);
DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(TypeInfo);
DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(Allocator);
DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(TensorTypeAndShapeInfo);
DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(RunOptions);
DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT(SessionOptions);

#undef DECLARE_DEFAULT_DELETER_FOR_ONNX_OBJECT

namespace onnxruntime {
class SessionOptionsWrapper {
 private:
  std::unique_ptr<ONNXRuntimeSessionOptions, decltype(&ONNXRuntimeReleaseObject)> value;
  ONNXRuntimeEnv* env_;
  SessionOptionsWrapper(_In_ ONNXRuntimeEnv* env, ONNXRuntimeSessionOptions* p) : value(p, ONNXRuntimeReleaseObject), env_(env){};

 public:
  //TODO: for the input arg, should we call addref here?
  SessionOptionsWrapper(_In_ ONNXRuntimeEnv* env) : value(ONNXRuntimeCreateSessionOptions(), ONNXRuntimeReleaseObject), env_(env){};
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(EnableSequentialExecution)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(DisableSequentialExecution)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(DisableProfiling)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(EnableMemPattern)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(DisableMemPattern)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(EnableCpuMemArena)
  ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL(DisableCpuMemArena)
  void EnableProfiling(_In_ const char* profile_file_prefix) {
    ONNXRuntimeEnableProfiling(value.get(), profile_file_prefix);
  }

  void SetSessionLogId(const char* logid) {
    ONNXRuntimeSetSessionLogId(value.get(), logid);
  }
  void SetSessionLogVerbosityLevel(uint32_t session_log_verbosity_level) {
    ONNXRuntimeSetSessionLogVerbosityLevel(value.get(), session_log_verbosity_level);
  }
  void SetSessionThreadPoolSize(int session_thread_pool_size) {
    ONNXRuntimeSetSessionThreadPoolSize(value.get(), session_thread_pool_size);
  }

  /**
  * The order of invocation indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
  */
  void AppendExecutionProvider(_In_ ONNXRuntimeProviderFactoryInterface** f) {
    ONNXRuntimeSessionOptionsAppendExecutionProvider(value.get(), f);
  }

  SessionOptionsWrapper clone() const {
    ONNXRuntimeSessionOptions* p = ONNXRuntimeCloneSessionOptions(value.get());
    return SessionOptionsWrapper(env_, p);
  }
#ifdef _WIN32
  ONNXSession* ONNXRuntimeCreateInferenceSession(_In_ const wchar_t* model_path) {
    ONNXSession* ret;
    ONNXRUNTIME_THROW_ON_ERROR(::ONNXRuntimeCreateInferenceSession(env_, model_path, value.get(), &ret));
    return ret;
  }
#else
  ONNXSession* ONNXRuntimeCreateInferenceSession(_In_ const char* model_path) {
    ONNXSession* ret;
    ONNXRUNTIME_THROW_ON_ERROR(::ONNXRuntimeCreateInferenceSession(env_, model_path, value.get(), &ret));
    return ret;
  }
#endif
  void AddCustomOp(_In_ const char* custom_op_path) {
    ONNXRuntimeAddCustomOp(value.get(), custom_op_path);
  }
};
inline ONNXValue* ONNXRuntimeCreateTensorAsONNXValue(_Inout_ ONNXRuntimeAllocator* env, const std::vector<size_t>& shape, OnnxRuntimeTensorElementDataType type) {
  ONNXValue* ret;
  ONNXRUNTIME_THROW_ON_ERROR(::ONNXRuntimeCreateTensorAsONNXValue(env, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline ONNXValue* ONNXRuntimeCreateTensorWithDataAsONNXValue(_In_ const ONNXRuntimeAllocatorInfo* info, _In_ void* p_data, size_t p_data_len, const std::vector<size_t>& shape, OnnxRuntimeTensorElementDataType type) {
  ONNXValue* ret;
  ONNXRUNTIME_THROW_ON_ERROR(::ONNXRuntimeCreateTensorWithDataAsONNXValue(info, p_data, p_data_len, shape.data(), shape.size(), type, &ret));
  return ret;
}

inline std::vector<int64_t> GetTensorShape(const ONNXRuntimeTensorTypeAndShapeInfo* info) {
  size_t dims = ONNXRuntimeGetNumOfDimensions(info);
  std::vector<int64_t> ret(dims);
  ONNXRuntimeGetDimensions(info, ret.data(), ret.size());
  return ret;
}
}  // namespace onnxruntime

#undef ONNXRUNTIME_REDIRECT_SIMPLE_FUNCTION_CALL
