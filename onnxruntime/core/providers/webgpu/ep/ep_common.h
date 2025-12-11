// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

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

// Ignores an OrtStatus* while taking ownership of it so that it does not get leaked.
#define IGNORE_ORTSTATUS(status_expr)   \
  do {                                  \
    OrtStatus* _status = (status_expr); \
    Ort::Status _ignored{_status};      \
  } while (false)

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define LOG(level, ...)                                                                            \
  do {                                                                                             \
    std::ostringstream ss;                                                                         \
    ss << __VA_ARGS__;                                                                             \
    IGNORE_ORTSTATUS(api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_##level, ss.str().c_str(), \
                                            EP_FILE, __LINE__, __FUNCTION__));                     \
  } while (false)

#define RETURN_ERROR(code, ...)                       \
  do {                                                \
    std::ostringstream ss;                            \
    ss << __VA_ARGS__;                                \
    return api_.CreateStatus(code, ss.str().c_str()); \
  } while (false)

#define THROW(...)       \
  std::ostringstream ss; \
  ss << __VA_ARGS__;     \
  throw std::runtime_error(ss.str())

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

struct FloatInitializer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// Returns an entry in the session option configurations, or a default value if not present.
inline OrtStatus* GetSessionConfigEntryOrDefault(const OrtSessionOptions& session_options,
                                                 const char* config_key, const std::string& default_val,
                                                 /*out*/ std::string& config_val) {
  try {
    Ort::ConstSessionOptions sess_opt{&session_options};
    config_val = sess_opt.GetConfigEntryOrDefault(config_key, default_val);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }

  return nullptr;
}

// Returns true (via output parameter) if the given OrtValueInfo represents a float tensor.
inline void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
  result = false;

  auto type_info = value_info.TypeInfo();
  ONNXType onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return;
  }
  result = true;
}

// Gets the tensor shape from `value_info`. Returns std::nullopt if `value_info` is not a tensor.
inline std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetShape();
}
