// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_enums_internal.h"

#include <algorithm>

#include "core/session/onnxruntime_cxx_api.h"

namespace {

struct LoggingLevelInfo {
  ORTLoggingLevel logging_level;
  OrtLoggingLevel capi_logging_level;
};

// supported ORT logging levels
// define the mapping from ORTLoggingLevel to C API OrtLoggingLevel here
constexpr LoggingLevelInfo kLoggingLevelInfos[]{
    {ORTLoggingLevelVerbose, ORT_LOGGING_LEVEL_VERBOSE},
    {ORTLoggingLevelInfo, ORT_LOGGING_LEVEL_INFO},
    {ORTLoggingLevelWarning, ORT_LOGGING_LEVEL_WARNING},
    {ORTLoggingLevelError, ORT_LOGGING_LEVEL_ERROR},
    {ORTLoggingLevelFatal, ORT_LOGGING_LEVEL_FATAL},
};

struct ValueTypeInfo {
  ORTValueType type;
  ONNXType capi_type;
};

// supported ORT value types
// define the mapping from ORTValueType to C API ONNXType here
constexpr ValueTypeInfo kValueTypeInfos[]{
    {ORTValueTypeUnknown, ONNX_TYPE_UNKNOWN},
    {ORTValueTypeTensor, ONNX_TYPE_TENSOR},
};

struct TensorElementTypeInfo {
  ORTTensorElementDataType type;
  ONNXTensorElementDataType capi_type;
  size_t element_size;
};

// supported ORT tensor element data types
// define the mapping from ORTTensorElementDataType to C API
// ONNXTensorElementDataType here
constexpr TensorElementTypeInfo kElementTypeInfos[]{
    {ORTTensorElementDataTypeUndefined, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0},
    {ORTTensorElementDataTypeFloat, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
    {ORTTensorElementDataTypeInt8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
    {ORTTensorElementDataTypeUInt8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
    {ORTTensorElementDataTypeInt32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
    {ORTTensorElementDataTypeUInt32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
};

template <typename Container, typename SelectFn, typename TransformFn>
auto SelectAndTransform(
    const Container& container, SelectFn select_fn, TransformFn transform_fn,
    const char* not_found_msg)
    -> decltype(transform_fn(*std::begin(container))) {
  const auto it = std::find_if(
      std::begin(container), std::end(container), select_fn);
  if (it == std::end(container)) {
    throw Ort::Exception{not_found_msg, ORT_NOT_IMPLEMENTED};
  }
  return transform_fn(*it);
}

}  // namespace

OrtLoggingLevel PublicToCAPILoggingLevel(ORTLoggingLevel logging_level) {
  return SelectAndTransform(
      kLoggingLevelInfos,
      [logging_level](const auto& logging_level_info) {
        return logging_level_info.logging_level == logging_level;
      },
      [](const auto& logging_level_info) {
        return logging_level_info.capi_logging_level;
      },
      "unsupported logging level");
}

ORTValueType CAPIToPublicValueType(ONNXType capi_type) {
  return SelectAndTransform(
      kValueTypeInfos,
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; },
      [](const auto& type_info) { return type_info.type; },
      "unsupported value type");
}

ONNXTensorElementDataType PublicToCAPITensorElementType(ORTTensorElementDataType type) {
  return SelectAndTransform(
      kElementTypeInfos,
      [type](const auto& type_info) { return type_info.type == type; },
      [](const auto& type_info) { return type_info.capi_type; },
      "unsupported tensor element type");
}

ORTTensorElementDataType CAPIToPublicTensorElementType(ONNXTensorElementDataType capi_type) {
  return SelectAndTransform(
      kElementTypeInfos,
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; },
      [](const auto& type_info) { return type_info.type; },
      "unsupported tensor element type");
}

size_t SizeOfCAPITensorElementType(ONNXTensorElementDataType capi_type) {
  return SelectAndTransform(
      kElementTypeInfos,
      [capi_type](const auto& type_info) { return type_info.capi_type == capi_type; },
      [](const auto& type_info) { return type_info.element_size; },
      "unsupported tensor element type");
}
