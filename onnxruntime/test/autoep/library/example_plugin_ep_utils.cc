// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "example_plugin_ep_utils.h"

#include <string>

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api, const OrtSessionOptions& session_options,
                                          const char* config_key, const std::string& default_val,
                                          /*out*/ std::string& config_val) {
  int has_config = 0;
  RETURN_IF_ERROR(ort_api.HasSessionConfigEntry(&session_options, config_key, &has_config));

  if (has_config != 1) {
    config_val = default_val;
    return nullptr;
  }

  size_t size = 0;
  RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key, nullptr, &size));

  config_val.resize(size);
  RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key, config_val.data(), &size));
  config_val.resize(size - 1);  // remove the terminating '\0'

  return nullptr;
}

OrtStatus* IsFloatTensor(const OrtApi& ort_api, const OrtValueInfo* value_info, bool& result) {
  result = false;

  const OrtTypeInfo* type_info = nullptr;
  RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(value_info, &type_info));

  ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
  RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(type_info, &onnx_type));
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return nullptr;
  }

  const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
  RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape));

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape, &elem_type));
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return nullptr;
  }

  result = true;
  return nullptr;
}
