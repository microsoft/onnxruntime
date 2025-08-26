// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "example_plugin_ep_utils.h"

#include <string>

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& /* ort_api */, const OrtSessionOptions& session_options,
                                          const char* config_key, const std::string& default_val,
                                          /*out*/ std::string& config_val) {
  try {
    Ort::ConstSessionOptions sess_opt{&session_options};
    bool has_config = sess_opt.HasConfigEntry(config_key);

    if (!has_config) {
      config_val = default_val;
      return nullptr;
    }

    config_val = sess_opt.GetConfigEntry(config_key);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex.what(), ex.GetOrtErrorCode());
    return status.release();
  }

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
