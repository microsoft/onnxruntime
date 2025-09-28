// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "example_plugin_ep_utils.h"

#include <gsl/gsl>
#include <string>

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& /* ort_api */, const OrtSessionOptions& session_options,
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

void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
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

std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetShape();
}

void GetKernelInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                                /*out*/ gsl::span<const float>& data,
                                /*out*/ std::vector<int64_t>& shape) {
  Ort::ConstValue input = kernel_context.GetInput(index);
  auto type_shape = input.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw Ort::Exception("EP Expected float32 inputs", ORT_EP_FAIL);

  const float* float_data = input.GetTensorData<float>();
  size_t num_elems = type_shape.GetElementCount();
  data = gsl::span<const float>(float_data, num_elems);
  shape = type_shape.GetShape();
}
