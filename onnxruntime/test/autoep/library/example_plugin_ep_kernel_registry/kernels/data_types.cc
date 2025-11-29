// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "data_types.h"

MLDataTypes::MLDataTypes() {}

/*static*/
MLDataTypes& MLDataTypes::GetInstance() {
  static MLDataTypes instance;
  return instance;
}

/*static*/
OrtStatus* MLDataTypes::GetTensorType(ONNXTensorElementDataType elem_type, /*out*/ const OrtMLDataType*& tensor_type) {
  MLDataTypes& instance = GetInstance();
  const OrtEpApi& ep_api = Ort::GetEpApi();

  auto iter = instance.tensor_types_map_.find(elem_type);
  if (iter == instance.tensor_types_map_.end()) {
    const OrtMLDataType* type = nullptr;

    RETURN_IF_ERROR(ep_api.GetTensorMLDataType(elem_type, &type));
    instance.tensor_types_map_.emplace(elem_type, type);

    tensor_type = type;
    return nullptr;
  }

  tensor_type = iter->second;
  return nullptr;
}

/*static*/
const OrtMLDataType* MLDataTypes::GetTensorType(ONNXTensorElementDataType elem_type) {
  const OrtMLDataType* result = nullptr;
  Ort::ThrowOnError(MLDataTypes::GetTensorType(elem_type, result));
  return result;
}

/*static*/
OrtStatus* MLDataTypes::GetAllFixedSizeTensorTypesIRv9(/*out*/ std::vector<const OrtMLDataType*>& result) {
  MLDataTypes& instance = GetInstance();
  if (instance.fixed_tensor_v9_.empty()) {
    auto add_tensor_type = [&instance](ONNXTensorElementDataType elem_type) -> OrtStatus* {
      const OrtMLDataType* type = nullptr;

      RETURN_IF_ERROR(instance.GetTensorType(elem_type, type));
      instance.fixed_tensor_v9_.push_back(type);
      return nullptr;
    };

    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4));
    RETURN_IF_ERROR(add_tensor_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4));
  }

  result = instance.fixed_tensor_v9_;
  return nullptr;
}

/*static*/
std::vector<const OrtMLDataType*> MLDataTypes::GetAllFixedSizeTensorTypesIRv9() {
  std::vector<const OrtMLDataType*> result;
  Ort::ThrowOnError(GetInstance().GetAllFixedSizeTensorTypesIRv9(result));
  return result;
}
