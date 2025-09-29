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
OrtStatus* MLDataTypes::AllFixedSizeTensorTypesIRv9(/*out*/ std::vector<const OrtMLDataType*>& result) {
  MLDataTypes& instance = GetInstance();
  const OrtEpApi& ep_api = Ort::GetEpApi();

  if (instance.fixed_tensor_v9_.empty()) {
    auto add_tensor_type = [&instance, &ep_api](ONNXTensorElementDataType elem_type) -> OrtStatus* {
      const OrtMLDataType* tensor_ml_type = nullptr;
      RETURN_IF_ERROR(ep_api.GetTensorMLDataType(elem_type, &tensor_ml_type));
      instance.fixed_tensor_v9_.push_back(tensor_ml_type);
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
