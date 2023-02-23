// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"

namespace onnxruntime {
namespace signal {

template <typename T>
static T get_scalar_value_from_tensor(const Tensor* tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "ratio input should have a single value.");
  const auto data_type = tensor->GetElementType();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return static_cast<T>(*tensor->Data<float>());
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return static_cast<T>(*tensor->Data<double>());
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return static_cast<T>(*tensor->Data<int32_t>());
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return static_cast<T>(*tensor->Data<int64_t>());
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

}  // namespace signal
}  // namespace onnxruntime
