// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

namespace onnxruntime {
namespace signal {

template <typename T>
static T get_scalar_value_from_tensor(const Tensor* tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "ratio input should have a single value.");

  auto data_type = tensor->DataType()->AsPrimitiveDataType()->GetDataType();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return static_cast<T>(*reinterpret_cast<const float*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return static_cast<T>(*reinterpret_cast<const double*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return static_cast<T>(*reinterpret_cast<const int32_t*>(tensor->DataRaw()));
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return static_cast<T>(*reinterpret_cast<const int64_t*>(tensor->DataRaw()));
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

}  // namespace signal
}  // namespace onnxruntime
