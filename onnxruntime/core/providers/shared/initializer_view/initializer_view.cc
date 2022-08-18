
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/initializer_view/initializer_view.h"
#include <utility>

namespace onnxruntime {

InitializerView::InitializerView(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  ORT_THROW_IF_ERROR(Create(tensor_proto));
}

common::Status InitializerView::Create(
    const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  dtype_ = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())
               ->GetElementType()
               ->AsPrimitiveDataType();
  shape_ = TensorShape(utils::GetTensorShapeFromTensorProto(tensor_proto));

  Path external_path;
  return utils::UnpackInitializerData(tensor_proto, external_path, unpacked_tensor_);
}
}  // namespace onnxruntime
