
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/initializer_view/initializer_view.h"

namespace onnxruntime {
std::unique_ptr<InitializerView> InitializerView::Create(
    const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto initializer = std::unique_ptr<InitializerView>(new InitializerView);

  auto proto_data_type = tensor_proto.data_type();
  auto proto_dims = utils::GetTensorShapeFromTensorProto(tensor_proto);
  initializer->shape_ = TensorShape(proto_dims);
  initializer->type_ = DataTypeImpl::TensorTypeFromONNXEnum(proto_data_type)->GetElementType();

  Path external_path;
  auto status = utils::UnpackInitializerData(tensor_proto, external_path, initializer->unpacked_tensor_);
  if (!status.IsOK()) {
    return {};
  }
  return initializer;
}
}  // namespace onnxruntime
