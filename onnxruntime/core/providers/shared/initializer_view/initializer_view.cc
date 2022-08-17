
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/initializer_view/initializer_view.h"
#include <utility>

namespace onnxruntime {
common::Status InitializerView::Create(
    const ONNX_NAMESPACE::TensorProto& tensor_proto, std::optional<InitializerView>& initializer) {
  initializer.emplace();  // create instance in place

  initializer->data_type_ = tensor_proto.data_type();
  auto proto_dims = utils::GetTensorShapeFromTensorProto(tensor_proto);
  initializer->shape_ = TensorShape(proto_dims);

  Path external_path;
  return utils::UnpackInitializerData(tensor_proto, external_path, initializer->unpacked_tensor_);
}
}  // namespace onnxruntime
