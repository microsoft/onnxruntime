// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "onnx/defs/tensor_proto_util.h"
#include "orttraining/training_api/interfaces.h"
#include "orttraining/training_api/checkpoint_property.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace training {

// TODO: Rename to api after all major classes implemented.
namespace api_test {

template <typename T>
TypedCheckpointProperty<T>::TypedCheckpointProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  std::vector<int64_t> tensor_shape_vec = utils::GetTensorShapeFromTensorProto(tensor_proto);
  int64_t expected_num_elements = 1;
  for (auto& d : tensor_shape_vec) {
    expected_num_elements *= d;
  }
  ORT_ENFORCE(expected_num_elements == 1, "Only scalar value support for checkpoint property.");
  Path model_path;
  std::vector<T> data_vector(1);
  T* p = data_vector.data();
  ORT_ENFORCE(utils::UnpackTensor<T>(tensor_proto, model_path, p, expected_num_elements).IsOK());
  prop_name_ = tensor_proto.name();
  prop_value_ = data_vector[0];
}

template <typename T>
ONNX_NAMESPACE::TensorProto TypedCheckpointProperty<T>::ToTensorProto() {
  auto t_proto = ONNX_NAMESPACE::ToTensor<T>(prop_value_);
  t_proto.set_name(prop_name_);
  return t_proto;
}

namespace {
std::shared_ptr<CheckpointProperty> CreateCheckpointPropertyFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto data_type = tensor_proto.data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT: {
      return std::static_pointer_cast<CheckpointProperty>(
          std::make_shared<TypedCheckpointProperty<float>>(tensor_proto));
      break;
    }
    case ONNX_NAMESPACE::TensorProto::STRING: {
      return std::static_pointer_cast<CheckpointProperty>(
          std::make_shared<TypedCheckpointProperty<std::string>>(tensor_proto));
      break;
    }
    case ONNX_NAMESPACE::TensorProto::INT64: {
      return std::static_pointer_cast<CheckpointProperty>(
          std::make_shared<TypedCheckpointProperty<int64_t>>(tensor_proto));
      break;
    }
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}
}  // namespace

void PropertyBag::AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  named_properties.insert({tensor_proto.name(), CreateCheckpointPropertyFromTensorProto(tensor_proto)});
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime
