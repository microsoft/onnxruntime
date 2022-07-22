// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx/defs/tensor_proto_util.h"
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "core/framework/tensorprotoutils.h"
#include "orttraining/training_api/include/checkpoint_property.h"

namespace onnxruntime {
namespace training {
namespace api {

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
  ORT_THROW_IF_ERROR(utils::UnpackTensor<T>(tensor_proto, model_path, p, expected_num_elements));
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

std::shared_ptr<CheckpointProperty> CreateCheckpointPropertyFromTensorProto(
    const ONNX_NAMESPACE::TensorProto& tensor_proto) {
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
  ORT_ENFORCE(named_properties_.find(tensor_proto.name()) == named_properties_.end(),
              "Duplicated property named ", tensor_proto.name());

  if (!IsSupportedDataType(tensor_proto.data_type())) {
    ORT_THROW("Failed to add property from tensorproto: float, int64_t and std::string data types supported only.");
  }

  named_properties_.insert({tensor_proto.name(), CreateCheckpointPropertyFromTensorProto(tensor_proto)});
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
