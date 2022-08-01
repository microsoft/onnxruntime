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
void ParsePropertyFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                  std::string& name,
                                  CheckPointPropertyDataType& value) {
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
  name = tensor_proto.name();
  value = data_vector[0];
}

CheckpointProperty::CheckpointProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  auto data_type = tensor_proto.data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT: {
      ParsePropertyFromTensorProto<float>(tensor_proto, prop_name_, prop_value_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto::STRING: {
      ParsePropertyFromTensorProto<std::string>(tensor_proto, prop_name_, prop_value_);
      break;
    }
    case ONNX_NAMESPACE::TensorProto::INT64: {
      ParsePropertyFromTensorProto<int64_t>(tensor_proto, prop_name_, prop_value_);
      break;
    }
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

ONNX_NAMESPACE::TensorProto CheckpointProperty::ToTensorProto() {
  onnx::TensorProto t_proto;
  if (std::holds_alternative<float>(prop_value_)) {
    float* fval = std::get_if<float>(&prop_value_);
    ORT_ENFORCE(fval, "Fail to parse the property value using float type.");
    t_proto = ONNX_NAMESPACE::ToTensor<float>(*fval);
  } else if (std::holds_alternative<int64_t>(prop_value_)) {
    int64_t* ival = std::get_if<int64_t>(&prop_value_);
    ORT_ENFORCE(ival, "Fail to parse the property value using int64_t type.");
    t_proto = ONNX_NAMESPACE::ToTensor<int64_t>(*ival);
  } else if (std::holds_alternative<std::string>(prop_value_)) {
    std::string* sval = std::get_if<std::string>(&prop_value_);
    ORT_ENFORCE(sval, "Fail to parse the property value using std::string type.");
    t_proto = ONNX_NAMESPACE::ToTensor<std::string>(*sval);
  } else {
    ORT_THROW("Should not go there, unexpected data_type for prop value.");
  }

  t_proto.set_name(prop_name_);
  return t_proto;
}

void PropertyBag::AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  ORT_ENFORCE(named_properties_.find(tensor_proto.name()) == named_properties_.end(),
              "Duplicated property named ", tensor_proto.name());

  if (!IsSupportedDataType(tensor_proto.data_type())) {
    ORT_THROW("Failed to add property from tensorproto: float, int64_t and std::string data types supported only.");
  }

  named_properties_.insert({tensor_proto.name(), std::make_shared<CheckpointProperty>(tensor_proto)});
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
