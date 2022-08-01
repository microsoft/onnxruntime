// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include <variant>

#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {
namespace api {

typedef std::variant<int64_t, float, std::string> CheckPointPropertyDataType;

/**
 * @brief Class for user defined checkpoint property.
 */
struct CheckpointProperty {
 public:
  CheckpointProperty() {}
  CheckpointProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);
  CheckpointProperty(const std::string& prop_name, const CheckPointPropertyDataType& prop_value)
      : prop_name_(prop_name), prop_value_(prop_value) {
  }

  ONNX_NAMESPACE::TensorProto ToTensorProto();

  std::string GetName() const {
    return prop_name_;
  }

  CheckPointPropertyDataType GetData() const {
    return prop_value_;
  }

 protected:
  std::string prop_name_;
  CheckPointPropertyDataType prop_value_;
};

/**
 * @brief Collection of user defined properties.
 * Currently supported scalar value of type int64_t, float, and std::string only.
 */
struct PropertyBag {
 public:
  PropertyBag() {}

  void AddProperty(std::string name, CheckPointPropertyDataType val) {
    ORT_ENFORCE(named_properties_.find(name) == named_properties_.end(),
                "Duplicated property named ", name);

    named_properties_.insert({name, std::make_shared<CheckpointProperty>(name, val)});
  }

  void AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  template <typename T>
  T GetProperty(const std::string& name) const {
    auto it = named_properties_.find(name);
    ORT_ENFORCE(it != named_properties_.end(), "No property named ", name);

    CheckPointPropertyDataType cloned_val = it->second->GetData();
    T* tval = std::get_if<T>(&cloned_val);
    ORT_ENFORCE(tval, "Fail to parse the property value using specified type.");
    return *tval;
  }

  void ToTensorProtos(std::vector<ONNX_NAMESPACE::TensorProto>& properties_tensor_protos) const {
    for (auto it = named_properties_.begin(); it != named_properties_.end(); ++it) {
      properties_tensor_protos.emplace_back((it->second)->ToTensorProto());
    }
  }

  size_t Size() const {
    return named_properties_.size();
  }

 private:
  const std::vector<int32_t> supported_data_types{
      ONNX_NAMESPACE::TensorProto::FLOAT,
      ONNX_NAMESPACE::TensorProto::INT64,
      ONNX_NAMESPACE::TensorProto::STRING};

  bool IsSupportedDataType(int32_t data_type) const {
    return std::find(supported_data_types.begin(), supported_data_types.end(), data_type) != supported_data_types.end();
  }

  std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>> named_properties_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
