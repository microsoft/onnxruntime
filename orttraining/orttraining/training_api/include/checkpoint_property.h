// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {
namespace api {

template <typename T>
struct TypedCheckpointProperty;

/**
 * @brief Base class for user defined checkpoint property.
 */
struct CheckpointProperty {
 public:
  CheckpointProperty() {}

  CheckpointProperty(const std::string& prop_name)
      : prop_name_(prop_name) {
  }

  virtual ~CheckpointProperty() {}
  virtual ONNX_NAMESPACE::TensorProto ToTensorProto() = 0;

  std::string GetName() const {
    return prop_name_;
  }

  template <typename T>
  T GetData() {
    auto ptr = dynamic_cast<TypedCheckpointProperty<T>*>(this);
    ORT_ENFORCE(ptr);
    return ptr->GetData();
  }

 protected:
  std::string prop_name_;
};

/**
 * @brief User defined checkpoint property.
 */
template <typename T>
struct TypedCheckpointProperty : public CheckpointProperty {
 public:
  TypedCheckpointProperty(const std::string& prop_name, const T& prop_value)
      : CheckpointProperty(prop_name), prop_value_(prop_value) {
  }
  TypedCheckpointProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  ONNX_NAMESPACE::TensorProto ToTensorProto() override;

  T GetData() const {
    return prop_value_;
  }

 private:
  T prop_value_;
};

/**
 * @brief Collection of user defined properties.
 * Currently supported scalar value of type int64_t, float, and std::string only.
 */
struct PropertyBag {
 public:
  PropertyBag() {}

  template <typename T>
  void AddProperty(std::string name, T val) {
    ORT_ENFORCE(named_properties.find(name) == named_properties.end(),
                "Duplicated property named ", name);

    if (!IsSupportedDataType<T>()) {
      ORT_THROW("Failed to add property: float, int64_t and std::string data types supported only.");
    }

    named_properties.insert({name, std::make_shared<TypedCheckpointProperty<T>>(name, val)});
  }

  void AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  template <typename T>
  T GetProperty(const std::string& name) const {
    if (!IsSupportedDataType<T>()) {
      ORT_THROW("Failed to get property: float, int64_t and std::string data types supported only.");
    }

    auto it = named_properties.find(name);
    ORT_ENFORCE(it != named_properties.end(), "No property named ", name);
    return it->second->GetData<T>();
  }

  void ToTensorProtos(std::vector<ONNX_NAMESPACE::TensorProto>& properties_tensor_protos) const {
    for (auto it = named_properties.begin(); it != named_properties.end(); ++it) {
      properties_tensor_protos.emplace_back((it->second)->ToTensorProto());
    }
  }

  int Size() const {
    return named_properties.size();
  }

 private:
  const std::vector<int32_t> supported_data_types{
      ONNX_NAMESPACE::TensorProto::FLOAT,
      ONNX_NAMESPACE::TensorProto::INT64,
      ONNX_NAMESPACE::TensorProto::STRING};

  bool IsSupportedDataType(int32_t data_type) const {
    return std::find(supported_data_types.begin(), supported_data_types.end(), data_type) != supported_data_types.end();
  }

  template <typename T>
  bool IsSupportedDataType() const {
    return (std::is_same<T, float>::value || std::is_same<T, int64_t>::value ||
            std::is_same<T, std::string>::value);
  }

  std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>> named_properties;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
