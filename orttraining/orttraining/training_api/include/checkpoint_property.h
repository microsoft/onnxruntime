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
 * Only support int64_t, std::string and float data types.
 */
template <typename T>
struct TypedCheckpointProperty : public CheckpointProperty {
 public:
  TypedCheckpointProperty(const std::string& prop_name, const T& prop_value)
      : CheckpointProperty(prop_name), prop_value_(prop_value) {
    // Align the data type support with  constructor from TensorProto.
    if (!(std::is_same<T, float>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, std::string>::value)) {
      ORT_THROW("Not supported data type, only support float, int64_t and std::string.");
    }
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
 * Supported scalar value of type int64_t, float, and std::string only.
 */
struct PropertyBag {
  template <typename T>
  void AddProperty(std::string name, T val) {
    named_properties.insert({name, std::make_shared<TypedCheckpointProperty<T>>(name, val)});
  }

  void AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  template <typename T>
  T GetProperty(const std::string& name) const {
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
  std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>> named_properties;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
