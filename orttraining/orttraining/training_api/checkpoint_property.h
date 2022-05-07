// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "onnx/defs/tensor_proto_util.h"
#include "orttraining/training_api/interfaces.h"

namespace onnxruntime {
namespace training {

// TODO: Rename to api after all major classes implemented.
namespace api_test {

template <typename T>
struct TypedCheckpointProperty;

/**
 * @brief Base class for user defined checkpoint property.
 */
struct CheckpointProperty {
 public:
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
  }

  ONNX_NAMESPACE::TensorProto ToTensorProto() override {
    auto t_proto = ONNX_NAMESPACE::ToTensor<T>(prop_value_);
    t_proto.set_name(prop_name_);
    return t_proto;
  }

  T GetData() const {
    return prop_value_;
  }

 private:
  T prop_value_;
};

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime
