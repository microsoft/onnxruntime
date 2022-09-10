// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <type_traits>
#include <variant>

#include "core/common/inlined_containers.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {
namespace api {

using PropertyDataType = std::variant<int64_t, float, std::string>;

/**
 * @brief Collection of user defined properties.
 * Currently supported scalar value of type int64_t, float, and std::string only.
 */
struct PropertyBag {
 public:
  PropertyBag() = default;

  void AddProperty(const std::string& name, const PropertyDataType& val) {
    ORT_ENFORCE(named_properties_.find(name) == named_properties_.end(),
                "Duplicated property named ", name);

    named_properties_.insert({name, val});
  }

  void AddProperty(const ONNX_NAMESPACE::TensorProto& tensor_proto);

  template <typename T>
  T GetProperty(const std::string& name) const {
    auto it = named_properties_.find(name);
    ORT_ENFORCE(it != named_properties_.end(), "No property named ", name);

    const T* tval = std::get_if<T>(&it->second);
    ORT_ENFORCE(tval, "Fail to get the property value using specified type.");
    return *tval;
  }

  void ToTensorProtos(std::vector<ONNX_NAMESPACE::TensorProto>& properties_tensor_protos) const {
    for (auto it = named_properties_.begin(); it != named_properties_.end(); ++it) {
      onnx::TensorProto t_proto;
      if (const float* fval = std::get_if<float>(&it->second); fval != nullptr) {
        t_proto = ONNX_NAMESPACE::ToTensor<float>(*fval);
      } else if (const int64_t* ival = std::get_if<int64_t>(&it->second); ival != nullptr) {
        t_proto = ONNX_NAMESPACE::ToTensor<int64_t>(*ival);
      } else if (const std::string* sval = std::get_if<std::string>(&it->second); sval != nullptr) {
        t_proto = ONNX_NAMESPACE::ToTensor<std::string>(*sval);
      } else {
        ORT_THROW("Should not go there, unexpected data_type for prop value.");
      }
      t_proto.set_name(it->first);
      properties_tensor_protos.emplace_back(t_proto);
    }
  }

  size_t Size() const {
    return named_properties_.size();
  }

 private:
  const InlinedVector<int32_t> supported_data_types{
      ONNX_NAMESPACE::TensorProto::FLOAT,
      ONNX_NAMESPACE::TensorProto::INT64,
      ONNX_NAMESPACE::TensorProto::STRING};

  bool IsSupportedDataType(int32_t data_type) const {
    return std::find(supported_data_types.begin(), supported_data_types.end(), data_type) != supported_data_types.end();
  }

  InlinedHashMap<std::string, PropertyDataType> named_properties_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
