// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <type_traits>
#include <variant>

#include "core/common/inlined_containers.h"

namespace onnxruntime::training::api {

using PropertyDataType = std::variant<int64_t, float, std::string>;

/**
 * @brief Collection of user defined properties.
 * Currently supported scalar value of type int64_t, float, and std::string only.
 */
struct PropertyBag {
 public:
  PropertyBag() = default;

  void AddProperty(const std::string& name, const PropertyDataType& val) {
    auto it = named_properties_.find(name);
    if (it == named_properties_.end()) {
      named_properties_.insert({name, val});
    } else {
      it->second = val;
    }
  }

  template <typename T>
  T GetProperty(const std::string& name) const {
    auto it = named_properties_.find(name);
    ORT_ENFORCE(it != named_properties_.end(), "No property named ", name);

    const T* tval = std::get_if<T>(&it->second);
    ORT_ENFORCE(tval, "Fail to get the property value using specified type.");
    return *tval;
  }

  auto begin() const {
    return named_properties_.begin();
  }

  auto end() const {
    return named_properties_.end();
  }

  size_t size() const {
    return named_properties_.size();
  }

  bool HasProperty(const std::string& property_name) const {
    return named_properties_.count(property_name);
  }

 private:
  InlinedHashMap<std::string, PropertyDataType> named_properties_;
};

template <>
inline PropertyDataType PropertyBag::GetProperty<PropertyDataType>(const std::string& name) const {
  auto it = named_properties_.find(name);
  ORT_ENFORCE(it != named_properties_.end(), "No property named ", name);

  return it->second;
}

}  // namespace onnxruntime::training::api
