// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <nlohmann/json.hpp>

namespace onnxruntime {
namespace external_functions {

using json = nlohmann::json;

class AttributesJsonParser {
 public:
  AttributesJsonParser(const std::string& json_str) { parsed_json_ = json::parse(json_str); }

  template <typename T>
  T GetAttributeOrDefault(const std::string& name, const T& default_value) {
    if (!parsed_json_.contains(name)) {
      return default_value;
    }

    const auto& attr = parsed_json_.at(name);
    bool is_type_matched = false;
    if (std::is_same<T, int>::value) is_type_matched = attr.is_number_integer();
    if (std::is_same<T, float>::value) is_type_matched = attr.is_number_float();
    if (std::is_same<T, bool>::value) is_type_matched = attr.is_boolean();
    if (!is_type_matched) {
      return default_value;
    }

    return attr.get<T>();
  }

  template <typename T>
  std::vector<T> GetListAttributeOrDefault(const std::string& name,
                                           const std::vector<T>& default_value = std::vector<T>{}) {
    if (!parsed_json_.contains(name)) {
      return default_value;
    }

    const auto& attr = parsed_json_.at(name);
    if (!attr.is_array()) {
      return default_value;
    }

    std::vector<T> list_attribute;
    for (const auto& elem : attr) {
      bool is_type_matched = false;
      if (std::is_same<T, int>::value) is_type_matched = attr.is_number_integer();
      if (std::is_same<T, float>::value) is_type_matched = attr.is_number_float();
      if (!is_type_matched) {
        return default_value;
      }

      list_attribute.emplace_back(elem.get<T>());
    }

    return list_attribute;
  }

 private:
  nlohmann::json parsed_json_;
};

}  // namespace external_functions
}  // namespace onnxruntime
