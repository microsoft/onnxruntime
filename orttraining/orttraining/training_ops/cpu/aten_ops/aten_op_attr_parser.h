// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <nlohmann/json.hpp>

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

class AttributesJsonParser {
 public:
  AttributesJsonParser(const std::string& json_str) { parsed_json_ = nlohmann::json::parse(json_str); }

  template <typename T>
  bool TryGetValue(const std::string& name, T& value) {
    if (!parsed_json_.contains(name)) {
      return false;
    }

    const auto& attr = parsed_json_.at(name);
    bool is_type_matched = false;
    if (std::is_same<T, int>::value)
      is_type_matched = attr.is_number_integer();
    else if (std::is_same<T, float>::value)
      is_type_matched = attr.is_number_float();
    else if (std::is_same<T, bool>::value)
      is_type_matched = attr.is_boolean();
    if (!is_type_matched) {
      return false;
    }

    value = attr.get<T>();
    return true;
  }

  template <typename T>
  bool TryGetListValue(const std::string& name, std::vector<T>& value) {
    if (!parsed_json_.contains(name)) {
      return false;
    }

    const auto& attr = parsed_json_.at(name);
    if (!attr.is_array()) {
      return false;
    }

    for (const auto& elem : attr) {
      bool is_type_matched = false;
      if (std::is_same<T, int>::value)
        is_type_matched = attr.is_number_integer();
      else if (std::is_same<T, float>::value)
        is_type_matched = attr.is_number_float();
      else if (std::is_same<T, bool>::value)
        is_type_matched = attr.is_boolean();
      if (!is_type_matched) {
        return false;
      }

      value.emplace_back(elem.get<T>());
    }

    return true;
  }

 private:
  nlohmann::json parsed_json_;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
