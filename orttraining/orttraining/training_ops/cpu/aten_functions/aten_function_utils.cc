// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/DLConvertor.h>
#include "core/dlpack/dlpack_converter.h"
#include "orttraining/training_ops/cpu/aten_functions/aten_function_utils.h"

namespace onnxruntime {
namespace contrib {
namespace aten_functions {

c10::IValue AttributesJsonParser::GetValue(const c10::Argument& argument) {
  const auto& name = argument.name();
  switch (argument.type()->kind()) {
    case c10::TypeKind::IntType:
      int int_value;
      if (TryGetValue<int>(name, int_value)) {
        return c10::IValue(static_cast<int64_t>(int_value));
      }
      break;
    case c10::TypeKind::FloatType:
      float float_value;
      if (TryGetValue<float>(name, float_value)) {
        return c10::IValue(float_value);
      }
      break;
    case c10::TypeKind::BoolType:
      bool bool_value;
      if (TryGetValue<bool>(name, bool_value)) {
        return c10::IValue(bool_value);
      }
      break;
    case c10::TypeKind::ListType:
      break;
    default:
      ORT_ENFORCE(false);
  }

  ORT_ENFORCE(argument.default_value().has_value());
  return argument.default_value().value();
}

template <typename T>
bool AttributesJsonParser::TryGetValue(const std::string& name, T& value) {
  if (!parsed_json_.contains(name)) {
    return false;
  }

  const auto& attr = parsed_json_.at(name);
  bool is_type_matched = false;
  if (std::is_same<T, int>::value) is_type_matched = attr.is_number_integer();
  if (std::is_same<T, float>::value) is_type_matched = attr.is_number_float();
  if (std::is_same<T, bool>::value) is_type_matched = attr.is_boolean();
  if (!is_type_matched) {
    return false;
  }

  value = attr.get<T>();
  return true;
}

template <typename T>
bool AttributesJsonParser::TryGetListValue(const std::string& name, const std::vector<T>& value) {
  if (!parsed_json_.contains(name)) {
    return false;
  }

  const auto& attr = parsed_json_.at(name);
  if (!attr.is_array()) {
    return false;
  }

  for (const auto& elem : attr) {
    bool is_type_matched = false;
    if (std::is_same<T, int>::value) is_type_matched = attr.is_number_integer();
    if (std::is_same<T, float>::value) is_type_matched = attr.is_number_float();
    if (std::is_same<T, bool>::value) is_type_matched = attr.is_boolean();
    if (!is_type_matched) {
      return false;
    }

    value.emplace_back(elem.get<T>());
  }

  return true;
}

at::Tensor ToTorchTensor(OrtValue& ort_value) { return at::fromDLPack(dlpack::OrtValueToDlpack(ort_value)); }

OrtValue FromTorchTensor(const at::Tensor& torch_tensor) {
  return dlpack::DlpackToOrtValue(at::toDLPack(torch_tensor), torch_tensor.dtype() == at::kBool);
}

}  // namespace aten_functions
}  // namespace contrib
}  // namespace onnxruntime
