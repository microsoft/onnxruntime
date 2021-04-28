// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <nlohmann/json.hpp>
#include <torch/torch.h>

namespace onnxruntime {
namespace contrib {
namespace aten_functions {

class AttributesJsonParser {
 public:
  AttributesJsonParser(const std::string& json_str) { parsed_json_ = nlohmann::json::parse(json_str); }
  c10::IValue GetValue(const c10::Argument& argument);

 private:
  template <typename T>
  bool TryGetValue(const std::string& name, T& value);

  template <typename T>
  bool TryGetListValue(const std::string& name, const std::vector<T>& value);

  nlohmann::json parsed_json_;
};

at::Tensor ToTorchTensor(OrtValue& ort_value);
OrtValue FromTorchTensor(const at::Tensor& torch_tensor);

}  // namespace aten_functions
}  // namespace contrib
}  // namespace onnxruntime
