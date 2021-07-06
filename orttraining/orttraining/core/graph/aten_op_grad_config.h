// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace training {

// To indicate the source of backward Op inputs.
enum BackwardInputSourceKind {
  GRAD_OUTPUT,     // Current input is i-th output grad, i.e., GO(i) in gradient builder.
  FORWARD_INPUT,   // Current input is i-th forward input, i.e., I(i) in gradient builder.
  FORWARD_OUTPUT,  // Current input is i-th forward output, i.e., O(i) in gradient builder.
};

struct BackwardInputSourceConfig {
  BackwardInputSourceKind kind;
  size_t index;
  std::string transform_func;

  BackwardInputSourceConfig(BackwardInputSourceKind _kind, size_t _index, std::string _transform_func)
      : kind(_kind), index(_index), transform_func(_transform_func) {}
};

// TODO: need to support default attribute value.
struct ATenOpGradConfig {
  std::string backward_op_name;
  // The source config of inputs of gradient of com.microsoft::ATenOp.
  std::vector<BackwardInputSourceConfig> backward_input_source_configs;
  // The mapping between gradient of com.microsoft::ATenOp's outputs and com.microsoft::ATenOp's inputs,
  // i.e., gradient_input_indices[i] means GI(gradient_input_indices[i]) in gradient builder.
  std::vector<size_t> gradient_input_indices;
};

ATenOpGradConfig ParseATenOpGradConfig(const std::string& config_str);

class ATenOpGradConfigs {
 public:
  static ATenOpGradConfigs& Instance() {
    static ATenOpGradConfigs instance;
    return instance;
  }

  const ATenOpGradConfig* GetConfig(const std::string& op_name) {
    auto it = configs_.find(op_name);
    return it != configs_.end() ? &it->second : nullptr;
  }

 private:
  ATenOpGradConfigs();
  ~ATenOpGradConfigs() = default;

  std::unordered_map<std::string, ATenOpGradConfig> configs_;
};

}  // namespace training
}  // namespace onnxruntime
