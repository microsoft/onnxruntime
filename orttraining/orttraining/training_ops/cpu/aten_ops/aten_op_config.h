// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

// TODO: We need to ATen operator config to pass arguments to PyTorch as well as building gradient graph.
// Currently these configs are C++ codes below, ideally we can use string/text configs
// just as derivatives.yaml in PyTorch, and parse that text to generate below configs.

// To indicate how to infer outputs' types.
enum OutputTypeInferKind {
  PROPAGATE_FROM_INPUT,  // Propagate current output's type from i-th input.
  CONCRETE_TYPE,         // Current output's type is concrete type with value of i (i.e., float if i = 1).
};

// To indicate the source of backward Op inputs.
enum BackwardInputSourceKind {
  GRAD_OUTPUT,     // Current input is i-th output grad, i.e., GO(i) in gradient builder.
  FORWARD_INPUT,   // Current input is i-th forward input, i.e., I(i) in gradient builder.
  FORWARD_OUTPUT,  // Current input is i-th forward output, i.e., O(i) in gradient builder.
};

// To indicete the argument kind of ATen Op.
enum ArgumentKind {
  TENSOR,
  INT,
  FLOAT,
  BOOL,
  INT_ARRAY,
  FLOAT_ARRAY,
  BOOL_ARRAY,
  // TODO: may need more type
};

// TODO: need to support default attribute value.
struct ATenOperatorConfig {
  std::string op_name;
  std::string backward_op_name;
  // Forward ATen Op's argument kind, name and if it's optional.
  std::vector<std::tuple<ArgumentKind, std::string, bool>> forward_argument_configs;
  // Backward ATen Op's argument kind, name and if it's optional.
  std::vector<std::tuple<ArgumentKind, std::string, bool>> backward_argument_configs;
  // The source config of inputs of com.microsoft::ATenOpGrad.
  std::vector<std::pair<BackwardInputSourceKind, size_t>> backward_input_source_configs;
  // The output type infer config of outputs of com.microsoft::ATenOp.
  std::vector<std::pair<OutputTypeInferKind, int>> forward_output_type_infer_configs;
  // The mapping between com.microsoft::ATenOpGrad's outputs and com.microsoft::ATenOp's inputs,
  // i.e., gradient_input_indices[i] means GI(gradient_input_indices[i]) in gradient builder.
  std::vector<size_t> gradient_input_indices;
  // Default argument values.
  std::unordered_map<std::string, int> default_int_values;
  std::unordered_map<std::string, float> default_float_values;
  std::unordered_map<std::string, bool> default_bool_values;
  std::unordered_map<std::string, std::vector<int>> default_int_array_values;
  std::unordered_map<std::string, std::vector<float>> default_float_array_values;
  std::unordered_map<std::string, std::vector<bool>> default_bool_array_values;

  template <typename T>
  bool TryGetDefaultValue(const std::string& name, T& value) const {
    bool has_default_value = false;
    if (std::is_same<T, int>::value) {
      auto it = default_int_values.find(name);
      if (it != default_int_values.end()) {
        has_default_value = true;
        value = static_cast<T>(it->second);
      }
    } else if (std::is_same<T, float>::value) {
      auto it = default_float_values.find(name);
      if (it != default_float_values.end()) {
        has_default_value = true;
        value = static_cast<T>(it->second);
      }
    } else if (std::is_same<T, bool>::value) {
      auto it = default_bool_values.find(name);
      if (it != default_bool_values.end()) {
        has_default_value = true;
        value = static_cast<T>(it->second);
      }
    }

    return has_default_value;
  }

  template <typename T>
  bool TryGetDefaultArrayValue(const std::string& name, std::vector<T>& value) const {
    bool has_default_value = false;
    if (std::is_same<T, int>::value) {
      auto it = default_int_array_values.find(name);
      if (it != default_int_array_values.end()) {
        has_default_value = true;
        for (auto elem : it->second) {
          value.emplace_back(static_cast<T>(elem));
        }
      }
    } else if (std::is_same<T, float>::value) {
      auto it = default_float_array_values.find(name);
      if (it != default_float_array_values.end()) {
        has_default_value = true;
        for (auto elem : it->second) {
          value.emplace_back(static_cast<T>(elem));
        }
      }
    } else if (std::is_same<T, bool>::value) {
      auto it = default_bool_array_values.find(name);
      if (it != default_bool_array_values.end()) {
        has_default_value = true;
        for (auto elem : it->second) {
          value.emplace_back(static_cast<T>(elem));
        }
      }
    }

    return has_default_value;
  }
};

ATenOperatorConfig Parse(const std::string& forward_function_str, const std::string& backward_function_str);

class ATenOperatorConfigs {
 public:
  static ATenOperatorConfigs& Instance() {
    static ATenOperatorConfigs instance;
    return instance;
  }

  const ATenOperatorConfig* GetConfig(const std::string& op_name) {
    auto it = configs_.find(op_name);
    return it != configs_.end() ? &it->second : nullptr;
  }

 private:
  ATenOperatorConfigs();
  ~ATenOperatorConfigs() = default;

  std::unordered_map<std::string, ATenOperatorConfig> configs_;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
