// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <torch/torch.h>

namespace onnxruntime {
namespace contrib {
namespace aten_functions {

using TensorTransformFunc = std::function<c10::IValue(const at::Tensor&)>;

enum ForwardTensorOutputTypeKind {
  PROPAGATE_FROM_INPUT,
  CONCRETE_TYPE,
};

enum BackwardTensorInputKind {
  GRAD_OUTPUT,
  FORWARD_INPUT,
  FORWARD_OUTPUT,
};

struct ATenFunctionConfig {
  std::string backward_function_name;
  std::vector<std::tuple<BackwardTensorInputKind, int>> backward_tensor_input_configs;
  std::vector<std::tuple<ForwardTensorOutputTypeKind, int>> forward_tensor_output_type_configs;
  std::vector<int> backward_output_configs;
  std::unordered_map<int, TensorTransformFunc> custom_transformers;

  ATenFunctionConfig(
      const std::string& i_backward_function_name,
      const std::vector<std::tuple<BackwardTensorInputKind, int>>& i_backward_tensor_input_configs,
      const std::vector<std::tuple<ForwardTensorOutputTypeKind, int>>& i_forward_tensor_output_type_configs,
      const std::vector<int>& i_backward_output_configs,
      const std::unordered_map<int, TensorTransformFunc>& i_custom_transformers = {}) {
    backward_function_name = i_backward_function_name;
    backward_tensor_input_configs.assign(i_backward_tensor_input_configs.begin(),
                                         i_backward_tensor_input_configs.end());
    forward_tensor_output_type_configs.assign(i_forward_tensor_output_type_configs.begin(),
                                              i_forward_tensor_output_type_configs.end());
    backward_output_configs.assign(i_backward_output_configs.begin(), i_backward_output_configs.end());
    custom_transformers.insert(i_custom_transformers.begin(), i_custom_transformers.end());
  }
};

static const TensorTransformFunc embedding_num_weights = [](const at::Tensor& tensor) {
  return c10::IValue(tensor.size(0));
};

static const std::unordered_map<std::string, ATenFunctionConfig> ATEN_FUNCTIONS = {
    {"aten::embedding",
     ATenFunctionConfig("aten::embedding_backward", {{GRAD_OUTPUT, 0}, {FORWARD_INPUT, 1}, {FORWARD_INPUT, 0}},
                        {{PROPAGATE_FROM_INPUT, 0}}, {0}, {{2, embedding_num_weights}})},
};

}  // namespace aten_functions
}  // namespace contrib
}  // namespace onnxruntime
