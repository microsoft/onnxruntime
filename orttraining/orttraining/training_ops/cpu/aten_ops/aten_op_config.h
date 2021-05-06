// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

// TODO: We need to ATen operator config to pass arguments to PyTorch as well as building gradient graph.
// Currently these configs are C++ codes below, ideally we can use string/text configs
// just as derivatives.yaml in PyTorch, and parse that text to generate below configs.

enum ForwardTensorOutputTypeKind {
  PROPAGATE_FROM_INPUT,
  CONCRETE_TYPE,
};

enum BackwardTensorInputKind {
  GRAD_OUTPUT,
  FORWARD_INPUT,
  FORWARD_OUTPUT,
};

enum ArgumentKind {
  TENSOR,
  INT,
  FLOAT,
  BOOL,
  // TODO: need more type, such as list type
};

// TODO: need to support default attribute value.
struct ATenOperatorConfig {
  std::string backward_op_name;
  std::vector<std::tuple<ArgumentKind, std::string>> forward_argument_configs;
  std::vector<std::tuple<ArgumentKind, std::string>> backward_argument_configs;
  std::vector<std::tuple<BackwardTensorInputKind, int>> backward_tensor_input_configs;
  std::vector<std::tuple<ForwardTensorOutputTypeKind, int>> forward_tensor_output_type_configs;
  std::vector<int> backward_output_configs;

  ATenOperatorConfig(
      const std::string& i_backward_op_name,
      const std::vector<std::tuple<ArgumentKind, std::string>>& i_forward_argument_configs,
      const std::vector<std::tuple<ArgumentKind, std::string>>& i_backward_argument_configs,
      const std::vector<std::tuple<BackwardTensorInputKind, int>>& i_backward_tensor_input_configs,
      const std::vector<std::tuple<ForwardTensorOutputTypeKind, int>>& i_forward_tensor_output_type_configs,
      const std::vector<int>& i_backward_output_configs) {
    backward_op_name = i_backward_op_name;
    forward_argument_configs.assign(i_forward_argument_configs.begin(), i_forward_argument_configs.end());
    backward_argument_configs.assign(i_backward_argument_configs.begin(), i_backward_argument_configs.end());
    backward_tensor_input_configs.assign(i_backward_tensor_input_configs.begin(),
                                         i_backward_tensor_input_configs.end());
    forward_tensor_output_type_configs.assign(i_forward_tensor_output_type_configs.begin(),
                                              i_forward_tensor_output_type_configs.end());
    backward_output_configs.assign(i_backward_output_configs.begin(), i_backward_output_configs.end());
  }
};

static const std::unordered_map<std::string, ATenOperatorConfig> ATEN_OPERATORS = {
    {"aten::embedding",
     ATenOperatorConfig("aten::embedding_backward",
                        {{TENSOR, "weight"},
                         {TENSOR, "indices"},
                         {INT, "padding_idx"},
                         {BOOL, "scale_grad_by_freq"},
                         {BOOL, "sparse"}},
                        {{TENSOR, "grad"},
                         {TENSOR, "indices"},
                         {TENSOR, "weight"},
                         {INT, "padding_idx"},
                         {BOOL, "scale_grad_by_freq"},
                         {BOOL, "sparse"}},
                        {{GRAD_OUTPUT, 0}, {FORWARD_INPUT, 1}, {FORWARD_INPUT, 0}}, {{PROPAGATE_FROM_INPUT, 0}}, {0})},
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
