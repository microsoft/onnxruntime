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
  // TODO: need more type, such as list type
};

// TODO: need to support default attribute value.
struct ATenOperatorConfig {
  std::string backward_op_name;
  // Forward ATen Op's argument kind and name.
  std::vector<std::tuple<ArgumentKind, std::string>> forward_argument_configs;
  // Backward ATen Op's argument kind and name.
  std::vector<std::tuple<ArgumentKind, std::string>> backward_argument_configs;
  // The source config of inputs of com.microsoft::ATenOpGrad.
  std::vector<std::tuple<BackwardInputSourceKind, int>> backward_input_source_configs;
  // The output type infer config of outputs of com.microsoft::ATenOp.
  std::vector<std::tuple<OutputTypeInferKind, int>> forward_output_type_infer_configs;
  // The mapping between com.microsoft::ATenOpGrad's outputs and com.microsoft::ATenOp's inputs,
  // i.e., gradient_input_indices[i] means GI(gradient_input_indices[i]) in gradient builder.
  std::vector<int> gradient_input_indices;

  ATenOperatorConfig(const std::string& _backward_op_name,
                     const std::vector<std::tuple<ArgumentKind, std::string>>& _forward_argument_configs,
                     const std::vector<std::tuple<ArgumentKind, std::string>>& _backward_argument_configs,
                     const std::vector<std::tuple<BackwardInputSourceKind, int>>& _backward_input_source_configs,
                     const std::vector<std::tuple<OutputTypeInferKind, int>>& _forward_output_type_infer_configs,
                     const std::vector<int>& _gradient_input_indices) {
    backward_op_name = _backward_op_name;
    forward_argument_configs.assign(_forward_argument_configs.begin(), _forward_argument_configs.end());
    backward_argument_configs.assign(_backward_argument_configs.begin(), _backward_argument_configs.end());
    backward_input_source_configs.assign(_backward_input_source_configs.begin(), _backward_input_source_configs.end());
    forward_output_type_infer_configs.assign(_forward_output_type_infer_configs.begin(),
                                             _forward_output_type_infer_configs.end());
    gradient_input_indices.assign(_gradient_input_indices.begin(), _gradient_input_indices.end());
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

inline const ATenOperatorConfig* GetATenOperatorConfig(const std::string& op_name) {
  auto it = ATEN_OPERATORS.find(op_name);
  return it != ATEN_OPERATORS.end() ? &it->second : nullptr;
}

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
