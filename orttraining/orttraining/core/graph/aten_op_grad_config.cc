// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/aten_op_grad_config.h"

#include <regex>
#include "core/common/common.h"

namespace onnxruntime {
namespace training {

static const std::vector<std::pair<std::string, std::string>> ATEN_OP_GRAD_CONFIG_STRS = {
    {"aten::embedding", "aten::embedding_backward(GO(0), I(1), I(0).size(0), I(2), I(3), I(4)) -> GI(0)"},
    {"aten::max_pool2d_with_indices",
     "aten::max_pool2d_with_indices_backward(GO(0), I(0), I(1), I(2), I(3), I(4), I(5), O(1)) -> GI(0)"},
    {"aten::unfold", "aten::unfold_backward(GO(0), I(0).sizes(), I(1), I(2), I(3)) -> GI(0)"}};

const std::regex regex_expr_whole("([a-z0-9:_]+)\\(([GIOa-z0-9_ ,.\\(\\)]+)\\) -> ([GI0-9 ,\\(\\)]+)");
const std::regex regex_expr_argument("(I|O|GI|GO)\\(([0-9]+)\\)(\\.([a-z0-9._\\(\\)]+))?");
const std::regex regex_expr_comma_space(", ");
// default constructor = end-of-sequence:
const std::regex_token_iterator<std::string::iterator> rend;

ATenOpGradConfig ParseATenOpGradConfig(const std::string& config_str) {
  std::smatch sm_function;
  ORT_ENFORCE(std::regex_match(config_str, sm_function, regex_expr_whole), config_str, " is not valid.");
  ATenOpGradConfig config;
  config.backward_op_name = sm_function.str(1);
  std::string arguments_str = sm_function.str(2);
  std::regex_token_iterator<std::string::iterator> arguments(arguments_str.begin(), arguments_str.end(),
                                                             regex_expr_comma_space, -1);
  while (arguments != rend) {
    std::string argument_str = *arguments++;
    std::smatch sm_argument;
    ORT_ENFORCE(std::regex_match(argument_str, sm_argument, regex_expr_argument), argument_str, " is not valid.");
    const auto& type_str = sm_argument.str(1);
    ORT_ENFORCE(type_str == "GO" || type_str == "I" || type_str == "O",
                "Input of gradient Op cannot be input's gradient.");
    BackwardInputSourceKind kind = type_str == "GO" ? GRAD_OUTPUT : type_str == "I" ? FORWARD_INPUT : FORWARD_OUTPUT;
    config.backward_input_source_configs.emplace_back(
        BackwardInputSourceConfig(kind, static_cast<size_t>(std::stoi(sm_argument.str(2))), sm_argument.str(4)));
  }

  std::string returns_str = sm_function.str(3);
  std::regex_token_iterator<std::string::iterator> returns(returns_str.begin(), returns_str.end(),
                                                           regex_expr_comma_space, -1);
  while (returns != rend) {
    std::string return_str = *returns++;
    std::smatch sm_argument;
    ORT_ENFORCE(std::regex_match(return_str, sm_argument, regex_expr_argument), return_str, " is not valid.");
    ORT_ENFORCE(sm_argument.str(1) == "GI", "Output of gradient Op should be one of input's gradient.");
    config.gradient_input_indices.emplace_back(static_cast<size_t>(std::stoi(sm_argument.str(2))));
  }

  return config;
}

ATenOpGradConfigs::ATenOpGradConfigs() {
  for (const auto& config_str : ATEN_OP_GRAD_CONFIG_STRS) {
    ATenOpGradConfig config = ParseATenOpGradConfig(config_str.second);
    configs_[config_str.first] = config;
  }
}

}  // namespace training
}  // namespace onnxruntime
