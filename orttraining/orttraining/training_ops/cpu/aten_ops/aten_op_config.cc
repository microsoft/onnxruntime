// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <regex>
#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "orttraining/training_ops/cpu/aten_ops/aten_op_config.h"

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

// TODO: Maybe it's better to put such config strings to a text file, with format instructions.
// We use regex to parse the strings, to make the parser simple, it requires some special formats
// for these function strings, such as spaces in the strings.
static const std::vector<std::pair<std::string, std::string>> ATEN_FUNCS = {
    {"aten::embedding(Tensor<T> weight, Tensor<int64> indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor<T> result",
     "aten::embedding_backward(Tensor<T> grad_result, Tensor<int64> indices, Tensor<T> weight, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor<T> grad_weight"}};

const std::regex regex_expr_whole("([a-z0-9:_]+)\\(([A-Za-z0-9_ ,.=+-\\[\\]<>]+)\\) -> \\(?([A-Za-z0-9_ ,<>]+)\\)?");
const std::regex regex_expr_argument(
    "(Tensor|int|bool|float)(<([A-Za-z0-9_]+)>)?(\\[\\])?(\\?)? ([a-z0-9_]+)(=([TFa-z0-9,.+-\\[\\]]+))?");
const std::regex regex_expr_comma_space(", ");
const std::regex regex_expr_comma(",");
// default constructor = end-of-sequence:
const std::regex_token_iterator<std::string::iterator> rend;

struct Argument {
  ArgumentKind type;
  std::string tensor_elem_type;
  bool is_optional;
  std::string name;
  std::string default_value;

  Argument(ArgumentKind _type, const std::string& _tensor_elem_type, bool _is_optional, const std::string& _name,
           const std::string& _default_value)
      : type(_type),
        tensor_elem_type(_tensor_elem_type),
        is_optional(_is_optional),
        name(_name),
        default_value(_default_value) {}
};

struct Function {
  std::string name;
  std::vector<Argument> arguments;
  std::vector<Argument> returns;

  std::vector<std::tuple<ArgumentKind, std::string, bool>> ToArgumentConfigs() {
    std::vector<std::tuple<ArgumentKind, std::string, bool>> argument_configs;
    for (const auto& argument : arguments) {
      argument_configs.emplace_back(std::make_tuple(argument.type, argument.name, argument.is_optional));
    }

    return argument_configs;
  }
};

Argument ParseArgument(const std::string& argument_str) {
  std::smatch sm_argument;
  ORT_ENFORCE(std::regex_match(argument_str, sm_argument, regex_expr_argument), argument_str,
              " is not a vaild argument.");
  const auto& type_str = sm_argument.str(1);
  const auto& tensor_elem_type = sm_argument.str(3);
  bool is_array = sm_argument.str(4) == "[]";
  ArgumentKind type;
  if (type_str == "Tensor") {
    ORT_ENFORCE(!is_array && tensor_elem_type != "", "Tensor type cannot be an array, and must have element type.");
    type = TENSOR;
  } else {
    ORT_ENFORCE(tensor_elem_type == "", "Non-tensor type should not have element type.");
    if (type_str == "int") {
      type = is_array ? INT_ARRAY : INT;
    } else if (type_str == "float") {
      type = is_array ? FLOAT_ARRAY : FLOAT;
    } else if (type_str == "bool") {
      type = is_array ? BOOL_ARRAY : BOOL;
    } else {
      ORT_ENFORCE(false, "Type ", type_str, " is not supported.");
    }
  }

  return Argument(type, tensor_elem_type, sm_argument.str(5) == "?", sm_argument.str(6), sm_argument.str(8));
}

Function ParseFunction(const std::string& function_str) {
  std::smatch sm_function;
  ORT_ENFORCE(std::regex_match(function_str, sm_function, regex_expr_whole), function_str, " is not a valid function.");
  Function function;
  function.name = sm_function.str(1);
  std::string arguments_str = sm_function.str(2);
  std::string returns_str = sm_function.str(3);
  std::regex_token_iterator<std::string::iterator> arguments(arguments_str.begin(), arguments_str.end(),
                                                             regex_expr_comma_space, -1);
  while (arguments != rend) {
    std::string argument_str = *arguments++;
    function.arguments.emplace_back(ParseArgument(argument_str));
  }

  std::regex_token_iterator<std::string::iterator> returns(returns_str.begin(), returns_str.end(),
                                                           regex_expr_comma_space, -1);
  while (returns != rend) {
    std::string return_str = *returns++;
    function.returns.emplace_back(ParseArgument(return_str));
  }

  return function;
}

int ToOnnxDataType(const std::string& type_str) {
  if (type_str == "float") return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  if (type_str == "bool") return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
  if (type_str == "double") return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
  if (type_str == "int8") return ONNX_NAMESPACE::TensorProto_DataType_INT8;
  if (type_str == "uint8") return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  if (type_str == "int16") return ONNX_NAMESPACE::TensorProto_DataType_INT16;
  if (type_str == "uint16") return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
  if (type_str == "int32" || type_str == "int") return ONNX_NAMESPACE::TensorProto_DataType_INT32;
  if (type_str == "uint32") return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
  if (type_str == "int64") return ONNX_NAMESPACE::TensorProto_DataType_INT64;
  if (type_str == "uint64") return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
  if (type_str == "float16" || type_str == "half") return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

int ParseInt(const std::string& value) {
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    ORT_ENFORCE(false, value, " is not a valid integer string.");
  }
}

float ParseFloat(const std::string& value) {
  try {
    return std::stof(value);
  } catch (const std::exception&) {
    ORT_ENFORCE(false, value, " is not a valid float string.");
  }
}

bool ParseBool(const std::string& value) {
  if (value == "True" || value == "true") {
    return true;
  }

  ORT_ENFORCE(value == "False" || value == "false", value, " is not a valid bool string.");
  return false;
}

std::vector<std::string> SplitValues(const std::string& value) {
  ORT_ENFORCE(value.at(0) == '[' && value.at(value.length() - 1) == ']',
              "Array values must be inside square brackets.");
  std::vector<std::string> values;
  std::string array_value = value.substr(1, value.length() - 2);
  if (!array_value.empty()) {
    std::regex_token_iterator<std::string::iterator> it(array_value.begin(), array_value.end(), regex_expr_comma, -1);
    while (it != rend) {
      values.emplace_back(*it++);
    }
  }

  return values;
}

void AddDefaultValue(const std::string& name, ArgumentKind type, const std::string& value_str,
                     ATenOperatorConfig& config) {
  switch (type) {
    case INT:
      config.default_int_values[name] = ParseInt(value_str);
      break;
    case FLOAT:
      config.default_float_values[name] = ParseFloat(value_str);
      break;
    case BOOL:
      config.default_bool_values[name] = ParseBool(value_str);
      break;
    case INT_ARRAY: {
      std::vector<int> int_values;
      for (const auto& value : SplitValues(value_str)) {
        int_values.emplace_back(ParseInt(value));
      }
      config.default_int_array_values[name] = int_values;
      break;
    }
    case FLOAT_ARRAY: {
      std::vector<float> float_values;
      for (const auto& value : SplitValues(value_str)) {
        float_values.emplace_back(ParseFloat(value));
      }
      config.default_float_array_values[name] = float_values;
    } break;
    case BOOL_ARRAY: {
      std::vector<bool> bool_values;
      for (const auto& value : SplitValues(value_str)) {
        bool_values.emplace_back(ParseBool(value));
      }
      config.default_bool_array_values[name] = bool_values;
    } break;
    default:
      ORT_ENFORCE(false, "Invalid argument type.");
  }
}

ATenOperatorConfig Parse(const std::string& forward_function_str, const std::string& backward_function_str) {
  Function forward_function = ParseFunction(forward_function_str);
  Function backward_function = ParseFunction(backward_function_str);
  ATenOperatorConfig config;
  config.op_name = forward_function.name;
  config.backward_op_name = backward_function.name;
  config.forward_argument_configs = forward_function.ToArgumentConfigs();
  config.backward_argument_configs = backward_function.ToArgumentConfigs();
  std::unordered_map<std::string, size_t> forward_arguments_name_to_index;
  std::unordered_map<std::string, int> forward_arguments_template_types;
  std::unordered_set<std::string> argument_names;
  for (size_t i = 0; i < forward_function.arguments.size(); i++) {
    const auto& argument = forward_function.arguments[i];
    forward_arguments_name_to_index[argument.name] = i;
    if (argument.type == TENSOR &&
        ToOnnxDataType(argument.tensor_elem_type) == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED &&
        forward_arguments_template_types.find(argument.tensor_elem_type) == forward_arguments_template_types.end()) {
      forward_arguments_template_types[argument.tensor_elem_type] = static_cast<int>(i);
    }

    if (argument.default_value != "") {
      argument_names.insert(argument.name);
      AddDefaultValue(argument.name, argument.type, argument.default_value, config);
    }
  }

  std::unordered_map<std::string, size_t> forward_returns_name_to_index;
  for (size_t i = 0; i < forward_function.returns.size(); i++) {
    const auto& argument = forward_function.returns[i];
    forward_returns_name_to_index[argument.name] = i;
    ORT_ENFORCE(argument.type == TENSOR, "Function can only return tensors.");
    const auto& tensor_elem_type = argument.tensor_elem_type;
    int onnx_type = ToOnnxDataType(tensor_elem_type);
    if (onnx_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      config.forward_output_type_infer_configs.emplace_back(std::make_pair(CONCRETE_TYPE, onnx_type));
    } else {
      ORT_ENFORCE(forward_arguments_template_types.find(tensor_elem_type) != forward_arguments_template_types.end(),
                  "Unknown template type in returns.");
      config.forward_output_type_infer_configs.emplace_back(
          std::make_pair(PROPAGATE_FROM_INPUT, forward_arguments_template_types.at(tensor_elem_type)));
    }
  }

  for (const auto& argument : backward_function.arguments) {
    if (argument.type != TENSOR) {
      if (argument.default_value != "" && argument_names.find(argument.name) == argument_names.end()) {
        AddDefaultValue(argument.name, argument.type, argument.default_value, config);
      }
    } else {
      const auto& name = argument.name;
      if (forward_arguments_name_to_index.find(name) != forward_arguments_name_to_index.end()) {
        config.backward_input_source_configs.emplace_back(
            std::make_pair(FORWARD_INPUT, forward_arguments_name_to_index.at(name)));
      } else if (forward_returns_name_to_index.find(name) != forward_returns_name_to_index.end()) {
        config.backward_input_source_configs.emplace_back(
            std::make_pair(FORWARD_OUTPUT, forward_returns_name_to_index.at(name)));
      } else if (forward_returns_name_to_index.find(name.substr(5UL)) != forward_returns_name_to_index.end()) {
        // Output gradient has "grad_" prefix.
        config.backward_input_source_configs.emplace_back(
            std::make_pair(GRAD_OUTPUT, forward_returns_name_to_index.at(name.substr(5UL))));
      } else {
        ORT_ENFORCE(false, "Argument ", name, " is not forward input, output or output gradient.");
      }
    }
  }

  for (const auto& argument : backward_function.returns) {
    std::string name = argument.name.substr(5UL);  // Input gradients have "grad_" prefix.
    ORT_ENFORCE(forward_arguments_name_to_index.find(name) != forward_arguments_name_to_index.end(),
                "Returnd input gradient is not for any of the forward inputs.");
    config.gradient_input_indices.emplace_back(forward_arguments_name_to_index.at(name));
  }

  return config;
}

ATenOperatorConfigs::ATenOperatorConfigs() {
  for (const auto& func : ATEN_FUNCS) {
    ATenOperatorConfig config = Parse(func.first, func.second);
    configs_[config.op_name] = config;
  }
}

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime
