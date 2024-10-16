// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/webnn/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class LogicalOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.

Status LogicalOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                               const logging::Logger& /* logger */) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  emscripten::val input0 = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val input1 = emscripten::val::undefined();
  if (input_defs.size() > 1) {
    input1 = model_builder.GetOperand(input_defs[1]->Name());
  }

  emscripten::val output = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  if (op_type == "Equal") {
    output = model_builder.GetBuilder().call<emscripten::val>("equal", input0, input1, options);
  } else if (op_type == "Greater") {
    output = model_builder.GetBuilder().call<emscripten::val>("greater", input0, input1, options);
  } else if (op_type == "GreaterOrEqual") {
    output = model_builder.GetBuilder().call<emscripten::val>("greaterOrEqual", input0, input1, options);
  } else if (op_type == "Less") {
    output = model_builder.GetBuilder().call<emscripten::val>("lesser", input0, input1, options);
  } else if (op_type == "LessOrEqual") {
    output = model_builder.GetBuilder().call<emscripten::val>("lesserOrEqual", input0, input1, options);
  } else if (op_type == "Not") {
    output = model_builder.GetBuilder().call<emscripten::val>("logicalNot", input0, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "LogicalOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool LogicalOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 2 && op_type != "Not") {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] requires at least 2 inputs, actual: "
                          << input_defs.size();
    return false;
  }
  return true;
}

bool LogicalOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;
  int32_t input1_type;

  if (!GetType(*input_defs[0], input0_type, logger))
    return false;

  if (op_type != "Not") {
    if (!GetType(*input_defs[1], input1_type, logger))
      return false;
    std::array<int32_t, 2> input_types{input0_type, input1_type};
    if (!AreInputDataTypesSame(op_type, input_types, logger)) {
      return false;
    }
  }

  std::string onnx_input_name = op_type == "Not" ? "X" : "A";
  return IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "a", onnx_input_name, logger);
}

void CreateLogicalOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Equal",
          "Greater",
          "GreaterOrEqual",
          "Less",
          "LessOrEqual",
          "Not",
      };

  op_registrations.builders.push_back(std::make_unique<LogicalOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
