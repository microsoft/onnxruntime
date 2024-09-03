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

class BinaryOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType device_type, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());

  emscripten::val input0 = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val input1 = model_builder.GetOperand(node.InputDefs()[1]->Name());
  emscripten::val output = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  if (op_type == "Add") {
    output = model_builder.GetBuilder().call<emscripten::val>("add", input0, input1, options);
  } else if (op_type == "Sub") {
    output = model_builder.GetBuilder().call<emscripten::val>("sub", input0, input1, options);
  } else if (op_type == "Mul") {
    output = model_builder.GetBuilder().call<emscripten::val>("mul", input0, input1, options);
  } else if (op_type == "Div") {
    output = model_builder.GetBuilder().call<emscripten::val>("div", input0, input1, options);
  } else if (op_type == "Pow") {
    output = model_builder.GetBuilder().call<emscripten::val>("pow", input0, input1, options);
  } else if (op_type == "PRelu") {
    output = model_builder.GetBuilder().call<emscripten::val>("prelu", input0, input1, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "BinaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool BinaryOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                        const Node& node,
                                        const WebnnDeviceType device_type,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  std::vector<int64_t> input0_shape;
  std::vector<int64_t> input1_shape;
  if (!GetShape(*input_defs[0], input0_shape, logger) ||
      !GetShape(*input_defs[1], input1_shape, logger)) {
    return false;
  }

  // 'prelu' op in WebNN CPU backend restricts the last dimension of input and slope to be same.
  // TODO: Remove this workaround once the associated issue is resolved in Chromium:
  // https://issues.chromium.org/issues/335517470.
  if (op_type == "PRelu" && device_type == WebnnDeviceType::CPU) {
    if (input0_shape.back() != input1_shape.back()) {
      LOGS(logger, VERBOSE) << "The last dimension of input and slope for PRelu must be same for WebNN CPU backend.";
      return false;
    }
  }

  return true;
}

bool BinaryOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;
  int32_t input1_type;

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger))
    return false;

  std::string webnn_op_type;
  if (!GetWebNNOpType(op_type, webnn_op_type))
    return false;

  std::string webnn_input_name = op_type == "PRelu" ? "input" : "a";
  if (!IsSupportedDataType(input0_type, wnn_limits[webnn_op_type][webnn_input_name]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input0_type
                          << "] is not supported for now";
    return false;
  }

  if (input0_type != input1_type) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input data types should be the same.";
    return false;
  }

  return true;
}

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "Pow",
          "PRelu",
      };

  op_registrations.builders.push_back(std::make_unique<BinaryOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
