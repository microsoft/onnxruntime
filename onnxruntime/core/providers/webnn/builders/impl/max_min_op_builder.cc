// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class MaxMinOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.

Status MaxMinOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  emscripten::val input0 = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

  const size_t input_count = input_defs.size();
  ORT_RETURN_IF(input_count < 1, op_type, "has no inputs");
  ORT_RETURN_IF_NOT(op_type == "Max" || op_type == "Min", "MaxMinOpBuilder, unknown op: ", op_type);

  emscripten::val output = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  if (input_count == 1) {
    // For 1 input, use identity instead.
    output = model_builder.GetBuilder().call<emscripten::val>("identity", input0, options);
  } else {
    std::string webnn_op_name = op_type == "Max" ? "max" : "min";

    emscripten::val input1 = model_builder.GetOperand(input_defs[1]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>(webnn_op_name.c_str(), input0, input1, options);

    for (size_t input_index = 2; input_index < input_count; ++input_index) {
      emscripten::val next_input = model_builder.GetOperand(input_defs[input_index]->Name());
      emscripten::val next_options = emscripten::val::object();
      next_options.set("label", node.Name() + "_" + input_defs[input_index]->Name());
      output = model_builder.GetBuilder().call<emscripten::val>(webnn_op_name.c_str(), output, next_input, next_options);
    }
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool MaxMinOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                        const Node& node,
                                        WebnnDeviceType /* device_type */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_defs.size() < 1) {
    LOGS(logger, VERBOSE) << op_type << " requires at least one input (data)";
    return false;
  }

  return true;
}

bool MaxMinOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;
  int32_t input1_type;

  if (!GetType(*input_defs[0], input0_type, logger))
    return false;

  if (input_defs.size() > 1 && !GetType(*input_defs[1], input1_type, logger)) {
    return false;
  }

  std::string webnn_op_type;
  if (!GetWebNNOpType(op_type, webnn_op_type))
    return false;

  if (!IsSupportedDataType(input0_type, wnn_limits[webnn_op_type]["a"]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input0_type
                          << "] is not supported for now";
    return false;
  }

  if (input_defs.size() > 1 && input0_type != input1_type) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input data types should be the same.";
    return false;
  }

  return true;
}

void CreateMaxMinOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Max",
          "Min",
      };

  op_registrations.builders.push_back(std::make_unique<MaxMinOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
