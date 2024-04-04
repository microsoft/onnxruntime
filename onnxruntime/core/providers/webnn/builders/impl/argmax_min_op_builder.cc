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

class ArgMaxMinOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

Status ArgMaxMinOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_rank = input_shape.size();

  NodeAttrHelper helper(node);
  int64_t axis = helper.Get("axis", 0);
  const auto keep_dims = helper.Get("keepdims", 1);
  const auto select_last_index = helper.Get("select_last_index", 0);

  axis = HandleNegativeAxis(axis, input_rank);
  emscripten::val axes = emscripten::val::array();
  axes.call<void>("push", static_cast<uint32_t>(axis));

  emscripten::val options = emscripten::val::object();
  options.set("axes", axes);
  options.set("keepDimensions", keep_dims == 1);
  options.set("selectLastIndex", select_last_index == 1);
  emscripten::val output = emscripten::val::object();

  const auto& op_type = node.OpType();
  if (op_type == "ArgMax") {
    output = model_builder.GetBuilder().call<emscripten::val>("argMax", input, options);
  } else if (op_type == "ArgMin") {
    output = model_builder.GetBuilder().call<emscripten::val>("argMin", input, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ArgMaxMinOpBuilder, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool ArgMaxMinOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                           const Node& node,
                                           WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  return true;
}

void CreateArgMaxMinOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "ArgMax",
          "ArgMin",
      };

  op_registrations.builders.push_back(std::make_unique<ArgMaxMinOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
