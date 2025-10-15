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

class ScatterElementsOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  mutable bool can_fallback_int64_to_int32_ = false;
};

// Add operator related.

Status ScatterElementsOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val data = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val indices = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val updates = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val options = emscripten::val::object();

  // ONNX specifies that indices must use int64, but some WebNN backends only support int32.
  // As a workaround for such backends, we can insert a Cast operation to convert the data type.
  if (can_fallback_int64_to_int32_) {
    options.set("label", node.Name() + "_cast_indices_to_int32");
    indices = model_builder.GetBuilder().call<emscripten::val>("cast", indices, emscripten::val("int32"), options);
  }

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const size_t rank = input_shape.size();
  NodeAttrHelper helper(node);
  const uint32_t axis = static_cast<uint32_t>(HandleNegativeAxis(helper.Get("axis", 0), rank));
  options.set("axis", axis);
  options.set("label", node.Name());

  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("scatterElements", data, indices, updates, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ScatterElementsOpBuilder::IsOpSupportedImpl(const GraphViewer&, const Node& node,
                                                 const WebnnDeviceType /* device_type */,
                                                 const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  if (helper.Get("reduction", "none") != "none") {
    LOGS(logger, VERBOSE) << "ScatterElements: WebNN only supports reduction type none (default)";
    return false;
  }

  return true;
}

bool ScatterElementsOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) const {
  const auto& data = *node.InputDefs()[0];
  const auto& indices = *node.InputDefs()[1];
  const auto& updates = *node.InputDefs()[2];

  int32_t data_type;
  int32_t indices_type;
  int32_t updates_type;
  if (!GetType(data, data_type, logger) || !GetType(indices, indices_type, logger) ||
      !GetType(updates, updates_type, logger)) {
    return false;
  }

  if (data_type != updates_type) {
    return false;
  }

  const std::string_view op_type = node.OpType();

  if (!IsDataTypeSupportedByOp(op_type, data_type, wnn_limits, "input", "data", logger) ||
      !IsInputRankSupportedByOp(node, wnn_limits, logger)) {
    return false;
  }

  // ONNX specifies that indices must use int64, but some WebNN backends only support int32.
  // Allows to use int32 as a workaround for such backends.
  can_fallback_int64_to_int32_ = CanFallbackInt64ToInt32(wnn_limits, "scatterElements", "indices");
  return can_fallback_int64_to_int32_ ||
         IsDataTypeSupportedByOp(op_type, indices_type, wnn_limits, "indices", "indices", logger);
}

void CreateScatterElementsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ScatterElementsOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
