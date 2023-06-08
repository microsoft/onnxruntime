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

class ShapeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType device_type, const logging::Logger& logger) const override;
};

Status ShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto rank = static_cast<int32_t>(input_shape.size());

  NodeAttrHelper helper(node);
  auto true_start = helper.Get("start", 0);
  auto true_end = helper.Get("end", rank);

  // Deal with negative(s) and clamp.
  true_start = true_start < 0 ? true_start + rank : true_start;
  true_start = true_start < 0 ? 0 : ((true_start > rank) ? rank : true_start);

  true_end = true_end < 0 ? true_end + rank : true_end;
  true_end = true_end < 0 ? 0 : ((true_end > rank) ? rank : true_end);

  emscripten::val new_shape = emscripten::val::array(input_shape);

  // Slice the input shape if start or end attribute exists.
  new_shape = new_shape.call<emscripten::val>("slice", true_start, true_end);

  emscripten::val desc = emscripten::val::object();
  desc.set("type", emscripten::val("int64"));
  emscripten::val dims = emscripten::val::array();
  auto slice_length = true_end < true_start ? 0 : (true_end - true_start);
  dims.call<void>("push", slice_length);
  desc.set("dimensions", dims);
  emscripten::val output_buffer = emscripten::val::global("BigInt64Array").new_(new_shape);
  // Since WebNN doesn't support Shape op, we calculate the Shape output and pass values to
  // WebNN's constant op as workaround.
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("constant", desc, output_buffer);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ShapeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType device_type,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  int32_t output_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
  if (!IsSupportedDataType(output_type, device_type)) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Output type: [" << output_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
