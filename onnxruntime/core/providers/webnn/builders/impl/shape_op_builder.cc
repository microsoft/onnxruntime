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

  emscripten::val desc = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc, ONNX_NAMESPACE::TensorProto_DataType_INT64), "Unsupported data type");
  emscripten::val dims = emscripten::val::array();
  dims.call<void>("push", rank);
  desc.set("dimensions", dims);
  emscripten::val shape_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(input_shape));
  emscripten::val shape_constant = model_builder.GetBuilder().call<emscripten::val>("constant", desc, shape_buffer);

  NodeAttrHelper helper(node);
  auto true_start = helper.Get("start", 0);
  auto true_end = helper.Get("end", rank);

  // Deal with negative(s) and clamp.
  true_start = std::clamp(true_start + (true_start < 0 ? rank : 0), 0, rank);
  true_end = std::clamp(true_end + (true_end < 0 ? rank : 0), true_start, rank);
  auto slice_length = true_end - true_start;

  emscripten::val starts = emscripten::val::array();
  starts.call<void>("push", true_start);
  emscripten::val sizes = emscripten::val::array();
  sizes.call<void>("push", slice_length);

  // Since WebNN doesn't support Shape op, we use constant + slice ops as workaround.
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("slice", shape_constant, starts, sizes);

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
