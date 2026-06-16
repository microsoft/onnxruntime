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

class GatherBlockQuantizedOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// WebNN doesn't provide a dedicated op for GatherBlockQuantizedOpBuilder, it can be simply
// decomposed by DequantizeLinear + Gather.
Status GatherBlockQuantizedOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                            const Node& node,
                                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> scales_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], scales_shape, logger), "Cannot get scales shape");
  const auto input_rank = input_shape.size();

  int32_t input_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input data type");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val indices = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val scales = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val common_options = emscripten::val::object();

  NodeAttrHelper helper(node);
  const int32_t bits = helper.Get("bits", 4);
  const uint32_t gather_axis = SafeInt<uint32_t>(HandleNegativeAxis(helper.Get("gather_axis", 0), input_rank));

  // GatherBlockQuantized only supports block-wise quantization, the input and scales should have the same rank.
  // So we don't need to reshape scales for broadcasting.
  emscripten::val zero_points = emscripten::val::undefined();
  if (TensorExists(input_defs, 3)) {  // zero_points
    zero_points = model_builder.GetOperand(input_defs[3]->Name());
  } else {
    const uint8_t default_zero_point = bits == 4 ? 0 : 128;
    // Create a constant for zero_points, which has the same shape as scales and same type as input.
    zero_points = model_builder.CreateOrGetConstant<uint8_t>(input_type,
                                                             default_zero_point,
                                                             GetNarrowedIntFromInt64<uint32_t>(scales_shape));
  }

  // dequantized_input = DequantizeLinear(input, scales, zero_points)
  common_options.set("label", node.Name() + "_dequantize_input");
  emscripten::val dequantized_input = model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear",
                                                                                       input,
                                                                                       scales,
                                                                                       zero_points,
                                                                                       common_options);

  // output = Gather(dequantized_input, indices, axis=gather_axis)
  common_options.set("label", node.Name() + "_gather");
  common_options.set("axis", gather_axis);
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("gather",
                                                                            dequantized_input,
                                                                            indices,
                                                                            common_options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool GatherBlockQuantizedOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                                      const Node& node,
                                                      const WebnnDeviceType /* device_type */,
                                                      const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  const int32_t bits = helper.Get("bits", 4);
  const int32_t block_size = helper.Get("block_size", 128);

  if (bits != 4 && bits != 8) {
    LOGS(logger, VERBOSE) << "GatherBlockQuantized only supports bits==4 or 8.";
    return false;
  }

  if (block_size < 16 || ((block_size - 1) & block_size) != 0) {
    LOGS(logger, VERBOSE) << "GatherBlockQuantized: 'block_size' must be a power of 2 and not less than 16.";
    return false;
  }

  return true;
}

bool GatherBlockQuantizedOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                           const emscripten::val& wnn_limits,
                                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> scales_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !GetShape(*input_defs[2], scales_shape, logger)) {
    return false;
  }

  if (input_shape.size() != scales_shape.size()) {
    LOGS(logger, VERBOSE) << "GatherBlockQuantized: input and scales must have the same rank.";
    return false;
  }

  const std::string_view op_type = node.OpType();
  int32_t input_type = 0;
  int32_t scales_type = 0;
  if (!GetType(*input_defs[0], input_type, logger) ||
      !GetType(*input_defs[2], scales_type, logger)) {
    return false;
  }

  // Only need to check the input data type of ops that consume the inputs of GatherBlockQuantized.
  // WebNN dequantizeLinear's input should be same as input. WebNN gather's input should be same as scales input.
  return IsDataTypeSupportedByWebNNOp(op_type, "dequantizeLinear", input_type, wnn_limits, "input", "data", logger) &&
         IsDataTypeSupportedByWebNNOp(op_type, "gather", scales_type, wnn_limits, "input", "scales", logger);

  return true;
}

bool GatherBlockQuantizedOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                            const emscripten::val& wnn_limits,
                                                            const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  // Only need to check the output data type of ops that produce the output of GatherBlockQuantized.
  // WebNN gather's output should be same as GatherBlockQuantized's output.
  return IsDataTypeSupportedByWebNNOp(op_type, "gather", output_type, wnn_limits, "output", "output", logger);
}

void CreateGatherBlockQuantizedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherBlockQuantizedOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
