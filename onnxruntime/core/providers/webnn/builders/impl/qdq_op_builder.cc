// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "core/providers/webnn/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class QDQOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

Status QDQOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;
  std::vector<uint32_t> zero_point_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");
  int32_t input_type = 0;
  int32_t output_type = 0;
  int32_t zero_point_type = 0;
  bool has_zero_point = false;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input data type");
  ORT_RETURN_IF_NOT(GetType(*output_defs[0], output_type, logger), "Cannot get output data type");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scale = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val zero_point = emscripten::val::null();

  if (TensorExists(input_defs, 2)) {
    zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
    has_zero_point = true;
  } else {
    // DequantizeLinear: x_zero_point's data type equals to input data type
    // QuantizeLinear: x_zero_point's data type equals to output data type
    zero_point_type = op_type == "DequantizeLinear" ? input_type : output_type;
  }

  const auto input_rank = input_shape.size();
  NodeAttrHelper helper(node);
  int32_t block_size = helper.Get("block_size", 0);
  int32_t axis = helper.Get("axis", 1);
  if (axis < 0) {
    axis = SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank));
  }

  // For per-axis quantization/dequantization and axis is not equal to input_rank - 1,
  // we need to reshape the scale and zero_point tensors to make them broadcastable with the input tensor.
  if (scale_shape.size() == 1 && input_rank > 1 &&
      block_size == 0 && axis != static_cast<int32_t>(input_rank - 1)) {
    // Insert ones before and after the axis dimension for broadcasting of scale tensor.
    std::vector<uint32_t> target_shape{SafeInt<uint32_t>(input_shape[axis])};
    target_shape.insert(target_shape.begin(), axis, 1);
    target_shape.insert(target_shape.end(), input_rank - axis - 1, 1);
    // zero_point has the same shape as the scale tensor.
    zero_point_shape = target_shape;
    emscripten::val reshape_scale_options = emscripten::val::object();
    reshape_scale_options.set("label", node.Name() + "_reshape_scale");
    scale = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                             scale,
                                                             emscripten::val::array(target_shape),
                                                             reshape_scale_options);

    if (has_zero_point) {
      // Reshape the zero_point tensor too.
      emscripten::val reshape_zero_point_options = emscripten::val::object();
      reshape_zero_point_options.set("label", node.Name() + "_reshape_zero_point");
      zero_point = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                    zero_point,
                                                                    emscripten::val::array(target_shape),
                                                                    reshape_zero_point_options);
    }
  }

  // If zero_point is not provided, create a zero constant with the same shape as the scale tensor.
  if (!has_zero_point) {
    if (zero_point_shape.empty()) {
      // zero_point has the same shape as the scale tensor.
      zero_point_shape = GetNarrowedIntFromInt64<uint32_t>(scale_shape);
    }
    // Create a zero constant with the same shape as the scale tensor.
    // The zero value has been pre-processed in the CreateOrGetConstant function,
    // so the type of T is not relevant here.
    zero_point = model_builder.CreateOrGetConstant<uint8_t>(zero_point_type, 0, zero_point_shape);
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  ORT_RETURN_IF(webnn_op_type.empty(), "Cannot get WebNN op type");

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      std::string(webnn_op_type).c_str(), input, scale, zero_point, options);

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.
bool QDQOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                     const Node& node,
                                     const WebnnDeviceType /* device_type */,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;

  if (!GetShape(*input_defs[0], input_shape, logger) || !GetShape(*input_defs[1], scale_shape, logger)) {
    return false;
  }

  // WebNN requires the scale_shape to be a subsample of the input_shape.
  if (scale_shape.size() > input_shape.size()) {
    LOGS(logger, VERBOSE) << "The rank of scale is larger than the rank of input";
    return false;
  }

  for (size_t i = 0; i < scale_shape.size(); ++i) {
    auto scale_dim = scale_shape[scale_shape.size() - i - 1];
    auto input_dim = input_shape[input_shape.size() - i - 1];
    if (input_dim % scale_dim != 0) {
      LOGS(logger, VERBOSE) << "The shape of scale is not a subsample of the shape of input";
      return false;
    }
  }

  return true;
}

bool QDQOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                          const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input0_type = 0;  // input data type
  int32_t input1_type = 0;  // x_scale data type
  int32_t input2_type = 0;  // x_zero_point data type
  bool has_input2 = TensorExists(input_defs, 2);

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger) ||
      (has_input2 && !GetType(*input_defs[2], input2_type, logger))) {
    return false;
  }

  return IsInputRankSupportedByOp(node, wnn_limits, logger) &&
         IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "input", "x", logger) &&
         IsDataTypeSupportedByOp(op_type, input1_type, wnn_limits, "scale", "x_scale", logger) &&
         (!has_input2 || IsDataTypeSupportedByOp(op_type, input2_type, wnn_limits, "zeroPoint", "x_zero_point", logger));
}

void CreateQDQOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "DequantizeLinear",
          "QuantizeLinear",
      };

  op_registrations.builders.push_back(std::make_unique<QDQOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
