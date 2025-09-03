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

class DynamicQuantizeLinearOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// DynamicQuantizeLinear is a function defined as follows:
// DynamicQuantizeLinear (x) => (y, y_scale, y_zero_point)
// {
//    Q_Min = Constant <value: tensor = float {0}> ()
//    Q_Max = Constant <value: tensor = float {255}> ()
//    X_Min = ReduceMin <keepdims: int = 0> (x)
//    X_Min_Adjusted = Min (X_Min, Q_Min)
//    X_Max = ReduceMax <keepdims: int = 0> (x)
//    X_Max_Adjusted = Max (X_Max, Q_Min)
//    X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
//    Scale = Div (X_Range, Q_Max)
//    Min_Scaled = Div (X_Min_Adjusted, Scale)
//    Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
//    Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
//    Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
//    Zeropoint = Cast <to: int = 2> (Rounded_ZeroPoint_FP)
//    y_scale = Identity (Scale) (Skip in WebNN)
//    y_zero_point = Identity (Zeropoint) (Skip in WebNN)
//    y = QuantizeLinear (x, Scale, Zeropoint)
// }
Status DynamicQuantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                             const Node& node,
                                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val common_options = emscripten::val::object();

  // Q_Min = Constant <value: tensor = float {0}> ()
  emscripten::val q_min = model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0.0f);
  // Q_Max = Constant <value: tensor = float {255}> ()
  emscripten::val q_max = model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 255.0f);

  // X_Min = ReduceMin <keepdims: int = 0> (x)
  common_options.set("label", node.Name() + "_x_min");
  emscripten::val x_min = model_builder.GetBuilder().call<emscripten::val>("reduceMin", input, common_options);

  // X_Min_Adjusted = Min (X_Min, Q_Min)
  common_options.set("label", node.Name() + "_x_min_adjusted");
  emscripten::val x_min_adjusted = model_builder.GetBuilder().call<emscripten::val>("min", x_min, q_min, common_options);

  // X_Max = ReduceMax <keepdims: int = 0> (x)
  common_options.set("label", node.Name() + "_x_max");
  emscripten::val x_max = model_builder.GetBuilder().call<emscripten::val>("reduceMax", input, common_options);

  // X_Max_Adjusted = Max (X_Max, Q_Min)
  common_options.set("label", node.Name() + "_x_max_adjusted");
  emscripten::val x_max_adjusted = model_builder.GetBuilder().call<emscripten::val>(
      "max", x_max, q_min, common_options);

  // X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
  common_options.set("label", node.Name() + "_x_range");
  emscripten::val x_range = model_builder.GetBuilder().call<emscripten::val>(
      "sub", x_max_adjusted, x_min_adjusted, common_options);

  // Scale = Div (X_Range, Q_Max)
  common_options.set("label", node.Name() + "_scale");
  emscripten::val scale = model_builder.GetBuilder().call<emscripten::val>("div", x_range, q_max, common_options);

  // Min_Scaled = Div (X_Min_Adjusted, Scale)
  common_options.set("label", node.Name() + "_min_scaled");
  emscripten::val min_scaled = model_builder.GetBuilder().call<emscripten::val>(
      "div", x_min_adjusted, scale, common_options);

  // Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
  common_options.set("label", node.Name() + "_initial_zero_point_fp");
  emscripten::val initial_zero_point_fp = model_builder.GetBuilder().call<emscripten::val>(
      "sub", q_min, min_scaled, common_options);

  // Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
  emscripten::val clip_options = emscripten::val::object();
  clip_options.set("label", node.Name() + "_clipped_zero_point_fp");
  clip_options.set("minValue", 0);
  clip_options.set("maxValue", 255);
  emscripten::val clipped_zero_point_fp = model_builder.GetBuilder().call<emscripten::val>(
      "clamp", initial_zero_point_fp, clip_options);

  // Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
  common_options.set("label", node.Name() + "_rounded_zero_point_fp");
  emscripten::val rounded_zero_point_fp = model_builder.GetBuilder().call<emscripten::val>(
      "roundEven", clipped_zero_point_fp, common_options);

  // Zeropoint = Cast <to: int = 2> (Rounded_ZeroPoint_FP)
  // to: int = 2 means cast to uint8
  common_options.set("label", node.Name() + "_zero_point");
  emscripten::val zero_point = model_builder.GetBuilder().call<emscripten::val>(
      "cast", rounded_zero_point_fp, emscripten::val("uint8"), common_options);

  // The WebNN quantizeLinear op requires the scale and zero_point tensors to have the same rank as the input tensor.
  // The scale and zero_point outputs are both scalars, so we need to reshape them to match the input rank.
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto input_rank = input_shape.size();
  emscripten::val new_scale = scale;
  emscripten::val new_zero_point = zero_point;
  if (input_rank > 0) {
    std::vector<uint32_t> new_shape(input_rank, 1);
    common_options.set("label", node.Name() + "_reshape_scale");
    new_scale = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", scale, emscripten::val::array(new_shape), common_options);

    common_options.set("label", node.Name() + "_reshape_zero_point");
    new_zero_point = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", zero_point, emscripten::val::array(new_shape), common_options);
  }

  // y = QuantizeLinear (x, Scale, Zeropoint)
  common_options.set("label", node.Name() + "_quantize_linear");
  emscripten::val y = model_builder.GetBuilder().call<emscripten::val>(
      "quantizeLinear", input, new_scale, new_zero_point, common_options);

  // Add output: y
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(y));
  // Add output: y_scale
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(scale));
  // Add output: y_zero_point
  model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(zero_point));

  return Status::OK();
}

// Operator support related.
bool DynamicQuantizeLinearOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                            const emscripten::val& wnn_limits,
                                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  int32_t input_type = 0;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << "DynamicQuantizeLinear only supports input data type float.";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }
  // It's complicated to check all the decomposed ops' input rank support.
  // Ensure at least the first input rank is supported by the decomposed ops.
  // (reduceMax, reduceMin and quantizeLinear accept the first input).
  const std::array<std::string_view, 3> operations = {"reduceMax", "reduceMin", "quantizeLinear"};
  for (const auto& op : operations) {
    if (!IsInputRankSupported(wnn_limits, op, "input", input_shape.size(), node.Name(), logger)) {
      return false;
    }
  }

  return true;
}

bool DynamicQuantizeLinearOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                             const emscripten::val& wnn_limits,
                                                             const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t y_type, y_scale_type, y_zero_point_type;
  if (!GetType(*output_defs[0], y_type, logger) ||
      !GetType(*output_defs[1], y_scale_type, logger) ||
      !GetType(*output_defs[2], y_zero_point_type, logger)) {
    return false;
  }

  // Only need to check the output data type of ops that produce the outputs of DynamicQuantizeLinear.
  // 1. QuantizeLinear -> y (uint8)
  // 2. Div -> y_scale (float32) (skip it as WebNN should support it by default)
  // 3. Cast -> y_zero_point (uint8)
  return IsDataTypeSupportedByWebNNOp(op_type, "quantizeLinear", y_type, wnn_limits, "output", "y", logger) &&
         IsDataTypeSupportedByWebNNOp(op_type, "cast", y_zero_point_type, wnn_limits, "output", "y_zero_point", logger);
}

void CreateDynamicQuantizeLinearOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DynamicQuantizeLinearOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
