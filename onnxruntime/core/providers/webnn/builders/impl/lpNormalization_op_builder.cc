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

class LpNormalizationOpBuilder : public BaseOpBuilder {
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

Status LpNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                       const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val wnn_builder = model_builder.GetBuilder();

  int32_t input_type;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input type");
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto rank = input_shape.size();

  NodeAttrHelper helper(node);
  const int64_t p = helper.Get("p", static_cast<int64_t>(2));
  int64_t axis = helper.Get("axis", static_cast<int64_t>(-1));
  axis = HandleNegativeAxis(axis, rank);
  std::vector<uint32_t> axes{SafeInt<uint32_t>(axis)};

  // WebNN has no dedicated LpNormalization operator, so decompose it into:
  //   output = input / max(Lp_norm(input, axis), eps)
  //   p==2: norm = reduceL2(input, {axis});  p==1: norm = reduceL1(input, {axis})
  // keepDimensions=true so the division broadcasts back over the reduced axis. The max(., eps) guard
  // avoids divide-by-zero; because input==0 along the axis implies norm==0, 0/eps == 0, which matches
  // the ONNX "when the Lp norm is zero, the output is zero" rule.
  emscripten::val reduce_options = emscripten::val::object();
  reduce_options.set("axes", emscripten::val::array(axes));
  reduce_options.set("keepDimensions", true);
  reduce_options.set("label", node.Name() + "_reduce");
  const char* reduce_op = (p == 1) ? "reduceL1" : "reduceL2";
  emscripten::val norm = wnn_builder.call<emscripten::val>(reduce_op, input, reduce_options);

  // The epsilon must be representable in the input's data type. Use the smallest positive normalized
  // value for each type to avoid divide-by-zero while preserving maximum numeric range.
  const float eps = (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)
                        ? std::numeric_limits<MLFloat16>::min().ToFloat()
                        : std::numeric_limits<float>::min();
  emscripten::val eps_constant = model_builder.CreateOrGetConstant<float>(input_type, eps);
  emscripten::val common_options = emscripten::val::object();
  common_options.set("label", node.Name() + "_max");
  emscripten::val denom = wnn_builder.call<emscripten::val>("max", norm, eps_constant, common_options);

  common_options.set("label", node.Name() + "_div");
  emscripten::val output = wnn_builder.call<emscripten::val>("div", input, denom, common_options);

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool LpNormalizationOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();

  int32_t input_type = 0;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }

  // LpNormalization is decomposed into reduceL1/reduceL2 + max + div. Check each decomposed WebNN op
  // supports the input data type.
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, input_type, wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }
  // reduceL2 consumes the input; div consumes the input as its first operand ("a").
  return IsRankSupportedByWebNNOp(wnn_limits, "reduceL2", "input", input_shape.size(), node.Name(), logger) &&
         IsRankSupportedByWebNNOp(wnn_limits, "div", "a", input_shape.size(), node.Name(), logger);
}

bool LpNormalizationOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                       const emscripten::val& wnn_limits,
                                                       const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  // Check if the output data type is supported by every decomposed WebNN op.
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, output_type, wnn_limits, "output", "output", logger)) {
      return false;
    }
  }

  return true;
}

void CreateLpNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LpNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
