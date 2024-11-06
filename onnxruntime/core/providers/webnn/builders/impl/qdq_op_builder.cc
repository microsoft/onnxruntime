// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
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
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

Status QDQOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");
  int32_t input_type = 0;
  int32_t output_type = 0;
  int32_t zero_point_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input data type");
  ORT_RETURN_IF_NOT(GetType(*output_defs[0], output_type, logger), "Cannot get output data type");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scale = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val zero_point = emscripten::val::null();

  if (input_defs.size() == 3 && input_defs[2]->Exists()) {
    zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
  } else {
    // DequantizeLinear: x_zero_point's data type equals to input data type
    // QuantizeLinear: x_zero_point's data type equals to output data type
    // WebNN requires the zero_point to have the same shape as the scale
    zero_point_type = op_type == "DequantizeLinear" ? input_type : output_type;
    const auto zero_point_shape = GetVecUint32FromVecInt64(scale_shape);
    zero_point = model_builder.GetZeroConstant(zero_point_type, zero_point_shape);
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  std::string webnn_op_type;
  ORT_RETURN_IF_NOT(GetWebNNOpType(op_type, webnn_op_type), "Cannot get WebNN op type");
  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>(webnn_op_type.c_str(), input, scale, zero_point, options);

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

bool QDQOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type = 0;  // input data type
  int32_t input1_type = 0;  // x_scale data type
  int32_t input2_type = 0;  // x_zero_point data type
  bool has_input2 = input_defs.size() > 2 && input_defs[2]->Exists();

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger) ||
      (has_input2 && !GetType(*input_defs[2], input2_type, logger))) {
    return false;
  }

  return IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "input", "x", logger) &&
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
