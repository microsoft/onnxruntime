// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/webnn/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class LogicalOpBuilder : public BaseOpBuilder {
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

const std::unordered_set<std::string> UNARY_LOGICAL_OPS = {
    "IsInf",
    "IsNaN",
    "Not",
};

bool IsUnaryOp(const std::string& op_type) {
  return UNARY_LOGICAL_OPS.count(op_type) > 0;
}

// Add operator related.

Status LogicalOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  emscripten::val input0 = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val input1 = emscripten::val::undefined();

  emscripten::val output = emscripten::val::object();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  ORT_RETURN_IF(webnn_op_type.empty(), "Cannot get WebNN op type");

  if (IsUnaryOp(op_type)) {
    if (op_type == "IsInf") {
      // IsInf has two attributes to control whether to detect positive or negative infinity.
      // WebNN's isInfinite detects both positive and negative infinity, so we need to emulate
      // the rest behaviors.
      const auto& helper = NodeAttrHelper(node);
      const bool detect_positive = helper.Get("detect_positive", 1) != 0;
      const bool detect_negative = helper.Get("detect_negative", 1) != 0;

      emscripten::val inf_constant = emscripten::val::undefined();
      if (!detect_positive || !detect_negative) {
        int32_t input0_type;
        ORT_RETURN_IF_NOT(GetType(*input_defs[0], input0_type, logger), "Cannot get data type of input");
        const bool is_float16 = (input0_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

        // WebNN only supports float and float16 data types for isInfinite.
        ORT_RETURN_IF_NOT(input0_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT || is_float16,
                          "WebNN only supports float32 and float16 data types for IsInf.");

        // Flag to decide whether to use positive or negative infinity.
        const bool use_pos_inf = detect_positive || (!detect_positive && !detect_negative);

        emscripten::val inf_constant_desc = emscripten::val::object();
        inf_constant_desc.set("shape", emscripten::val::array());
        inf_constant_desc.set("dataType", is_float16 ? "float16" : "float32");

        emscripten::val inf_buffer = emscripten::val::undefined();
        if (is_float16 && !model_builder.IsFloat16ArrayAvailable()) {
          // Fallback to Uint16Array for float16.
          inf_buffer = emscripten::val::global("Uint16Array").new_(1);
          inf_buffer.set(0, emscripten::val(use_pos_inf ? 0x7C00 : 0xFC00));  // +inf or -inf in float16
        } else {
          inf_buffer = emscripten::val::global(is_float16 ? "Float16Array" : "Float32Array").new_(1);
          emscripten::val inf_value = use_pos_inf
                                          ? emscripten::val::global("Number")["POSITIVE_INFINITY"]
                                          : emscripten::val::global("Number")["NEGATIVE_INFINITY"];

          inf_buffer.set(0, inf_value);
        }

        inf_constant = model_builder.GetBuilder().call<emscripten::val>("constant", inf_constant_desc, inf_buffer);
      }

      if (detect_positive && detect_negative) {
        // Both positive and negative infinity are detected, use isInfinite directly.
        output = model_builder.GetBuilder().call<emscripten::val>("isInfinite", input0, options);
      } else if (detect_positive || detect_negative) {
        // Only positive or negative infinity is detected, use equal(input, +inf) or equal(input, -inf).
        output = model_builder.GetBuilder().call<emscripten::val>("equal", input0, inf_constant, options);
      } else {
        // Both positive and negative infinity are not detected, return all false, use greater(input, inf_constant).
        output = model_builder.GetBuilder().call<emscripten::val>("greater", input0, inf_constant, options);
      }
    } else {
      output = model_builder.GetBuilder().call<emscripten::val>(std::string(webnn_op_type).c_str(), input0, options);
    }
  } else {
    input1 = model_builder.GetOperand(input_defs[1]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>(
        std::string(webnn_op_type).c_str(), input0, input1, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool LogicalOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  size_t expected_input_count = IsUnaryOp(op_type) ? 1 : 2;
  if (input_defs.size() != expected_input_count) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] expected input count: "
                          << expected_input_count << ", actual: " << input_defs.size();
    return false;
  }

  return true;
}

bool LogicalOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                              const emscripten::val& wnn_limits,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input0_type;
  int32_t input1_type;

  if (!GetType(*input_defs[0], input0_type, logger))
    return false;

  if (!IsUnaryOp(std::string(op_type))) {
    if (!GetType(*input_defs[1], input1_type, logger))
      return false;
    std::array<int32_t, 2> input_types{input0_type, input1_type};
    if (!AreDataTypesSame(op_type, input_types, logger)) {
      return false;
    }
  }

  const std::string_view webnn_input_name = GetWebNNOpFirstInputName(op_type);
  std::string onnx_input_name = IsUnaryOp(std::string(op_type)) ? "X" : "A";
  return IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, webnn_input_name, onnx_input_name, logger) &&
         IsInputRankSupportedByOp(node, wnn_limits, logger);
}

void CreateLogicalOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "And",
          "Equal",
          "IsInf",
          "IsNaN",
          "Greater",
          "GreaterOrEqual",
          "Less",
          "LessOrEqual",
          "Not",
          "Or",
          "Xor",
      };

  op_registrations.builders.push_back(std::make_unique<LogicalOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
