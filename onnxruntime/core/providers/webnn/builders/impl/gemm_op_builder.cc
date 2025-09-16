// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class GemmOpBuilder : public BaseOpBuilder {
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

// Helper function
common::Status ProcessZeroPointAndScale(
    gsl::span<const int64_t> input_shape,
    const size_t zero_point_index,
    ModelBuilder& model_builder,
    const Node& node,
    const int32_t zero_point_type,
    const logging::Logger& logger,
    emscripten::val& zero_point,
    emscripten::val& scale) {
  ORT_RETURN_IF_NOT((zero_point_index == 2 || zero_point_index == 3), "zero_point_index should be 2 or 3.");
  // The WebNN dequantizeLinear op requires the scale and zero_point tensors to have the
  // same rank as the input tensor. So we need to reshape the zero_point tensors
  // to match the input rank.

  // Initially set target zero point shape to [1, 1, ..., 1].
  std::vector<uint32_t> target_zero_point_shape(input_shape.size(), 1);
  if (TensorExists(node.InputDefs(), zero_point_index)) {
    const auto& zero_point_arg = node.InputDefs()[zero_point_index];
    const auto zero_point_name = zero_point_arg->Name();
    zero_point = model_builder.GetOperand(zero_point_name);
    std::vector<int64_t> zero_point_shape;
    ORT_RETURN_IF_NOT(GetShape(*zero_point_arg, zero_point_shape, logger),
                      "Cannot get shape of zero_point: " + zero_point_name);

    // Assume zero_point shape has the same shape as input or is 1-D.
    // Adjust zero_point shape only if it's 1-D to match the input shape.
    const auto target_zero_point_rank = target_zero_point_shape.size();
    const auto zero_point_rank = zero_point_shape.size();
    if (zero_point_rank == 1 && zero_point_shape[0] != 1) {
      // For a_zero_point, its shape may be:
      // - [K] when the A input shape is [1, K], it should be reshaped to [1, K]
      // - [M] when the A input shape is [M, K], it should be reshaped to [M, 1]
      // For b_zero_point, its shape may be:
      // - [K] when the B input shape is [K, 1], it should be reshaped to [K, 1]
      // - [N] when the B input shape is [K, N], it should be reshaped to [1, N]
      const uint32_t safe_zero_point_shape = SafeInt<uint32_t>(zero_point_shape[0]);
      if (safe_zero_point_shape == input_shape[0]) {
        target_zero_point_shape[0] = safe_zero_point_shape;
      } else {
        target_zero_point_shape.back() = safe_zero_point_shape;
      }
    }

    if (zero_point_rank != target_zero_point_rank) {
      // Reshape zero_point to the target shape.
      emscripten::val common_options = emscripten::val::object();
      common_options.set("label", node.Name() + "_reshape_zero_point_" + zero_point_name);
      zero_point = model_builder.GetBuilder().call<emscripten::val>(
          "reshape", zero_point, emscripten::val::array(target_zero_point_shape), common_options);
    } else {
      // Set target_zero_point_shape to the actual shape of zero_point.
      target_zero_point_shape = GetNarrowedIntFromInt64<uint32_t>(zero_point_shape);
    }
  } else {
    // If zero_point is not provided, create default zero_point with the same rank as input.
    zero_point = model_builder.CreateOrGetConstant<uint8_t>(zero_point_type, 0, target_zero_point_shape);
  }

  // Create scale with the same shape as the zero point.
  scale = model_builder.CreateOrGetConstant<float>(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 1.0f, target_zero_point_shape);

  return Status::OK();
}

// Add operator related.
Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  std::vector<int64_t> a_shape;
  std::vector<int64_t> b_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[a_idx], a_shape, logger), "Can not get shape of A");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[b_idx], b_shape, logger), "Can not get shape of B");

  emscripten::val a = model_builder.GetOperand(node.InputDefs()[a_idx]->Name());
  emscripten::val b = model_builder.GetOperand(node.InputDefs()[b_idx]->Name());
  emscripten::val output = emscripten::val::object();
  emscripten::val common_options = emscripten::val::object();

  // MatMul and MatMulInteger in ONNX allow 1-D inputs while matmul in WebNN only supports at least 2-D inputs.
  // We can support 1-D inputs by reshaping them to 2-D. We don't care Gemm here because it only provides 2-D inputs.
  const auto a_rank = a_shape.size();
  const auto b_rank = b_shape.size();
  const bool is_input_1d = (a_rank == 1 || b_rank == 1);

  // If the input A is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
  if (a_rank == 1) {
    a_shape.insert(a_shape.begin(), 1);
    emscripten::val a_shape_arr = emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(a_shape));
    common_options.set("label", node.Name() + "_reshape_a");
    a = model_builder.GetBuilder().call<emscripten::val>("reshape", a, a_shape_arr, common_options);
  }
  // If the input B is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
  if (b_rank == 1) {
    b_shape.push_back(1);
    emscripten::val b_shape_arr = emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(b_shape));
    common_options.set("label", node.Name() + "_reshape_b");
    b = model_builder.GetBuilder().call<emscripten::val>("reshape", b, b_shape_arr, common_options);
  }

  if (op_type == "MatMul") {
    common_options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("matmul", a, b, common_options);
  } else if (op_type == "MatMulInteger") {
    // WebNN doesn't provide a dedicated op for MatMulInteger, it can be simply decomposed by
    // DequantizeLinear A, B -> MatMul -> Cast (to int32)
    int32_t a_type;
    ORT_RETURN_IF_NOT(GetType(*input_defs[0], a_type, logger), "Cannot get data type of input A");

    emscripten::val a_zero_point, b_zero_point, a_scale, b_scale;

    ORT_RETURN_IF_ERROR(
        ProcessZeroPointAndScale(a_shape, 2, model_builder, node, a_type, logger, a_zero_point, a_scale));
    ORT_RETURN_IF_ERROR(
        ProcessZeroPointAndScale(b_shape, 3, model_builder, node, a_type, logger, b_zero_point, b_scale));

    // Dequantize A to Float32
    common_options.set("label", node.Name() + "_dequantized_a");
    emscripten::val dequantized_a = model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear",
                                                                                     a,
                                                                                     a_scale,
                                                                                     a_zero_point,
                                                                                     common_options);
    // Dequantize B to Float32
    common_options.set("label", node.Name() + "_dequantized_b");
    emscripten::val dequantized_b = model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear",
                                                                                     b,
                                                                                     b_scale,
                                                                                     b_zero_point,
                                                                                     common_options);
    // MatMul dequantized A and B
    common_options.set("label", node.Name() + "_matmul_dequantized_ab");
    emscripten::val matmul_dequantized_ab = model_builder.GetBuilder().call<emscripten::val>("matmul",
                                                                                             dequantized_a,
                                                                                             dequantized_b,
                                                                                             common_options);
    // Cast matmul_dequantized_ab to int32
    common_options.set("label", node.Name() + "_cast_output");
    output = model_builder.GetBuilder().call<emscripten::val>("cast",
                                                              matmul_dequantized_ab,
                                                              emscripten::val("int32"),
                                                              common_options);
  } else {  // Gemm
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    common_options.set("aTranspose", emscripten::val(transA == 1));
    const auto transB = helper.Get("transB", 0);
    common_options.set("bTranspose", emscripten::val(transB == 1));
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    common_options.set("alpha", alpha);
    common_options.set("beta", beta);

    // Add bias if present.
    if (input_defs.size() > 2) {
      common_options.set("c", model_builder.GetOperand(node.InputDefs()[c_idx]->Name()));
    }

    common_options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("gemm", a, b, common_options);
  }

  // For MatMul or MatMulInteger, if either the A or B input is 1-D,
  // we need to reshape the output back to its original shape.
  if (is_input_1d && (op_type == "MatMul" || op_type == "MatMulInteger")) {
    std::vector<int64_t> output_shape;
    ORT_RETURN_IF_NOT(GetShape(*node.OutputDefs()[0], output_shape, logger), "Can not get output shape");
    common_options.set("label", node.Name() + "_reshape_output");
    emscripten::val output_shape_arr = emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(output_shape));
    output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                              output,
                                                              output_shape_arr,
                                                              common_options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool GemmOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                      const Node& node,
                                      const WebnnDeviceType /* device_type */,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs(node.InputDefs());
  const size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C

  std::vector<int64_t> a_shape;
  if (!GetShape(*input_defs[a_idx], a_shape, logger))
    return false;
  if (Product(a_shape) == 0) {
    LOGS(logger, VERBOSE) << "A must be non-empty";
    return false;
  }

  std::vector<int64_t> b_shape;
  if (!GetShape(*input_defs[b_idx], b_shape, logger))
    return false;
  if (Product(b_shape) == 0) {
    LOGS(logger, VERBOSE) << "B must be non-empty";
    return false;
  }

  if (op_type == "Gemm") {
    if (a_shape.size() != 2 || b_shape.size() != 2) {
      LOGS(logger, VERBOSE) << "A and B must be 2D for Gemm";
      return false;
    }

    // C of Gemm.
    if (input_defs.size() == 3) {
      std::vector<int64_t> c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape, logger))
        return false;

      size_t c_dim = c_shape.size();

      if (c_dim > 1) {
        // TODO: Supports other shape of C.
        // Currently WebNN implementation in Chromium only supports 1-D C.
        return false;
      }
      if (c_dim == 0) {
        LOGS(logger, VERBOSE) << "C of Gemm is a scalar";
      } else {
        auto c_size = c_shape[c_dim - 1];
        NodeAttrHelper helper(node);
        const auto transB = helper.Get("transB", 0);
        if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
          LOGS(logger, VERBOSE) << "C of Gemm must be a vector of b_shape["
                                << (transB == 0 ? "1" : "0") << "]"
                                << " b_shape: [" << b_shape[0] << ", " << b_shape[1] << "]"
                                << " c_size: " << c_size;

          return false;
        }
      }
    }
  }

  return true;
}

bool GemmOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                           const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input0_type;  // A data type
  int32_t input1_type;  // B data type
  int32_t input2_type;  // C or a_zero_point data type
  int32_t input3_type;  // b_zero_point data type
  bool has_input2 = TensorExists(input_defs, 2);
  bool has_input3 = TensorExists(input_defs, 3);

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger) ||
      (has_input2 && !GetType(*input_defs[2], input2_type, logger)) ||
      (has_input3 && !GetType(*input_defs[3], input3_type, logger))) {
    return false;
  }

  InlinedVector<int32_t, 4> input_types = {input0_type, input1_type};
  if (has_input2) {
    input_types.push_back(input2_type);
  }
  if (has_input3) {
    input_types.push_back(input3_type);
  }
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (op_type == "Gemm") {
    return IsInputRankSupportedByOp(node, wnn_limits, logger) &&
           IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "a", "A", logger);
  } else if (op_type == "MatMulInteger") {
    // Check up to 4 inputs for MatMulInteger
    for (size_t i = 0; i < input_defs.size(); ++i) {
      std::vector<int64_t> shape;
      if (!GetShape(*input_defs[i], shape, logger)) {
        return false;
      }

      // We made workaround to support 1D for input A and B, skip further checks if they are 1D
      if (i <= 1 && shape.size() == 1) {
        continue;
      }

      // For DequantizeLinear, input indices: 0 (x), 1 (scale), 2 (zero_point)
      if (!IsInputRankSupported(wnn_limits, "dequantizeLinear",
                                (i < 2) ? "input" : "zeroPoint",
                                shape.size(), node.Name(), logger)) {
        return false;
      }
    }
    return IsDataTypeSupportedByOp("DequantizeLinear", input0_type, wnn_limits, "input", "x", logger);
  } else {  // MatMul
    for (int i = 0; i < 2; ++i) {
      std::vector<int64_t> shape;
      if (!GetShape(*input_defs[i], shape, logger)) {
        return false;
      }

      if (shape.size() == 1) {
        continue;
      }

      if (!IsInputRankSupported(wnn_limits, "matmul", (i == 0) ? "a" : "b", shape.size(), node.Name(), logger)) {
        return false;
      }
    }
    return IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "a", "A", logger);
  }
}

bool GemmOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                            const logging::Logger& logger) const {
  const auto& output = *node.OutputDefs()[0];
  const std::string_view op_type = node.OpType();
  int32_t output_type;
  if (!GetType(output, output_type, logger)) {
    return false;
  }

  if (op_type == "MatMulInteger") {
    // The last decomposed op of MatMulInteger is Cast, and so
    // we only need to ensure it supports the output_type.
    return IsDataTypeSupportedByOp("Cast", output_type, wnn_limits, "output", "Output", logger);
  } else {
    return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, "output", "Output", logger);
  }
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Gemm",
          "MatMul",
          "MatMulInteger",
      };

  op_registrations.builders.push_back(std::make_unique<GemmOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}
}  // namespace webnn
}  // namespace onnxruntime
