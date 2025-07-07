// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class MatMulNBitsBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

void MatMulNBitsBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Inputs B and zero_points (if present) must be initializers. If they are of type uint8,
  // they should be stored as uint4 constants in WebNN. Therefore, we skip them here and
  // delay their registration as WebNN constants.
  const auto& input_defs = node.InputDefs();
  if (input_defs[1]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // B
    if (TensorExists(input_defs, 3)) {
      model_builder.AddInitializerToSkip(input_defs[3]->Name());  // zero_points
    }
  }
}

// WebNN doesn't provide a dedicated op for MatMulNBits, it can be simply decomposed by
// DequantizeLinear + Transpose + MatMul. Given that the CPU EP currently only supports
// 4-bit quantization, we only handle 4-bit quantization here.
//
// To align with WebNN's dequantizeLinear op contraints, the following transformations are
// required for MatMulNBits inputs:
// 1. B: must be a constant initializer and registered as a 'uint4' WebNN constant with shape
//       [N, n_blocks_per_col, blob_size * 2].
// 2. scales: reshape it to [N, n_blocks_per_col, 1].
// 3. zero_points: it has the same shape as reshaped scales. If it presents, it must be a
//                 constant initializer and registered as a 'uint4' WebNN constant.
//                 Otherwise, it must be registered as a 'uint4' WebNN constant with default value 8.
Status MatMulNBitsBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const auto& initializers = model_builder.GetInitializerTensors();

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scales = model_builder.GetOperand(input_defs[2]->Name());

  std::vector<int64_t> B_shape;  // [N, n_blocks_per_col, blob_size]
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], B_shape, logger), "Cannot get B shape");

  NodeAttrHelper helper(node);
  const uint32_t K = helper.Get("K", 0);
  const uint32_t N = helper.Get("N", 0);
  const uint32_t n_blocks_per_col = SafeInt<uint32_t>(B_shape[1]);
  const uint32_t double_blob_size = SafeInt<uint32_t>(B_shape[2] * 2);

  // Prepare DequantizeLinear's x input
  // Input B is an initializer with data type 'uint8', we need to register it as 'uint4' WebNN constant
  const std::vector<uint32_t> x_shape{N, n_blocks_per_col, double_blob_size};
  emscripten::val x_shape_array = emscripten::val::array(x_shape);
  emscripten::val x_desc = emscripten::val::object();
  x_desc.set("dataType", emscripten::val("uint4"));
  x_desc.set("shape", x_shape_array);
  x_desc.set("dimensions", x_shape_array);
  emscripten::val dq_x = emscripten::val::undefined();
  const auto B_tensor = *initializers.at(input_defs[1]->Name());
  ORT_RETURN_IF_ERROR(model_builder.RegisterConstant(B_tensor, dq_x, x_desc, logger));

  // Prepare DequantizeLinear's x_scale input
  // DequantizeLinear's x_scale should be [N, n_blocks_per_col, 1], reshape scales to [N, n_blocks_per_col, 1]
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name() + "_reshape_scales");
  const std::vector<uint32_t> x_scale_shape{N, n_blocks_per_col, 1};
  emscripten::val x_scale_shape_array = emscripten::val::array(x_scale_shape);
  emscripten::val x_scale =
      model_builder.GetBuilder().call<emscripten::val>("reshape", scales, x_scale_shape_array, options);

  // Prepare DequantizeLinear's x_zero_point input
  // x_zero_point has the same shape as x_scale
  const bool has_zero_points = TensorExists(input_defs, 3);
  emscripten::val x_zero_point = emscripten::val::undefined();
  emscripten::val zero_points_desc = emscripten::val::object();
  zero_points_desc.set("dataType", emscripten::val("uint4"));
  zero_points_desc.set("shape", x_scale_shape_array);
  zero_points_desc.set("dimensions", x_scale_shape_array);
  if (has_zero_points) {
    // zero_points is an initializer with data type 'uint8', we need to register it as 'uint4' WebNN constant
    const auto zero_points_tensor = *initializers.at(input_defs[3]->Name());
    ORT_RETURN_IF_ERROR(model_builder.RegisterConstant(zero_points_tensor, x_zero_point, zero_points_desc, logger));
  } else {
    // zero_points' default value is 8, referred from CPU EP
    const int8_t default_zero_point = 8;
    // Always create a new WebNN constant for zero_points to facilitate MatMulNBits fusion in Chromium
    auto num_elements = (Product(x_scale_shape) + 1) / 2;
    emscripten::val default_zero_point_buffer = emscripten::val::global("Uint8Array").new_(num_elements);
    default_zero_point_buffer.call<void>("fill",
                                         emscripten::val(PackInt8ToUint8DoubledNibbles(
                                             default_zero_point, ONNX_NAMESPACE::TensorProto_DataType_UINT4)));
    x_zero_point =
        model_builder.GetBuilder().call<emscripten::val>("constant", zero_points_desc, default_zero_point_buffer);
  }

  // DequantizeLinear
  options.set("label", node.Name() + "_dequantizeLinear");
  emscripten::val dq =
      model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear", dq_x, x_scale, x_zero_point, options);

  // Reshape DequantizeLinear to [N, K]
  options.set("label", node.Name() + "_reshape_dequantizeLinear");
  const std::vector<uint32_t> new_dq_shape{N, K};
  emscripten::val new_dq_shape_array = emscripten::val::array(new_dq_shape);
  emscripten::val dq_reshaped =
      model_builder.GetBuilder().call<emscripten::val>("reshape", dq, new_dq_shape_array, options);

  // Transpose reshaped DequantizeLinear to [K, N]
  options.set("label", node.Name() + "_transpose_dequantizeLinear");
  emscripten::val dq_transposed = model_builder.GetBuilder().call<emscripten::val>("transpose", dq_reshaped, options);

  // MatMul
  options.set("label", node.Name() + "_matmul");
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("matmul", input, dq_transposed, options);

  // Add output with bias if present
  if (TensorExists(input_defs, 5)) {
    emscripten::val bias = model_builder.GetOperand(input_defs[5]->Name());
    options.set("label", node.Name() + "_add_bias");
    output = model_builder.GetBuilder().call<emscripten::val>("add", output, bias, options);
  }

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

bool MatMulNBitsBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                           const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  // Inputs B and zero_points (if present) must be initializers
  if (!graph_viewer.GetConstantInitializer(input_defs[1]->Name())) {  // B
    LOGS(logger, VERBOSE) << "Input B of MatMulNBits [" << name << "] must be known as initializer";
    return false;
  }
  if (TensorExists(input_defs, 3) && !graph_viewer.GetConstantInitializer(input_defs[3]->Name())) {  // zero_points
    LOGS(logger, VERBOSE) << "Input zero_points of MatMulNBits [" << name << "] must be known as initializer";
    return false;
  }

  // WebNN doesn't support g_idx input
  if (TensorExists(input_defs, 4)) {  // g_idx
    LOGS(logger, VERBOSE) << "Input g_idx of MatMulNBits [" << name << "] is not supported";
    return false;
  }

  NodeAttrHelper helper(node);
  if (helper.Get("bits", 4) != 4) {
    LOGS(logger, VERBOSE) << "Only 4-bit quantization is supported for MatMulNBits, additional bits support is planned";
  }

  return true;
}

bool MatMulNBitsBuilder::HasSupportedInputsImpl(const GraphViewer&,
                                                const Node& node, const emscripten::val& wnn_limits,
                                                const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();

  int32_t A_type = 0;
  int32_t B_type = 0;
  int32_t scales_type = 0;
  int32_t zero_points_type = 0;
  if (!GetType(*input_defs[0], A_type, logger) ||
      !GetType(*input_defs[1], B_type, logger) ||
      !GetType(*input_defs[2], scales_type, logger)) {
    return false;
  }

  const bool has_zero_points = TensorExists(input_defs, 3);
  if (has_zero_points && !GetType(*input_defs[3], zero_points_type, logger)) {
    return false;
  }

  InlinedVector<int32_t, 2> input_types = {A_type, scales_type};
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (A_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && A_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << "WebNN only supports float32 or float16 data type for input A of MatMulNBits";
    return false;
  }
  if (B_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS(logger, VERBOSE) << "WebNN only supports uint8 data type for input B of MatMulNBits";
    return false;
  }
  if (has_zero_points && zero_points_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS(logger, VERBOSE) << "WebNN only supports uint8 data type for input zero_points of MatMulNBits";
    return false;
  }

  // We only support 4-bit quantization, which is represented as the uint4 data type in WebNN.
  // Ensure that uint4 is supported.
  return IsDataTypeSupportedByOp("DequantizeLinear", ONNX_NAMESPACE::TensorProto_DataType_UINT4,
                                 wnn_limits, "input", "x", logger);
}

bool MatMulNBitsBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                 const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();

  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << "WebNN only supports float32 or float16 data type for output of MatMulNBits";
    return false;
  }

  return true;
}

void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<MatMulNBitsBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
