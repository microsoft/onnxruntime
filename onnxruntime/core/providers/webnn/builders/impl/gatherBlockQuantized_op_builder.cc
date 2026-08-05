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
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node, const WebnnDeviceType /* device_type */,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// The quantized data (T1) is one of {int4, uint4, uint8}. int4/uint4 map to WebNN's int4/uint4 dtypes
// directly. uint8 with bits==4 packs two 4-bit values per byte and has no matching WebNN dtype, so it is
// relabeled as a uint4 constant with the logical (unpacked) shape, leaving the raw bytes unchanged.
// uint8 with bits==8 stays as 8-bit.
static bool RequiresUint4Reinterpret(int32_t data_type, int64_t bits) {
  return data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 && bits == 4;
}

void GatherBlockQuantizedOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  // The GetType() helper takes a logger for diagnostics, but this callback isn't given one, so read
  // the declared element type straight from the proto instead.
  const int32_t data_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
  if (!RequiresUint4Reinterpret(data_type, helper.Get("bits", 4))) {
    return;
  }
  // The reinterpreted data/zero_point are registered as uint4 constants in AddToModelBuilderImpl,
  // so skip their default (uint8) registration to avoid a duplicate, oversized operand.
  model_builder.AddInitializerToSkip(input_defs[0]->Name());  // data
  if (TensorExists(input_defs, 3)) {
    model_builder.AddInitializerToSkip(input_defs[3]->Name());  // zero_points
  }
}

// WebNN doesn't provide a dedicated op for GatherBlockQuantizedOpBuilder, it can be simply
// decomposed by DequantizeLinear + Gather.
Status GatherBlockQuantizedOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> scales_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], scales_shape, logger), "Cannot get scales shape");
  const auto input_rank = input_shape.size();

  int32_t input_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input data type");

  NodeAttrHelper helper(node);
  const int64_t bits = helper.Get("bits", 4);
  const uint32_t gather_axis = SafeInt<uint32_t>(HandleNegativeAxis(helper.Get("gather_axis", 0), input_rank));
  const bool has_zero_points = TensorExists(input_defs, 3);
  const bool requires_reinterpret_u4 = RequiresUint4Reinterpret(input_type, bits);

  emscripten::val input = emscripten::val::undefined();
  emscripten::val zero_points = emscripten::val::undefined();

  if (requires_reinterpret_u4) {
    // Reinterpret the uint8-packed data/zero_point as WebNN uint4 constants. The raw bytes are
    // unchanged; only the element type and logical shape differ (2 uint4 values per byte). The data's
    // last (quantize) axis doubles; the zero_point takes the scales shape, which dequantizeLinear
    // requires to equal the scale shape.
    const auto& initializers = model_builder.GetInitializerTensors();
    auto register_u4 = [&](const std::string& name, const std::vector<int64_t>& logical_shape,
                           emscripten::val& out) -> Status {
      const std::vector<uint32_t> shape_u32 = GetNarrowedIntFromInt64<uint32_t>(logical_shape);
      emscripten::val desc = emscripten::val::object();
      desc.set("dataType", emscripten::val("uint4"));
      desc.set("shape", emscripten::val::array(shape_u32));
      desc.set("dimensions", emscripten::val::array(shape_u32));
      return model_builder.RegisterConstant(*initializers.at(name), out, desc, logger);
    };

    std::vector<int64_t> data_logical(input_shape);
    data_logical.back() *= 2;  // 8 / bits, with bits == 4
    ORT_RETURN_IF_ERROR(register_u4(input_defs[0]->Name(), data_logical, input));
    if (has_zero_points) {
      ORT_RETURN_IF_ERROR(register_u4(input_defs[3]->Name(), scales_shape, zero_points));
    }
  } else {
    input = model_builder.GetOperand(input_defs[0]->Name());
    if (has_zero_points) {
      zero_points = model_builder.GetOperand(input_defs[3]->Name());
    }
  }

  emscripten::val indices = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val scales = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val common_options = emscripten::val::object();

  // GatherBlockQuantized only supports block-wise quantization, the input and scales should have the same rank.
  // So we don't need to reshape scales for broadcasting.
  if (zero_points.isUndefined()) {
    // Create a default zero_point with the same shape as scales.
    const std::vector<uint32_t> zp_shape = GetNarrowedIntFromInt64<uint32_t>(scales_shape);
    if (requires_reinterpret_u4) {
      // Default zero_point for uint8-packed 4-bit is 8, stored as uint4 (two per byte, so 0x88).
      emscripten::val zp_desc = emscripten::val::object();
      zp_desc.set("dataType", emscripten::val("uint4"));
      zp_desc.set("shape", emscripten::val::array(zp_shape));
      zp_desc.set("dimensions", emscripten::val::array(zp_shape));
      emscripten::val zp_buffer = emscripten::val::global("Uint8Array").new_((Product(scales_shape) + 1) / 2);
      zp_buffer.call<void>("fill", emscripten::val(0x88));
      zero_points = model_builder.GetBuilder().call<emscripten::val>("constant", zp_desc, zp_buffer);
    } else {
      // Default zero_point is 128 for uint8, 0 for int4/uint4.
      const uint8_t default_zero_point = input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ? 128 : 0;
      zero_points = model_builder.CreateOrGetConstant<uint8_t>(input_type, default_zero_point, zp_shape);
    }
  }

  // dequantized_input = DequantizeLinear(input, scales, zero_points)
  common_options.set("label", node.Name() + "_dequantize_input");
  emscripten::val dequantized_input =
      model_builder.GetBuilder().call<emscripten::val>("dequantizeLinear", input, scales, zero_points, common_options);

  // output = Gather(dequantized_input, indices, axis=gather_axis)
  common_options.set("label", node.Name() + "_gather");
  common_options.set("axis", gather_axis);
  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("gather", dequantized_input, indices, common_options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool GatherBlockQuantizedOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                                      const WebnnDeviceType /* device_type */,
                                                      const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
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

  // uint8-packed 4-bit data is reinterpreted to uint4 as a zero-copy relabel of the same byte buffer.
  // WebNN reads that buffer as a flat row-major stream of 4-bit values (low nibble first), while ONNX
  // packs the two values sharing a byte along the quantize axis. Those two values are memory-adjacent
  // (and therefore correctly reinterpreted by doubling that axis) only when the quantize axis is the
  // innermost (last) axis; for an inner axis the packed pair is strided apart and the flat reinterpret
  // would interleave the wrong values. So restrict support to quantize_axis == last axis (else fallback).
  int32_t data_type = 0;
  if (!GetType(*input_defs[0], data_type, logger)) {
    return false;
  }
  if (RequiresUint4Reinterpret(data_type, bits)) {
    // The uint4 reinterpret path registers 'data' (and 'zero_points', if present) as WebNN constants
    // from their initializer bytes, so both must be constant initializers (else fallback).
    if (!graph_viewer.GetConstantInitializer(input_defs[0]->Name())) {  // data
      LOGS(logger, VERBOSE) << "GatherBlockQuantized: uint8-packed 4-bit 'data' must be a constant initializer.";
      return false;
    }
    if (TensorExists(input_defs, 3) && !graph_viewer.GetConstantInitializer(input_defs[3]->Name())) {  // zero_points
      LOGS(logger, VERBOSE) << "GatherBlockQuantized: uint8-packed 4-bit 'zero_points' must be a constant "
                               "initializer.";
      return false;
    }

    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger) || input_shape.empty()) {
      return false;
    }
    const int64_t quantize_axis = HandleNegativeAxis(helper.Get("quantize_axis", 1), input_shape.size());
    if (quantize_axis != static_cast<int64_t>(input_shape.size()) - 1) {
      LOGS(logger, VERBOSE) << "GatherBlockQuantized: uint8-packed 4-bit data is only supported when "
                               "quantize_axis is the last axis.";
      return false;
    }
  }

  return true;
}

bool GatherBlockQuantizedOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                           const emscripten::val& wnn_limits,
                                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> scales_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) || !GetShape(*input_defs[2], scales_shape, logger)) {
    return false;
  }

  if (input_shape.size() != scales_shape.size()) {
    LOGS(logger, VERBOSE) << "GatherBlockQuantized: input and scales must have the same rank.";
    return false;
  }

  const std::string_view op_type = node.OpType();
  int32_t input_type = 0;
  int32_t scales_type = 0;
  if (!GetType(*input_defs[0], input_type, logger) || !GetType(*input_defs[2], scales_type, logger)) {
    return false;
  }

  // For uint8-packed 4-bit data, dequantizeLinear actually receives a uint4 operand (the uint8 tensor
  // is reinterpreted), so validate the reinterpreted dtype rather than the declared one.
  NodeAttrHelper helper(node);
  int32_t dq_input_type = input_type;
  if (RequiresUint4Reinterpret(input_type, helper.Get("bits", 4))) {
    dq_input_type = ONNX_NAMESPACE::TensorProto_DataType_UINT4;
  }

  // Only need to check the input data type of ops that consume the inputs of GatherBlockQuantized.
  // WebNN dequantizeLinear's input should be same as input. WebNN gather's input should be same as scales input.
  return IsDataTypeSupportedByWebNNOp(op_type, "dequantizeLinear", dq_input_type, wnn_limits, "input", "data",
                                      logger) &&
         IsDataTypeSupportedByWebNNOp(op_type, "gather", scales_type, wnn_limits, "input", "scales", logger);
}

bool GatherBlockQuantizedOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
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
