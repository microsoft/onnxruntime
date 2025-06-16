// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

// WebNN doesn't provide a dedicated op for RotaryEmbedding. Instead, we implement it by using a
// combination of WebNN ops. The decomposed graph is referenced from DML EP at:
// onnxruntime/core/providers/dml/DmlExecutionProvider/src/Operators/DmlOperatorRotaryEmbedding.cpp
/*
                 Input                            CosCache   PositionIds     SinCache
                   |                                 |           |              |
                   |                                 |  +--------+-----------+  |
                 Split                               |  |                    |  |
                  |  |                              Gather                  Gather
          +-------+  |                                |                        |
          |          |                                |                        |
          |     Identity----------+                   |                        |
          |        |              |                   |                        |
          |        |              |                   |                        |
          |    --Split--          |                   |                        |
          |    \       /          | +-----------------+                        |
          |     \     /           | |                                          |
          |      \   /            Mul                                          |
          |       \ /              |                                           |
          |        X               |                                           |
          |       / \              |                                           |
          |      /   \             |                                           |
          |       Join             |                                           |
          |        |               |                                           |
          |        | +---------------------------------------------------------+
          |        | |             |
          |        Mul             |
          |         |              |
          |         +-----+ +------+
          |               | |
          |               Add
          |                |
          +-------------+  |
                        |  |
                        Join
*/
namespace onnxruntime {
namespace webnn {

class RotaryEmbeddingOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

Status RotaryEmbeddingOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t input_data_type;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_data_type, logger), "Cannot get input type");

  const bool is_onnx_domain = IsOnnxDomain(node.Domain());
  // The input indexes for the onnx domain and the microsoft domain are different.
  const size_t cos_cache_idx = is_onnx_domain ? 1 : 2;
  const size_t sin_cache_idx = is_onnx_domain ? 2 : 3;
  const size_t position_ids_idx = is_onnx_domain ? 3 : 1;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> position_ids_shape;
  std::vector<int64_t> cos_cache_shape;
  // Since opset 23, the position_ids input is optional.
  const bool has_position_ids = TensorExists(input_defs, position_ids_idx);
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[cos_cache_idx], cos_cache_shape, logger), "Cannot get cos_cache shape");
  if (has_position_ids) {
    ORT_RETURN_IF_NOT(GetShape(*input_defs[position_ids_idx], position_ids_shape, logger),
                      "Cannot get position_ids shape");
  }
  const bool input_is_4d = input_shape.size() == 4;
  // When position_ids is a 1D tensor, it represents the start offset for each sequence.
  const bool position_ids_is_offset = has_position_ids && position_ids_shape.size() == 1;

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val position_ids;
  if (has_position_ids) {
    position_ids = model_builder.GetOperand(input_defs[position_ids_idx]->Name());
  }
  emscripten::val cos_cache = model_builder.GetOperand(input_defs[cos_cache_idx]->Name());
  emscripten::val sin_cache = model_builder.GetOperand(input_defs[sin_cache_idx]->Name());

  const auto node_name = node.Name();
  emscripten::val wnn_builder = model_builder.GetBuilder();

  NodeAttrHelper helper(node);
  const bool interleaved = static_cast<bool>(helper.Get("interleaved", 0));
  uint32_t num_heads = helper.Get("num_heads", 0);
  uint32_t rotary_embedding_dim = helper.Get("rotary_embedding_dim", 0);

  // The input can be:
  // - 3D: [batch_size, sequence_length, hidden_size]
  // - 4D: [batch_size, num_heads, sequence_length, head_size]
  const uint32_t batch_size = static_cast<uint32_t>(input_shape[0]);
  uint32_t sequence_length, hidden_size, head_size;
  if (input_is_4d) {
    sequence_length = static_cast<uint32_t>(input_shape[2]);
    hidden_size = static_cast<uint32_t>(input_shape[1] * input_shape[3]);
    num_heads = static_cast<uint32_t>(input_shape[1]);
    head_size = static_cast<uint32_t>(input_shape[3]);
  } else {
    sequence_length = static_cast<uint32_t>(input_shape[1]);
    hidden_size = static_cast<uint32_t>(input_shape[2]);
    // Since opset 23, if the input is 3D, the num_heads must be provided.
    if (is_onnx_domain) {
      assert(num_heads != 0);
      head_size = hidden_size / num_heads;
    } else {
      if (num_heads == 0) {
        head_size = static_cast<uint32_t>(cos_cache_shape[1]) * 2;
        num_heads = hidden_size / head_size;
      } else {
        head_size = hidden_size / num_heads;
      }
    }
  }

  if (rotary_embedding_dim == 0) {
    rotary_embedding_dim = head_size;
  }

  const uint32_t half_rotary_embedding_dim = rotary_embedding_dim / 2;
  emscripten::val transpose_options = emscripten::val::object();

  // Ensure the input is reshaped to: [batch_size, sequence_length, num_heads, head_size].
  if (input_is_4d) {
    // The input is already in 4D shape, but we need to ensure the order is
    // [batch_size, sequence_length, num_heads, head_size] to make it broadcastable with
    // the coming mul operator with cos_cache and sin_cache.
    const std::vector<uint32_t> permutation{0, 2, 1, 3};
    transpose_options.set("label", node_name + "_transpose_input");
    transpose_options.set("permutation", emscripten::val::array(permutation));
    input = wnn_builder.call<emscripten::val>("transpose", input, transpose_options);
  } else {
    const std::vector<uint32_t> new_shape{batch_size, sequence_length, num_heads, head_size};
    emscripten::val reshape_input_options = emscripten::val::object();
    reshape_input_options.set("label", node_name + "_reshape_input");
    input = wnn_builder.call<emscripten::val>(
        "reshape", input, emscripten::val::array(new_shape), reshape_input_options);
  }

  // Split the input to perform the rotary embedding only on a subregion of the tensor if needed.
  // The split inputs will be joined back together at the end.
  emscripten::val partial_input0 = input;
  emscripten::val partial_input1 = emscripten::val::undefined();
  if (head_size != rotary_embedding_dim) {
    const std::vector<uint32_t> splits{rotary_embedding_dim, head_size - rotary_embedding_dim};
    emscripten::val split_input_options = emscripten::val::object();
    split_input_options.set("label", node_name + "_split_input");
    split_input_options.set("axis", 3);
    emscripten::val split = wnn_builder.call<emscripten::val>(
        "split", input, emscripten::val::array(splits), split_input_options);
    partial_input0 = split[0];
    partial_input1 = split[1];
  }

  // Split the partial input0 data into 2 equal parts.
  // Firstly reshape the partial input0.
  const std::vector<uint32_t> new_partial_input0_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, num_heads, half_rotary_embedding_dim, 2})
                  : std::vector<uint32_t>({batch_size, sequence_length, num_heads, 2, half_rotary_embedding_dim});
  emscripten::val reshape_partial_input0_options = emscripten::val::object();
  reshape_partial_input0_options.set("label", node_name + "_reshape_partial_input0");
  partial_input0 = wnn_builder.call<emscripten::val>(
      "reshape", partial_input0, emscripten::val::array(new_partial_input0_shape), reshape_partial_input0_options);
  // Split partial input0.
  const int split_axis = interleaved ? 4 : 3;
  emscripten::val split_partial_input0_options = emscripten::val::object();
  split_partial_input0_options.set("label", node_name + "_split_partial_input0");
  split_partial_input0_options.set("axis", split_axis);
  emscripten::val split_partial_input0 = wnn_builder.call<emscripten::val>(
      "split", partial_input0, 2, split_partial_input0_options);

  // Swap the two halves and join them together.
  emscripten::val concat_partial_input0_options = emscripten::val::object();
  concat_partial_input0_options.set("label", node_name + "_concat_partial_input0");
  emscripten::val concated_partial_input0 = wnn_builder.call<emscripten::val>(
      "concat", split_partial_input0.call<emscripten::val>("reverse"), split_axis, concat_partial_input0_options);

  if (position_ids_is_offset) {
    // We generate a sequence from 0 to sequence_length and add the offset to it.
    const std::vector<uint32_t> position_ids_range_shape = {1, sequence_length};
    std::string typed_array_name = "BigInt64Array";
    int position_ids_data_type = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    const bool is_int64_supported = model_builder.IsInt64Supported();
    if (!is_int64_supported) {
      // Int64 is not supported by current context, use int32 instead.
      typed_array_name = "Int32Array";
      position_ids_data_type = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }
    emscripten::val position_ids_range_buffer = emscripten::val::global(typed_array_name.c_str()).new_(sequence_length);
    for (uint32_t i = 0; i < sequence_length; i++) {
      position_ids_range_buffer.set(i, is_int64_supported ? emscripten::val::global("BigInt")(i) : emscripten::val(i));
    }
    emscripten::val position_ids_range_desc = emscripten::val::object();
    position_ids_range_desc.set("shape", emscripten::val::array(position_ids_range_shape));
    position_ids_range_desc.set("dimensions", emscripten::val::array(position_ids_range_shape));
    ORT_RETURN_IF_NOT(SetWebnnDataType(position_ids_range_desc, position_ids_data_type),
                      "WebNN backend does not support data type: ", position_ids_data_type);
    emscripten::val position_ids_range = wnn_builder.call<emscripten::val>(
        "constant", position_ids_range_desc, position_ids_range_buffer);
    // Add the offset to the sequence.
    emscripten::val position_ids_add_range_options = emscripten::val::object();
    position_ids_add_range_options.set("label", node_name + "_position_ids_add_range");
    position_ids = wnn_builder.call<emscripten::val>(
        "add", position_ids, position_ids_range, position_ids_add_range_options);
  }

  // Gather the cosine/sine values based on the position_ids (if it presents).
  emscripten::val gather_cos = cos_cache;
  emscripten::val gather_sin = sin_cache;
  if (has_position_ids) {
    emscripten::val gather_cos_sin_options = emscripten::val::object();
    gather_cos_sin_options.set("label", node_name + "_gather_cos_sin");
    gather_cos_sin_options.set("axis", 0);
    gather_cos = wnn_builder.call<emscripten::val>("gather", gather_cos, position_ids, gather_cos_sin_options);
    gather_sin = wnn_builder.call<emscripten::val>("gather", gather_sin, position_ids, gather_cos_sin_options);
  }

  // If it is full rotation, we need to slice the gathered cosine/sine
  // to get the shape [batch_size, sequence_length, rotary_embedding_dim / 2].
  if (cos_cache_shape.back() != static_cast<int64_t>(half_rotary_embedding_dim)) {
    emscripten::val slice_gather_cos_sin_options = emscripten::val::object();
    slice_gather_cos_sin_options.set("label", node_name + "_slice_gather_cos_sin");
    const std::vector<uint32_t> starts{0, 0, 0};
    const std::vector<uint32_t> sizes{batch_size, sequence_length, half_rotary_embedding_dim};
    gather_cos = wnn_builder.call<emscripten::val>("slice", gather_cos, emscripten::val::array(starts),
                                                   emscripten::val::array(sizes), slice_gather_cos_sin_options);
    gather_sin = wnn_builder.call<emscripten::val>("slice", gather_sin, emscripten::val::array(starts),
                                                   emscripten::val::array(sizes), slice_gather_cos_sin_options);
  }

  // Reshape and broadcast them to match the number of heads of the input data.
  const std::vector<uint32_t> reshaped_cos_sin_shape =
      interleaved ? std::vector<uint32_t>({batch_size, sequence_length, 1, half_rotary_embedding_dim, 1})
                  : std::vector<uint32_t>({batch_size, sequence_length, 1, 1, half_rotary_embedding_dim});
  emscripten::val reshape_gather_cos_sin_options = emscripten::val::object();
  reshape_gather_cos_sin_options.set("label", node_name + "_reshape_gather_cos_sin");
  gather_cos = wnn_builder.call<emscripten::val>(
      "reshape", gather_cos, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_cos_sin_options);
  gather_sin = wnn_builder.call<emscripten::val>(
      "reshape", gather_sin, emscripten::val::array(reshaped_cos_sin_shape), reshape_gather_cos_sin_options);

  // Multiply the non-rotated data with the cosine and the rotated data with the sine.
  emscripten::val mul_cos_options = emscripten::val::object();
  mul_cos_options.set("label", node_name + "_mul_cos");
  emscripten::val mul_cos = wnn_builder.call<emscripten::val>(
      "mul", partial_input0, gather_cos, mul_cos_options);
  emscripten::val mul_sin_options = emscripten::val::object();
  mul_sin_options.set("label", node_name + "_mul_sin");
  emscripten::val mul_sin = wnn_builder.call<emscripten::val>(
      "mul", concated_partial_input0, gather_sin, mul_sin_options);

  // Create a vector that contains the sign values {-1, 1}.
  emscripten::val sign_buffer = emscripten::val::undefined();
  const std::vector<uint32_t> sign_shape = interleaved ? std::vector<uint32_t>({1, 1, 1, 2})
                                                       : std::vector<uint32_t>({1, 1, 2, 1});
  emscripten::val sign_constant_desc = emscripten::val::object();
  sign_constant_desc.set("shape", emscripten::val::array(sign_shape));
  sign_constant_desc.set("dimensions", emscripten::val::array(sign_shape));
  ORT_RETURN_IF_NOT(SetWebnnDataType(sign_constant_desc, input_data_type),
                    "WebNN backend does not support data type: ", input_data_type);
  if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    sign_buffer = emscripten::val::global("Float32Array").new_(2);
    sign_buffer.set(0, -1.0f);
    sign_buffer.set(1, 1.0f);
  } else if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (model_builder.IsFloat16ArrayAvailable()) {
      // Float16Array is avaliable - use Float16Array.
      sign_buffer = emscripten::val::global("Float16Array").new_(2);
      sign_buffer.set(0, -1.0f);
      sign_buffer.set(1, 1.0f);
    } else {
      // Float16Array is not available - use Uint16Array instead.
      sign_buffer = emscripten::val::global("Uint16Array").new_(2);
      sign_buffer.set(0, PackFloat32ToUint16AsFloat16(-1.0f));
      sign_buffer.set(1, PackFloat32ToUint16AsFloat16(1.0f));
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type: ", input_data_type);
  }
  emscripten::val sign_constant = wnn_builder.call<emscripten::val>("constant", sign_constant_desc, sign_buffer);

  // Multiply the broadcasted sign values with the rotated input.
  emscripten::val mul_sign_options = emscripten::val::object();
  mul_sign_options.set("label", node_name + "_mul_sign");
  mul_sin = wnn_builder.call<emscripten::val>("mul", mul_sin, sign_constant, mul_sign_options);

  // Reshape mul_cos and mul_sin to (batch_size, sequence_length, num_heads, rotary_embedding_dim).
  const std::vector<uint32_t> reshaped_mul_cos_sin_shape =
      {batch_size, sequence_length, num_heads, rotary_embedding_dim};
  emscripten::val reshape_mul_cos_sin_options = emscripten::val::object();
  reshape_mul_cos_sin_options.set("label", node_name + "_reshape_mul_cos_sign");
  mul_cos = wnn_builder.call<emscripten::val>(
      "reshape", mul_cos, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);
  mul_sin = wnn_builder.call<emscripten::val>(
      "reshape", mul_sin, emscripten::val::array(reshaped_mul_cos_sin_shape), reshape_mul_cos_sin_options);

  // Add the multiplied cos and sin values together.
  emscripten::val add_mul_cos_sin_options = emscripten::val::object();
  add_mul_cos_sin_options.set("label", node_name + "_add_mul_cos_sin");
  emscripten::val output = wnn_builder.call<emscripten::val>(
      "add", mul_cos, mul_sin, add_mul_cos_sin_options);

  // Join the added values with the rest of the input.
  if (head_size != rotary_embedding_dim) {
    emscripten::val concat_back_input_options = emscripten::val::object();
    concat_back_input_options.set("label", node_name + "_concat_back_input");
    emscripten::val concat_inputs = emscripten::val::array();
    concat_inputs.call<void>("push", output);
    concat_inputs.call<void>("push", partial_input1);
    output = wnn_builder.call<emscripten::val>("concat", concat_inputs, 3, concat_back_input_options);
  }

  if (input_is_4d) {
    // The output is in 4D shape, we need to transpose it back to the original shape.
    // Reuse the transpose_options' permutation because the original permutation also
    // happens to be its own inverse. (inserve({0, 2, 1, 3} == {0, 2, 1, 3})
    transpose_options.set("label", node_name + "_transpose_output");
    output = wnn_builder.call<emscripten::val>("transpose", output, transpose_options);
  } else {
    // The output is in 3D shape, we need to reshape it back to the original shape.
    // The output shape is same as the input shape.
    const std::vector<uint32_t> output_shape = GetNarrowedIntfromInt64<uint32_t>(input_shape);
    emscripten::val reshape_output_options = emscripten::val::object();
    reshape_output_options.set("label", node_name + "_reshape_output");
    output = wnn_builder.call<emscripten::val>(
        "reshape", output, emscripten::val::array(output_shape), reshape_output_options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool RotaryEmbeddingOpBuilder::IsOpSupportedImpl(const GraphViewer&, const Node& node,
                                                 const WebnnDeviceType /* device_type */,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const bool is_onnx_domain = IsOnnxDomain(node.Domain());
  // The input indexes for the onnx domain and the microsoft domain are different.
  const size_t cos_cache_idx = is_onnx_domain ? 1 : 2;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> cos_cache_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) return false;
  if (!GetShape(*input_defs[cos_cache_idx], cos_cache_shape, logger)) return false;
  const auto input_size = input_shape.size();
  if (input_size != 3 && input_size != 4) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding only supports 3D or 4D input shape, input is " << input_size << "D shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const int is_packed_batching = helper.Get("is_packed_batching", 0);
  const int num_heads = helper.Get("num_heads", 0);
  const int rotary_embedding_dim = helper.Get("rotary_embedding_dim", 0);
  const auto sequence_length = input_size == 4 ? input_shape[2] : input_shape[1];

  if (is_onnx_domain) {
    if (input_size == 3 && num_heads == 0) {
      LOGS(logger, VERBOSE) << "RotaryEmbedding: num_heads must be provided if input is 3D";
      return false;
    }
  } else {
    if (is_packed_batching == 0 && sequence_length > cos_cache_shape[0]) {
      LOGS(logger, VERBOSE) << "RotaryEmbedding: updating cos_cache and sin_cache is not currently supported";
      return false;
    }

    if (rotary_embedding_dim > 0 && num_heads == 0) {
      LOGS(logger, VERBOSE) << "RotaryEmbedding: num_heads must be provided if rotary_embedding_dim is specified";
      return false;
    }
  }

  if (input_size == 4 && num_heads != 0 && num_heads != input_shape[1]) {
    LOGS(logger, VERBOSE) << "RotaryEmbedding: when input has 4 dimensions, num_heads must be 0 or have the same value "
                             "as the second dimension of the input";
    return false;
  }

  return true;
}

bool RotaryEmbeddingOpBuilder::HasSupportedInputsImpl(const GraphViewer&,
                                                      const Node& node,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  const bool is_onnx_domain = IsOnnxDomain(node.Domain());
  // The input indexes for the onnx domain and the microsoft domain are different.
  const size_t cos_cache_idx = is_onnx_domain ? 1 : 2;
  const size_t sin_cache_idx = is_onnx_domain ? 2 : 3;
  const size_t position_ids_idx = is_onnx_domain ? 3 : 1;
  int32_t input_type = 0;
  int32_t position_ids_type = 0;
  int32_t cos_cache_type = 0;
  int32_t sin_cache_type = 0;
  // Since opset 23, the position_ids is an optional input.
  const bool has_position_ids = TensorExists(input_defs, position_ids_idx);

  if (!GetType(*input_defs[0], input_type, logger) ||
      (has_position_ids && !GetType(*input_defs[position_ids_idx], position_ids_type, logger)) ||
      !GetType(*input_defs[cos_cache_idx], cos_cache_type, logger) ||
      !GetType(*input_defs[sin_cache_idx], sin_cache_type, logger)) {
    return false;
  }

  std::array<int32_t, 3> input_types{input_type, cos_cache_type, sin_cache_type};
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (has_position_ids && position_ids_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    return false;
  }

  // Check if the input data type is supported by each decomposed WebNN op.
  // Decomposed ops include: "Add", "Concat", "Gather", "Mul", "Reshape" and "Split".
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, input_type, wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  return true;
}

bool RotaryEmbeddingOpBuilder::HasSupportedOutputsImpl(const Node& node,
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
    const std::string_view webnn_output_name = webnn_op_type == "split" ? "outputs" : "output";
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, output_type, wnn_limits, webnn_output_name, "output", logger)) {
      return false;
    }
  }

  return true;
}

void CreateRotaryEmbeddingOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<RotaryEmbeddingOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
